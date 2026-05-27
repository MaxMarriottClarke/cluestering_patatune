# HGCal CLUEstering Parameter Tuning with PATATUNE MOPSO

## Project Overview

This project tunes the 5 parameters of CLUEstering for 3 HGCal subdetector regions
(CEE, HSi, HSci) using multi-objective particle swarm optimisation (MOPSO) via PATATUNE.
The optimisation runs over mixed electron and electron-pion events, optimising 3 objectives
simultaneously without any weighting — the output is a Pareto front of non-dominated
parameter configurations.

### Parameters (per subdetector region)

```
density_radius     float   radius for local density estimation
min_density        float   minimum density to be a seed
outlier_distance   float   distance threshold to flag outliers
seeding_distance   float   distance threshold for seeding
w_z                float   weight applied to the z-axis coordinate
```

Three independent MOPSO runs are performed: one per subdetector region (CEE, HSi, HSci).
Each run optimises its own 5-parameter vector independently.

### Subdetector Regions

Layer clusters are assigned to a subdetector region based on their z position:

- **CEE** (electromagnetic): |z| < 352 cm  
- **CHE** (hadronic, silicon + scintillator): |z| >= 352 cm

> Note: Within CHE, HSi and HSci are distinguished by layer number or eta if needed,
> but for the purposes of this tuning the primary split is CEE vs CHE by z position.
> Refine this boundary if a more detailed subdetector map is available.

---

## Step 1: Data Loading and Preprocessing

### Input Files

- `ee_events.root` — 500 two-electron events
- `epion_events.root` — 500 electron-pion events

Both files contain the branch `simtrackstersCP;1` with truth information.

### 1.1 Load Truth Information from ROOT

Use `uproot` to load both files. Read the following arrays from `simtrackstersCP;1`
for every event:

```
vertices_x          array of arrays   x position of each layer cluster (LC)
vertices_y          array of arrays   y position of each LC
vertices_z          array of arrays   z position of each LC
vertices_energy     array of arrays   energy of each LC
vertices_multiplicity  array of arrays  1/multiplicity gives the fractional energy
                                        contribution of this LC to this simulated shower
vertices_indexes    array of arrays   global LC index within the event
```

Each outer array has one entry per simulated shower (CaloParticle / CP) in the event.
Each inner array has one entry per LC belonging to that shower.

### 1.2 Interleave Events from Both Files

After loading, interleave events so that the combined dataset alternates between
particle types:

```
[ee_event_0, epion_event_0, ee_event_1, epion_event_1, ...]
```

This ensures the MOPSO objective evaluations see a balanced mix throughout optimisation.
Keep a `particle_type` label per event (`"ee"` or `"epion"`) for downstream diagnostics.

### 1.3 Resolve Ambiguous LC Assignments

A single LC index can appear in multiple simulated showers within the same event
(shared/overlapping hits). This must be resolved before computing objectives.

**Resolution rule:** For each LC index that appears in more than one shower, assign it
exclusively to the shower for which `1 / multiplicity` is **largest** (i.e. the shower
that contributes the greatest fractional energy to that LC).

Implementation:

```python
for each event:
    build a dict: lc_index -> list of (shower_id, 1/multiplicity)
    for each lc_index with multiple showers:
        winning_shower = argmax over 1/multiplicity
        remove lc_index from all other showers
```

After this step, every LC index belongs to exactly one simulated shower per event.

### 1.4 Build the Per-Event Data Structure

After resolution, each event should be represented as:

```python
{
    "particle_type": str,           # "ee" or "epion"
    "sim_showers": [                # list of length = number of CPs (always 2)
        {
            "shower_id": int,
            "lc_indexes": np.ndarray,    # shape (N_lc,)
            "lc_x":       np.ndarray,
            "lc_y":       np.ndarray,
            "lc_z":       np.ndarray,
            "lc_energy":  np.ndarray,
            "true_energy": float,        # sum of lc_energy for this shower
        },
        ...
    ],
    "all_lcs": {                    # flat array of ALL LCs in the event
        "indexes":  np.ndarray,
        "x":        np.ndarray,
        "y":        np.ndarray,
        "z":        np.ndarray,
        "energy":   np.ndarray,
        "subdet":   np.ndarray,     # string or int: "CEE" or "CHE" per LC
    }
}
```

The `subdet` field is assigned per LC based on z position:

```python
subdet = "CEE" if abs(z) < 352.0 else "CHE"
```

The `all_lcs` dict is the input passed to CLUEstering. The `sim_showers` list is used
only for computing objectives after clustering.

### 1.5 Validation Checks

After preprocessing, assert:

- Every event has exactly 2 entries in `sim_showers`
- No LC index appears in more than one shower
- `true_energy` > 0 for all showers
- `subdet` values are only `"CEE"` or `"CHE"`

---

## Step 2: Objective Functions

The three objectives are evaluated **per subdetector region** on the set of reconstructed
tracksters produced by CLUEstering for that region's LCs.

All three objectives are to be **minimised**. Return `(inf, inf, inf)` for any parameter
configuration that produces fewer than 2 tracksters in any event — this is a hard
infeasibility constraint.

### 2.1 RecoToSim Score

Given the simplification that in our data each LC belongs to exactly one CP and is
reconstructed into exactly one trackster, the general score formula (eq. 4.1) reduces
to:

```
score(trackster_i, CP_j) = (energy of LCs in trackster_i NOT from CP_j)
                           / (total energy of trackster_i)
```

i.e. the fraction of the trackster's energy that is impure (not from the best-matching CP).

Implementation:

```python
def reco_to_sim_score(trackster_lc_indexes, trackster_lc_energies,
                      cp_lc_index_set):
    """
    trackster_lc_indexes: array of LC indexes in this trackster
    trackster_lc_energies: corresponding energies
    cp_lc_index_set: set of LC indexes belonging to candidate CP j
    Returns: scalar score in [0, 1]. 0 = perfectly pure, 1 = fully impure.
    """
    total_energy = trackster_lc_energies.sum()
    if total_energy == 0:
        return 0.0
    impure_energy = sum(
        e for idx, e in zip(trackster_lc_indexes, trackster_lc_energies)
        if idx not in cp_lc_index_set
    )
    return impure_energy / total_energy
```

### 2.2 Best-CP Assignment

For each trackster, find the CP that minimises its RecoToSim score:

```python
def assign_tracksters_to_cps(tracksters, sim_showers):
    """
    Returns: dict mapping trackster_id -> (best_cp_id, score)
    """
    assignments = {}
    for t_id, trackster in enumerate(tracksters):
        best_cp, best_score = None, np.inf
        for cp in sim_showers:
            cp_set = set(cp["lc_indexes"])
            s = reco_to_sim_score(trackster["lc_indexes"],
                                  trackster["lc_energies"], cp_set)
            if s < best_score:
                best_score = s
                best_cp = cp["shower_id"]
        assignments[t_id] = (best_cp, best_score)
    return assignments
```

### 2.3 Objective 1 — Worst-Case Purity (Minimise)

```
F1 = mean over events of [ max over tracksters of score(trackster, best_CP) ]
```

For each event, take the **maximum** (worst) RecoToSim score across all reconstructed
tracksters. Then average across all events.

```python
def objective_purity(events_results):
    per_event = []
    for result in events_results:
        if result["infeasible"]:
            return np.inf
        worst = max(assignment["score"] for assignment in result["assignments"].values())
        per_event.append(worst)
    return np.mean(per_event)
```

### 2.4 Objective 2 — Worst-Case Efficiency (Minimise)

For each CP j, compute the fraction of its true energy recovered by all tracksters
assigned to it:

```
efficiency_j = sum_{tracksters assigned to CP_j} E_shared(trackster, CP_j)
               / E_true_j
```

where `E_shared(trackster, CP_j)` is the sum of energies of LCs in the trackster that
also belong to CP_j.

For each event, take the **minimum** efficiency across the 2 CPs (worst-case).
Then average the efficiency deficit `1 - min_efficiency` across events:

```
F2 = mean over events of [ 1 - min_{j in {CP1, CP2}} efficiency_j ]
```

```python
def shared_energy(trackster, cp):
    cp_set = set(cp["lc_indexes"])
    return sum(
        e for idx, e in zip(trackster["lc_indexes"], trackster["lc_energies"])
        if idx in cp_set
    )

def objective_efficiency(events_results, sim_showers_per_event):
    per_event = []
    for result, sim_showers in zip(events_results, sim_showers_per_event):
        if result["infeasible"]:
            return np.inf
        cp_shared = {cp["shower_id"]: 0.0 for cp in sim_showers}
        for t_id, trackster in enumerate(result["tracksters"]):
            best_cp_id = result["assignments"][t_id][0]
            cp = next(cp for cp in sim_showers if cp["shower_id"] == best_cp_id)
            cp_shared[best_cp_id] += shared_energy(trackster, cp)
        efficiencies = [
            cp_shared[cp["shower_id"]] / cp["true_energy"]
            for cp in sim_showers
        ]
        per_event.append(1.0 - min(efficiencies))
    return np.mean(per_event)
```

### 2.5 Objective 3 — Excess Tracksters (Minimise)

```
F3 = mean over events of [ max(0, N_tracksters - 2) ]
```

Any event with N_tracksters < 2 is **infeasible** — return np.inf for all objectives.

```python
def objective_trackster_count(events_results):
    per_event = []
    for result in events_results:
        if result["infeasible"]:
            return np.inf
        per_event.append(max(0, result["n_tracksters"] - 2))
    return np.mean(per_event)
```

### 2.6 Combined Objective Wrapper

The PATATUNE objective function takes a 5-element parameter vector for one subdetector
region, runs CLUEstering on all events for that region with those parameters, computes
all three objectives, and returns them as a tuple:

```python
def make_objective(subdet, events, sim_showers_per_event):
    """
    subdet: "CEE" or "CHE"
    events: preprocessed event list
    Returns: callable params -> (F1, F2, F3)
    """
    def objective(params):
        density_radius, min_density, outlier_distance, seeding_distance, w_z = params

        events_results = []
        for event, sim_showers in zip(events, sim_showers_per_event):
            lcs = filter_lcs_by_subdet(event["all_lcs"], subdet)
            if len(lcs["indexes"]) == 0:
                # no LCs in this subdetector for this event — skip
                continue

            tracksters = run_cluestering(
                lcs, density_radius, min_density,
                outlier_distance, seeding_distance, w_z
            )
            n = len(tracksters)
            infeasible = n < 2
            assignments = {} if infeasible else assign_tracksters_to_cps(
                tracksters, sim_showers
            )
            events_results.append({
                "infeasible": infeasible,
                "n_tracksters": n,
                "tracksters": tracksters,
                "assignments": assignments,
            })

        if any(r["infeasible"] for r in events_results):
            return (np.inf, np.inf, np.inf)

        f1 = objective_purity(events_results)
        f2 = objective_efficiency(events_results, sim_showers_per_event)
        f3 = objective_trackster_count(events_results)
        return (f1, f2, f3)

    return objective
```

---

## Step 3: MOPSO Optimisation and Results

### 3.1 Parameter Bounds

Set physically motivated bounds for each of the 5 parameters. Adjust these based on
prior knowledge of the detector geometry and typical cluster separations:

```python
BOUNDS = {
    "CEE": {
        "lower": [0.5,  0.5,  0.5,  0.5,  0.1],
        "upper": [5.0, 10.0, 10.0, 10.0,  5.0],
    },
    "CHE": {
        "lower": [0.5,  0.5,  0.5,  0.5,  0.1],
        "upper": [8.0, 15.0, 15.0, 15.0,  5.0],
    },
}
PARAM_NAMES = ["density_radius", "min_density", "outlier_distance",
               "seeding_distance", "w_z"]
```

### 3.2 Running MOPSO per Subdetector

Run one independent MOPSO instance per subdetector region:

```python
import patatune
import numpy as np

results = {}

for subdet in ["CEE", "CHE"]:
    print(f"Optimising {subdet}...")

    obj_fn = make_objective(subdet, events, sim_showers_per_event)

    f1 = lambda p: obj_fn(p)[0]
    f2 = lambda p: obj_fn(p)[1]
    f3 = lambda p: obj_fn(p)[2]

    objectives = patatune.ElementWiseObjective([f1, f2, f3])

    lb = BOUNDS[subdet]["lower"]
    ub = BOUNDS[subdet]["upper"]

    mopso = patatune.MOPSO(
        objectives,
        lower_bounds=lb,
        upper_bounds=ub,
        num_particles=50,
        inertia_weight=0.4,
        cognitive_coefficient=1.5,
        social_coefficient=2.0,
    )

    pareto = mopso.optimize(num_iterations=200)
    results[subdet] = pareto
    print(f"  {len(pareto)} Pareto-optimal solutions found for {subdet}")
```

> Note: Calling the objective function three times per particle evaluation (once per
> objective) is wasteful since all three are computed together. If PATATUNE supports a
> vector objective interface, use it to call CLUEstering only once per evaluation.
> Check the PATATUNE docs for `VectorObjective` or equivalent.

### 3.3 Saving Results

Save the full Pareto archive for each subdetector in multiple formats for easy analysis.

#### 3.3.1 CSV — human-readable Pareto front

```python
import pandas as pd

for subdet, pareto in results.items():
    rows = []
    for p in pareto:
        row = dict(zip(PARAM_NAMES, p.position))
        row["F1_purity"]     = p.fitness[0]
        row["F2_efficiency"] = p.fitness[1]
        row["F3_n_tracksters"] = p.fitness[2]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(f"pareto_{subdet}.csv", index=False)
    print(f"Saved pareto_{subdet}.csv ({len(df)} solutions)")
```

#### 3.3.2 JSON — full structured archive

```python
import json

for subdet, pareto in results.items():
    archive = [
        {
            "params": dict(zip(PARAM_NAMES, p.position.tolist())),
            "objectives": {
                "F1_purity": p.fitness[0],
                "F2_efficiency": p.fitness[1],
                "F3_n_tracksters": p.fitness[2],
            }
        }
        for p in pareto
    ]
    with open(f"pareto_{subdet}.json", "w") as f:
        json.dump(archive, f, indent=2)
```

#### 3.3.3 NumPy arrays — for plotting and further analysis

```python
for subdet, pareto in results.items():
    positions = np.array([p.position for p in pareto])
    fitnesses = np.array([p.fitness  for p in pareto])
    np.save(f"pareto_{subdet}_positions.npy", positions)
    np.save(f"pareto_{subdet}_fitnesses.npy",  fitnesses)
```

### 3.4 Diagnostics to Compute After Optimisation

For each saved Pareto front, compute and save the following per-particle-type breakdown
to verify neither `ee` nor `epion` events dominate the combined objective:

```python
for subdet in ["CEE", "CHE"]:
    for solution in pareto_solutions[subdet]:
        params = solution["params"]
        for ptype in ["ee", "epion"]:
            subset_events = [e for e in events if e["particle_type"] == ptype]
            # re-evaluate objectives on subset only
            # save F1, F2, F3 per particle type
```

This lets you check post-hoc whether the combined optimisation produced parameters
that are balanced across both particle types.

### 3.5 Suggested Convergence Metrics

PATATUNE provides built-in metrics. Log these per iteration if checkpointing is enabled:

- **Hypervolume indicator** — volume of objective space dominated by the Pareto archive
  (higher = better coverage of the front)
- **Generational Distance (GD)** — average distance of archive solutions from a reference
  front (lower = better)
- **Inverted Generational Distance (IGD)** — average distance from reference front to
  nearest archive solution (lower = better)

Use `FileManager` for checkpointing so long runs can be resumed:

```python
fm = patatune.FileManager(directory=f"checkpoints_{subdet}/", format="pickle")
mopso = patatune.MOPSO(..., file_manager=fm)
pareto = mopso.optimize(num_iterations=200)
```

---

## Implementation Notes

### CLUEstering Interface

The `run_cluestering` function should wrap the CLUEstering Python API. The input
coordinates passed to the clusterer must apply `w_z` to the z coordinate:

```python
from CLUEstering import clusterer as Clusterer

def run_cluestering(lcs, density_radius, min_density,
                    outlier_distance, seeding_distance, w_z):
    x = lcs["x"]
    y = lcs["y"]
    z = lcs["z"] * w_z      # apply z weight here
    energy = lcs["energy"]

    clue = Clusterer(
        density_radius=density_radius,
        min_density=min_density,
        outlier_distance=outlier_distance,
        seeding_distance=seeding_distance,
    )
    clue.read_data(...)      # fill in with correct CLUEstering API call
    clue.run_clue()

    # extract tracksters: list of dicts with lc_indexes, lc_energies
    tracksters = extract_tracksters(clue, lcs)
    return tracksters
```

> Verify the correct CLUEstering API for reading data and extracting cluster labels
> from the existing codebase before adapting this wrapper.

### Performance

Each MOPSO iteration evaluates `num_particles` objective function calls. With 50
particles, 200 iterations, and 1000 events per evaluation this is 10 million event
clusterings per subdetector. Consider:

- Reducing to a representative subset of events (e.g. 100-200) during optimisation,
  then validating the final Pareto front on the full 1000
- Parallelising particle evaluations if PATATUNE supports it, or using
  `multiprocessing` around the objective function
- Caching event preprocessing so it is done once before optimisation begins

### Dependencies

```
uproot         ROOT file reading
awkward        jagged array handling
numpy          numerical operations
pandas         CSV output
patatune       MOPSO optimisation
CLUEstering    clustering algorithm
```
