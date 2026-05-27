# cluestering_patatune

Multi-objective optimisation of [CLUEstering](https://github.com/cms-patatrack/CLUEstering)
parameters for the HGCal detector using [patatune](https://github.com/cms-patatrack/patatune) MOPSO.

## What it does

Tunes 5 CLUEstering parameters independently for two HGCal subdetector regions:

| Region | z boundary |
|--------|-----------|
| **CEE** | \|z\| < 352 cm (electromagnetic) |
| **CHE** | \|z\| ≥ 352 cm (hadronic) |

Three objectives are minimised simultaneously (Pareto front):

| # | Name | Meaning |
|---|------|---------|
| F1 | `purity` | Mean worst-case RecoToSim score — fraction of trackster energy from wrong CP |
| F2 | `efficiency_deficit` | Mean `1 − min(CP efficiency)` — measures energy lost |
| F3 | `excess_tracksters` | Mean `max(0, N_tracksters − 2)` — penalises fragmentation |

Any parameter set producing fewer than 2 tracksters in any event is treated as
infeasible and receives `(inf, inf, inf)`.

## Quick start

```bash
python optimize.py
```

Loads both ROOT files, interleaves events, runs MOPSO for CEE then CHE, and writes:

```
pareto_CEE.csv / pareto_CHE.csv
pareto_CEE.json / pareto_CHE.json
pareto_CEE_positions.npy / pareto_CEE_fitnesses.npy
pareto_CHE_positions.npy / pareto_CHE_fitnesses.npy
```

## File layout

| File | Purpose |
|------|---------|
| `config.py` | Data paths, parameter bounds, MOPSO settings |
| `data.py` | ROOT loading, LC-to-shower assignment, event interleaving |
| `objective.py` | CLUEstering wrapper + F1/F2/F3 objective functions |
| `optimize.py` | Entry point — wires everything, runs MOPSO, saves results |
| `metrics.py` | *(unused — kept as stub)* |

## Parameters tuned (per subdetector)

```
density_radius    radius for local density estimation
min_density       minimum density to be a seed
outlier_distance  distance threshold to flag outliers
seeding_distance  distance threshold for seeding
w_z               weight applied to the z-axis coordinate
```

## Changing things

- **Data paths / file names** → `DATA_DIR`, `FILES` in `config.py`
- **Subdetector z boundary** → `CEE_Z_BOUNDARY` in `config.py`
- **Parameter search range** → `BOUNDS` in `config.py`
- **MOPSO budget** → `NUM_PARTICLES` / `NUM_ITERATIONS` in `config.py`
- **Objective definitions** → `objective.py`

## Dependencies

```
uproot      ROOT file reading
awkward     jagged array handling
numpy       numerical operations
pandas      CLUEstering data input
patatune    MOPSO optimisation
CLUEstering clustering algorithm
```
