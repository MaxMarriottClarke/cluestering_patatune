"""
Objective functions for CLUEstering parameter tuning.

CLUEstering is run independently on CEE and CHE LCs (different detector geometry
requires different parameters), but quality is scored globally across both
subdetectors — a trackster is just a trackster regardless of which region it
came from.

Three objectives — all minimised:

  F1  purity      mean over events of [max RecoToSim score across all tracksters]
                  0 = every trackster is perfectly pure
                  1 = every trackster is entirely contaminated

  F2  efficiency  mean over events of [1 - min(CP efficiency across both CPs)]
                  0 = both CPs fully recovered
                  1 = at least one CP entirely lost

  F3  fragmentation  mean over events of [extra tracksters assigned to same CP]
                  = sum over CPs of max(0, N_tracksters_assigned - 1)
                  0 = each CP has exactly one trackster (ideal)
                  1+ = CPs are being split — e.g. a pion's CEE and CHE portions
                       are two separate tracksters both assigned to the same CP

  Note on F3: a pion event naturally produces 3 tracksters (1 electron in CEE,
  1 pion-CEE portion, 1 pion-CHE portion).  Penalising N_total > 2 would punish
  a perfect reconstruction.  Fragmentation counts extra tracksters *per CP*,
  which is zero for that 3-trackster case if each is cleanly assigned.

Infeasibility: if total tracksters (CEE + CHE combined) < 2 in any event,
               return (inf, inf, inf).

Public entry point:
    make_objective(events) -> callable(params[10]) -> [F1, F2, F3]
    params[:5] = [density_radius, min_density, outlier_distance,
                  seeding_distance, w_z]  for CEE
    params[5:] = same five parameters                             for CHE
"""

import numpy as np
import pandas as pd

try:
    import CLUEstering as clue
except ImportError:
    clue = None   # raises a clear error at runtime if clustering is attempted

from config import SUBDETS


# ── CLUEstering wrapper ────────────────────────────────────────────────────────

def run_cluestering(lcs, density_radius, min_density,
                    outlier_distance, seeding_distance, w_z):
    """
    Run CLUEstering on one subdetector's LC point cloud.

    Parameters
    ----------
    lcs : dict with keys 'indexes', 'x', 'y', 'z', 'energy'
        Must be non-empty.

    Returns
    -------
    tracksters : list of dicts, each with 'lc_indexes' and 'lc_energies'
    """
    if clue is None:
        raise ImportError("CLUEstering is not installed in this environment")

    data = pd.DataFrame({
        'x0':     lcs['x'],
        'x1':     lcs['y'],
        'x2':     lcs['z'] * float(w_z),   # z axis scaled by w_z
        'weight': lcs['energy'],
    })

    c = clue.clusterer(
        float(density_radius),
        float(min_density),
        float(outlier_distance),
        float(seeding_distance),
    )
    c.read_data(data)
    c.run_clue()

    cluster_ids = np.array(c.cluster_ids)
    return _extract_tracksters(cluster_ids, lcs)


def _extract_tracksters(cluster_ids, lcs):
    """Convert cluster_id array to list of trackster dicts. Outliers (id -1) are dropped."""
    tracksters = []
    for cid in np.unique(cluster_ids):
        if cid == -1:
            continue
        mask = cluster_ids == cid
        tracksters.append({
            'lc_indexes':  lcs['indexes'][mask],
            'lc_energies': lcs['energy'][mask],
        })
    return tracksters


def filter_lcs_by_subdet(all_lcs, subdet):
    """Return the subset of all_lcs whose subdet label matches."""
    mask = all_lcs['subdet'] == subdet
    return {
        'indexes': all_lcs['indexes'][mask],
        'x':       all_lcs['x'][mask],
        'y':       all_lcs['y'][mask],
        'z':       all_lcs['z'][mask],
        'energy':  all_lcs['energy'][mask],
    }


# ── Scoring primitives ─────────────────────────────────────────────────────────

def reco_to_sim_score(trackster_lc_indexes, trackster_lc_energies, cp_lc_index_set):
    """
    Fraction of trackster energy that does NOT come from CP j.
    0 = perfectly pure,  1 = fully contaminated.
    cp_lc_index_set contains LC indexes from *both* CEE and CHE for this CP.
    """
    total = float(np.asarray(trackster_lc_energies).sum())
    if total == 0:
        return 0.0
    impure = sum(
        float(e)
        for idx, e in zip(trackster_lc_indexes, trackster_lc_energies)
        if idx not in cp_lc_index_set
    )
    return impure / total


def assign_tracksters_to_cps(tracksters, sim_showers):
    """
    Assign each trackster to the CP that minimises its RecoToSim score.
    CP LC index sets span both CEE and CHE — the trackster just needs to
    share LCs with the right CP regardless of subdetector.

    Returns
    -------
    assignments : dict  trackster_idx -> (best_cp_id, score)
    """
    assignments = {}
    for t_id, trackster in enumerate(tracksters):
        best_cp, best_score = None, np.inf
        for cp in sim_showers:
            cp_set = set(cp['lc_indexes'].tolist())
            s = reco_to_sim_score(
                trackster['lc_indexes'], trackster['lc_energies'], cp_set
            )
            if s < best_score:
                best_score = s
                best_cp    = cp['shower_id']
        assignments[t_id] = (best_cp, best_score)
    return assignments


def _shared_energy(trackster, cp):
    """Energy of LCs in trackster that also belong to cp (any subdetector)."""
    cp_set = set(cp['lc_indexes'].tolist())
    return float(sum(
        float(e)
        for idx, e in zip(trackster['lc_indexes'], trackster['lc_energies'])
        if idx in cp_set
    ))


# ── Objective sub-functions ────────────────────────────────────────────────────

def _objective_purity(events_results):
    """
    F1: mean over events of max RecoToSim score across all tracksters.
    Tracksters from CEE and CHE are treated identically.
    """
    per_event = []
    for result in events_results:
        if result['infeasible']:
            return np.inf
        worst = max(score for _, score in result['assignments'].values())
        per_event.append(worst)
    return float(np.mean(per_event))


def _objective_efficiency(events_results):
    """
    F2: mean over events of (1 - min CP efficiency).

    CP efficiency = total energy recovered across ALL tracksters assigned to it
                    (summing CEE and CHE tracksters together)
                    / CP true_energy (which already spans both subdetectors).
    """
    per_event = []
    for result in events_results:
        if result['infeasible']:
            return np.inf

        sim_showers = result['sim_showers']
        tracksters  = result['tracksters']
        assignments = result['assignments']

        cp_shared = {cp['shower_id']: 0.0 for cp in sim_showers}
        for t_id, trackster in enumerate(tracksters):
            best_cp_id = assignments[t_id][0]
            cp = next(cp for cp in sim_showers if cp['shower_id'] == best_cp_id)
            cp_shared[best_cp_id] += _shared_energy(trackster, cp)

        efficiencies = [
            cp_shared[cp['shower_id']] / cp['true_energy']
            if cp['true_energy'] > 0 else 0.0
            for cp in sim_showers
        ]
        per_event.append(1.0 - min(efficiencies))

    return float(np.mean(per_event))


def _objective_fragmentation(events_results):
    """
    F3: mean over events of [sum over CPs of max(0, N_assigned - 1)].

    Counts how many extra tracksters are assigned to each CP beyond the first.
    This is zero when each CP has exactly one trackster — regardless of whether
    that means 2 total (ee event) or 3 total (pion with CEE+CHE components,
    each cleanly assigned to separate CPs).
    """
    per_event = []
    for result in events_results:
        if result['infeasible']:
            return np.inf
        cp_counts = {}
        for _, (best_cp_id, _) in result['assignments'].items():
            cp_counts[best_cp_id] = cp_counts.get(best_cp_id, 0) + 1
        excess = sum(max(0, count - 1) for count in cp_counts.values())
        per_event.append(float(excess))
    return float(np.mean(per_event))


# ── Public factory ─────────────────────────────────────────────────────────────

def make_objective(events):
    """
    Build the multi-objective callable for joint CEE+CHE optimisation.

    Parameters
    ----------
    events : list of per-event dicts from data.load_events()

    Returns
    -------
    objective_fn : callable
        params[0:5]  = [density_radius, min_density, outlier_distance,
                        seeding_distance, w_z]  for CEE
        params[5:10] = same five parameters      for CHE
        returns [F1, F2, F3] — all three minimised
    """
    def objective_fn(params):
        cee_params = params[:5]   # density_radius, min_density, outlier_distance,
        che_params = params[5:]   # seeding_distance, w_z  — same layout for CHE

        events_results = []
        for event in events:
            # Run CLUEstering separately in each subdetector
            tracksters = []
            for subdet, subdet_params in zip(SUBDETS, [cee_params, che_params]):
                lcs = filter_lcs_by_subdet(event['all_lcs'], subdet)
                if len(lcs['indexes']) == 0:
                    continue   # this subdetector has no LCs in this event
                tracksters.extend(
                    run_cluestering(lcs, *subdet_params)
                )

            # Score globally — tracksters from CEE and CHE pooled together
            n          = len(tracksters)
            infeasible = n < 2

            assignments = (
                {}
                if infeasible
                else assign_tracksters_to_cps(tracksters, event['sim_showers'])
            )

            events_results.append({
                'infeasible':   infeasible,
                'n_tracksters': n,
                'tracksters':   tracksters,
                'assignments':  assignments,
                'sim_showers':  event['sim_showers'],
            })

        if not events_results or any(r['infeasible'] for r in events_results):
            return [np.inf, np.inf, np.inf]

        f1 = _objective_purity(events_results)
        f2 = _objective_efficiency(events_results)
        f3 = _objective_fragmentation(events_results)
        return [f1, f2, f3]

    return objective_fn
