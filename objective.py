"""
Objective function for CLUEstering parameter optimisation.

This is the primary file to edit when changing what "good clustering" means.

To add or change an objective:
  1. Modify _run_and_score() to compute additional per-event metrics.
  2. Update objective_fn inside make_objective_fn() to return the new values.
  3. Update the directions list in optimize.py to match ('minimize'/'maximize').

Current objectives (returned by objective_fn as [f1, f2, f3]):
  f1  purity_delta  — δ(L1) vs CLUE3D baseline; minimise
  f2  energy_ratio  — weighted mean r across particle types; maximise
  f3  count_delta   — δ(N_T) + floor penalty for N_T < 2; minimise

Parameters: x = [dc, rhoc, do, ds, wz]
Weights:    ALPHA (EM fraction) and LAMBDA (floor penalty) live in config.py
"""

import numpy as np
import pandas as pd
import CLUEstering as clue

from config import PARTICLE_TYPES, ALPHA, LAMBDA
from data import assign_lcs_to_particles
from metrics import compute_r2s


# ── CLUEstering wrapper ────────────────────────────────────────────────────────

def run_cluestering(xs, ys, zs, energies, dc, rhoc, do, ds, wz):
    """Run CLUEstering on one event's LC point cloud. Returns cluster_ids; -1 = outlier."""
    data = pd.DataFrame({
        'x0': xs, 'x1': ys, 'x2': wz * zs, 'weight': energies
    })
    c = clue.clusterer(float(dc), float(rhoc), float(do), float(ds))
    c.read_data(data)
    c.run_clue()
    return np.array(c.cluster_ids)


# ── Core scoring loop ──────────────────────────────────────────────────────────

def _delta(new_val, base_val, higher_is_better=False):
    """Signed relative change vs baseline. Negative = improvement over baseline."""
    if base_val == 0:
        return 0.0
    if higher_is_better:
        return (base_val - new_val) / base_val
    return (new_val - base_val) / base_val


def _run_and_score(params, raw, truth_lc_by_eid, lc_energy_lookup,
                   particle_types=None):
    """
    Run CLUEstering with *params* on every event and return mean metrics.

    Parameters
    ----------
    truth_lc_by_eid : dict
        Pre-grouped truth LCs: {global_event_id: DataFrame slice}.
        Build once with _group_truth_lcs(df_lc) before the optimisation loop.

    Returns
    -------
    scores : {'em': {'L1': float, 'r': float, 'NT': float}, 'pion': {...}}
    """
    if particle_types is None:
        particle_types = PARTICLE_TYPES

    dc, rhoc, do, ds, wz = params
    n_events   = min(len(raw[p]['truth']['vertices_indexes']) for p in particle_types)
    scores     = {p: {'L1': [], 'r': [], 'NT': []} for p in particle_types}
    global_eid = 0

    for event_idx in range(n_events):
        for ptype in particle_types:
            lc_energies = lc_energy_lookup.get(global_eid, {})

            ev = truth_lc_by_eid.get(global_eid)
            if ev is None or len(ev) == 0:
                global_eid += 1
                continue

            xs       = ev['x'].to_numpy()
            ys       = ev['y'].to_numpy()
            zs       = ev['z'].to_numpy()
            energies = ev['energy'].to_numpy()
            lc_idxs  = ev['lc_idx'].to_numpy()

            cluster_ids = run_cluestering(xs, ys, zs, energies, dc, rhoc, do, ds, wz)

            mask           = cluster_ids != -1
            lc_idxs_cl     = lc_idxs[mask]
            energies_cl    = energies[mask]
            cluster_ids_cl = cluster_ids[mask]
            unique_ids     = np.unique(cluster_ids_cl)
            N_T            = len(unique_ids)

            _, assigned   = assign_lcs_to_particles(raw[ptype]['truth'], event_idx)
            truth_lc_sets = {cp: set(lcs) for cp, lcs in assigned.items() if lcs}

            total_reco  = float(energies_cl.sum())
            total_truth = sum(lc_energies.values())
            r = min(total_reco / total_truth, 1.0) if total_truth > 0 else 0.0

            if N_T == 0 or not truth_lc_sets:
                L1 = 1.0
            else:
                weighted_sum, total_reco_e = 0.0, 0.0
                for cid in unique_ids:
                    sel      = cluster_ids_cl == cid
                    reco_set = set(lc_idxs_cl[sel])
                    e_reco   = float(energies_cl[sel].sum())
                    best_r2s = min(
                        compute_r2s(reco_set, sim_set, lc_energies)
                        for sim_set in truth_lc_sets.values()
                    )
                    weighted_sum  += e_reco * best_r2s
                    total_reco_e  += e_reco
                L1 = weighted_sum / total_reco_e if total_reco_e > 0 else 1.0

            scores[ptype]['L1'].append(L1)
            scores[ptype]['r'].append(r)
            scores[ptype]['NT'].append(N_T)
            global_eid += 1

    return {p: {k: float(np.mean(v)) for k, v in s.items()} for p, s in scores.items()}


def _group_truth_lcs(df_lc):
    """Pre-group truth LCs by global_event_id for O(1) lookup inside the scoring loop."""
    truth = df_lc[df_lc['source'] == 'truth']
    return {geid: grp for geid, grp in truth.groupby('global_event_id')}


# ── Public factory ─────────────────────────────────────────────────────────────

def make_objective_fn(raw, df_lc, lc_energy_lookup, baseline_metrics,
                      particle_types=None, alpha=None, lam=None):
    """
    Build the multi-objective callable for patatune.

    Returns
    -------
    objective_fn : callable
        x = [dc, rhoc, do, ds, wz]  →  [purity_delta, energy_ratio, count_delta]
        Directions: minimize, maximize, minimize
    """
    if particle_types is None:
        particle_types = PARTICLE_TYPES
    if alpha is None:
        alpha = ALPHA
    if lam is None:
        lam = LAMBDA

    weights          = {p: (alpha if p == 'em' else 1.0 - alpha) for p in particle_types}
    truth_lc_by_eid  = _group_truth_lcs(df_lc)

    def objective_fn(x):
        scores = _run_and_score(x, raw, truth_lc_by_eid, lc_energy_lookup, particle_types)

        # f1: purity improvement — minimise (negative = better than CLUE3D)
        d1 = sum(
            weights[p] * _delta(scores[p]['L1'], baseline_metrics[p]['L1'])
            for p in particle_types
        )

        # f2: energy recovery — maximise (raw ratio, patatune handles direction)
        r_combined = sum(weights[p] * scores[p]['r'] for p in particle_types)

        # f3: trackster count + floor penalty — minimise
        mean_NT = sum(weights[p] * scores[p]['NT'] for p in particle_types)
        d3 = sum(
            weights[p] * _delta(scores[p]['NT'], baseline_metrics[p]['NT'])
            for p in particle_types
        )
        penalty = lam * max(0.0, 2.0 - mean_NT) ** 2

        return [d1, r_combined, d3 + penalty]

    return objective_fn
