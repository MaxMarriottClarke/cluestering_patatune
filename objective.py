"""
Objective functions for CLUEstering parameter tuning.

CLUEstering is run independently on CEE and CHE LCs (different detector geometry
requires different parameters), but quality is scored globally across both
subdetectors — a trackster is just a trackster regardless of which region it
came from.

Three objectives — all minimised:

  F1  purity        mean over events of [max RecoToSim score across all tracksters]
  F2  efficiency    mean over events of [1 - min(CP efficiency across CEE+CHE)]
  F3  fragmentation mean over events of [max(0, N_total_tracksters - 2)]

Infeasibility: total tracksters (CEE + CHE) < 2 in any event → (inf, inf, inf).

Parallelism:
  make_objective(events, n_jobs=1) returns a callable.
  With n_jobs > 1, each objective call parallelises event evaluation across
  n_jobs worker processes using multiprocessing (fork).  Workers inherit the
  shared event list without any pickling overhead — only the small
  (event_index, params) tuples are sent over the wire.

Public entry point:
    make_objective(events, n_jobs=1) -> callable(params[10]) -> [F1, F2, F3]
    params[:5]  = [density_radius, min_density, outlier_distance,
                   seeding_distance, w_z]  for CEE
    params[5:]  = same five parameters      for CHE
"""

import multiprocessing as mp
import numpy as np
import pandas as pd

from config import SUBDETS, CLUE3D_BASELINES



_EVENTS = None   # set by make_objective before pool creation



def run_cluestering(lcs, density_radius, min_density,
                    outlier_distance, seeding_distance, w_z):
    try:
        import CLUEstering as clue
    except ImportError:
        raise ImportError("CLUEstering is not installed in this environment")

    data = pd.DataFrame({
        'x0':     lcs['x'],
        'x1':     lcs['y'],
        'x2':     lcs['z'] * float(w_z),
        'weight': lcs['energy'],
    })
    c = clue.clusterer(
        float(density_radius), float(min_density),
        float(outlier_distance), float(seeding_distance),
    )
    c.read_data(data)
    c.run_clue()
    return _extract_tracksters(np.array(c.cluster_ids), lcs)


def _extract_tracksters(cluster_ids, lcs):
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
    mask = all_lcs['subdet'] == subdet
    return {
        'indexes': all_lcs['indexes'][mask],
        'x':       all_lcs['x'][mask],
        'y':       all_lcs['y'][mask],
        'z':       all_lcs['z'][mask],
        'energy':  all_lcs['energy'][mask],
    }



def _eval_event(args):
    """
    Evaluate one event.  Must be a module-level function to be picklable by
    multiprocessing.  Reads from _EVENTS (inherited via fork, no pickling).

    args : (event_index, cee_params_list, che_params_list)
    """
    idx, cee_params, che_params = args
    event = _EVENTS[idx]

    tracksters = []
    for subdet, subdet_params in zip(SUBDETS, [cee_params, che_params]):
        lcs = filter_lcs_by_subdet(event['all_lcs'], subdet)
        if len(lcs['indexes']) == 0:
            continue
        new_tracksters = run_cluestering(lcs, *subdet_params)
        for t in new_tracksters:
            t['subdet'] = subdet
        tracksters.extend(new_tracksters)

    n          = len(tracksters)
    infeasible = n < 2
    assignments = (
        {} if infeasible
        else assign_tracksters_to_cps(tracksters, event['sim_showers'])
    )
    return {
        'infeasible':    infeasible,
        'n_tracksters':  n,
        'tracksters':    tracksters,
        'assignments':   assignments,
        'sim_showers':   event['sim_showers'],
        'particle_type': event['particle_type'],
    }



def reco_to_sim_score(trackster_lc_indexes, trackster_lc_energies, cp_lc_index_set):
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
    cp_set = set(cp['lc_indexes'].tolist())
    return float(sum(
        float(e)
        for idx, e in zip(trackster['lc_indexes'], trackster['lc_energies'])
        if idx in cp_set
    ))



def _objective_purity(events_results):
    per_event = []
    for result in events_results:
        if result['infeasible']:
            return np.inf
        worst = max(score for _, score in result['assignments'].values())
        per_event.append(worst)
    return float(np.mean(per_event))


def _objective_efficiency(events_results):
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
    """F3: total tracksters beyond the 2 input particles per event."""
    per_event = []
    for result in events_results:
        if result['infeasible']:
            return np.inf
        per_event.append(float(max(0, result['n_tracksters'] - 2)))
    return float(np.mean(per_event))


# ── Public factory ─────────────────────────────────────────────────────────────

def make_objective(events, n_jobs=1):
    """
    Build the 6-objective callable for joint CEE+CHE optimisation.

    Objectives are computed separately for ee and ππ events, then each is
    divided by the corresponding pre-computed CLUE3D baseline so that values
    < 1 mean "better than CLUE3D".  CLUE3D maps to the point (1,1,1,1,1,1).

    Parameters
    ----------
    events : list of per-event dicts from data.load_events()
    n_jobs : int
        Number of worker processes.

    Returns
    -------
    objective_fn : callable
        params[0:5]  CEE parameters
        params[5:10] CHE parameters
        returns [R1_ee, R2_ee, R3_ee, R1_pi, R2_pi, R3_pi]
        where R_i_t = F_i_t / CLUE3D_F_i_t  (ratio to CLUE3D)
    """
    global _EVENTS
    _EVENTS = events

    pool = mp.Pool(processes=n_jobs) if n_jobs > 1 else None

    # Pre-load baselines once — never recomputed during optimisation
    b_ee    = CLUE3D_BASELINES['ee']
    b_pi = CLUE3D_BASELINES['pi']

    def objective_fn(params):
        cee_params = list(params[:5])
        che_params = list(params[5:10])
        args = [(i, cee_params, che_params) for i in range(len(events))]

        if pool is not None:
            all_results = pool.map(_eval_event, args)
        else:
            all_results = [_eval_event(a) for a in args]

        # Split by particle type
        res_ee    = [r for r in all_results if r['particle_type'] == 'ee']
        res_pi = [r for r in all_results if r['particle_type'] == 'pi']

        # Any infeasible event in either subset → whole evaluation infeasible
        if (not res_ee or not res_pi
                or any(r['infeasible'] for r in all_results)):
            return [np.inf] * 6

        f1_ee    = _objective_purity(res_ee)
        f2_ee    = _objective_efficiency(res_ee)
        f3_ee    = _objective_fragmentation(res_ee)
        f1_pi = _objective_purity(res_pi)
        f2_pi = _objective_efficiency(res_pi)
        f3_pi = _objective_fragmentation(res_pi)

        return [
            f1_ee    / b_ee['f1'],
            f2_ee    / b_ee['f2'],
            f3_ee    / b_ee['f3'],
            f1_pi / b_pi['f1'],
            f2_pi / b_pi['f2'],
            f3_pi / b_pi['f3'],
        ]

    objective_fn._pool = pool
    return objective_fn


def make_staged_objective(events, n_jobs=1, subdet='CEE', fixed_other_params=None):
    """
    5-parameter, 3-objective callable for staged single-subdet optimisation.

    subdet='CEE': optimise CEE params on ee events only;
                  CHE fixed at fixed_other_params (defaults to DEFAULT_PARAMS[5:])
    subdet='CHE': optimise CHE params on pi events only;
                  CEE fixed at fixed_other_params (must be supplied — best from Phase 1)

    Returns callable(params[5]) -> [R1, R2, R3] for the relevant particle type,
    each divided by the corresponding CLUE3D baseline (< 1 = better than CLUE3D).

    The pool is forked *after* _EVENTS is set, so workers always see the correct
    filtered event list.  Close the returned obj._pool before calling this again.
    """
    global _EVENTS

    assert subdet in ('CEE', 'CHE'), f"subdet must be 'CEE' or 'CHE', got {subdet!r}"
    ptype = 'ee' if subdet == 'CEE' else 'pi'

    filtered = [e for e in events if e['particle_type'] == ptype]
    if not filtered:
        raise ValueError(f"No events with particle_type='{ptype}' found in event list")

    _EVENTS = filtered   # must be set before pool fork

    if fixed_other_params is None:
        from config import DEFAULT_PARAMS
        fixed_other_params = DEFAULT_PARAMS[5:] if subdet == 'CEE' else DEFAULT_PARAMS[:5]
    fixed_other_params = list(fixed_other_params)

    pool = mp.Pool(processes=n_jobs) if n_jobs > 1 else None
    b    = CLUE3D_BASELINES[ptype]

    def objective_fn(params):
        if subdet == 'CEE':
            cee_params = list(params[:5])
            che_params = fixed_other_params
        else:
            cee_params = fixed_other_params
            che_params = list(params[:5])

        args = [(i, cee_params, che_params) for i in range(len(filtered))]
        all_results = (pool.map(_eval_event, args) if pool is not None
                       else [_eval_event(a) for a in args])

        if not all_results or any(r['infeasible'] for r in all_results):
            return [np.inf] * 3

        f1 = _objective_purity(all_results)
        f2 = _objective_efficiency(all_results)
        f3 = _objective_fragmentation(all_results)
        return [f1 / b['f1'], f2 / b['f2'], f3 / b['f3']]

    objective_fn._pool = pool
    return objective_fn
