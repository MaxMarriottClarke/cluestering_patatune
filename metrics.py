import numpy as np
import awkward as ak

from config import PARTICLE_TYPES
from data import assign_lcs_to_particles


def compute_r2s(reco_lc_idxs, sim_lc_idxs, lc_energies):
    """
    Fraction of reco energy² not from the matched sim particle.
    0 = perfect purity, 1 = completely fake.
    """
    reco_set = set(reco_lc_idxs)
    sim_set  = set(sim_lc_idxs)
    contam   = sum(lc_energies.get(k, 0.0) ** 2 for k in reco_set if k not in sim_set)
    total    = sum(lc_energies.get(k, 0.0) ** 2 for k in reco_set)
    return 1.0 if total == 0 else contam / total


def compute_baseline(raw, lc_energy_lookup, particle_types=None):
    """
    Compute mean L1, r, N_T for CLUE3D per particle type.

    Returns
    -------
    baseline : {'em': {'L1': float, 'r': float, 'NT': float}, 'pion': {...}}
    """
    if particle_types is None:
        particle_types = PARTICLE_TYPES

    n_events   = min(len(raw[p]['truth']['vertices_indexes']) for p in particle_types)
    result     = {p: {'L1': [], 'r': [], 'NT': []} for p in particle_types}
    global_eid = 0

    for event_idx in range(n_events):
        for ptype in particle_types:
            lc_energies = lc_energy_lookup.get(global_eid, {})

            _, assigned   = assign_lcs_to_particles(raw[ptype]['truth'], event_idx)
            truth_lc_sets = {cp: set(lcs) for cp, lcs in assigned.items() if lcs}

            clue = raw[ptype]['clue']
            n_ts = len(clue['vertices_indexes'][event_idx])

            # collect (lc_idxs, total_energy) per CLUE3D trackster
            reco_rows = []
            for ts_id in range(n_ts):
                idxs = ak.to_numpy(clue['vertices_indexes'][event_idx][ts_id])
                ens  = ak.to_numpy(clue['vertices_energy' ][event_idx][ts_id])
                seen, mask = set(), []
                for lc_idx in idxs:
                    mask.append(lc_idx not in seen)
                    seen.add(lc_idx)
                mask = np.array(mask)
                idxs, ens = idxs[mask], ens[mask]
                reco_rows.append((idxs, float(ens.sum())))

            total_truth = sum(lc_energies.values())

            if not reco_rows or not truth_lc_sets:
                result[ptype]['L1'].append(1.0)
                result[ptype]['r'].append(0.0)
                result[ptype]['NT'].append(0)
                global_eid += 1
                continue

            total_reco = sum(e for _, e in reco_rows)
            r = min(total_reco / total_truth, 1.0) if total_truth > 0 else 0.0

            # L1: energy-weighted min r2s per reco trackster
            weighted_sum, total_reco_e = 0.0, 0.0
            for idxs, e_reco in reco_rows:
                best = min(
                    compute_r2s(idxs, sim_set, lc_energies)
                    for sim_set in truth_lc_sets.values()
                )
                weighted_sum  += e_reco * best
                total_reco_e  += e_reco
            L1 = weighted_sum / total_reco_e if total_reco_e > 0 else 1.0

            result[ptype]['L1'].append(L1)
            result[ptype]['r'].append(r)
            result[ptype]['NT'].append(len(reco_rows))
            global_eid += 1

    return {
        p: {k: float(np.mean(v)) for k, v in s.items()}
        for p, s in result.items()
    }
