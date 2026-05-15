import uproot
import numpy as np
import awkward as ak
import pandas as pd

from config import BRANCHES, TRUTH_BRANCH, CLUE_BRANCH, PARTICLE_TYPES


def load_branch_with_highest_cycle(file, branch_name):
    all_keys      = file.keys()
    matching_keys = [k for k in all_keys if k.startswith(branch_name)]
    if not matching_keys:
        raise ValueError(
            f"No branch '{branch_name}' found. Available: {all_keys[:10]}"
        )
    return file[max(matching_keys, key=lambda k: int(k.split(';')[1]))]


def load_tree(path, filename, branch_name):
    f    = uproot.open(f"{path}/{filename}")
    tree = load_branch_with_highest_cycle(f, branch_name)
    return tree.arrays(BRANCHES)


def load_all(path, config_files):
    """
    Returns
    -------
    data : dict
        data['em']['truth'] / data['em']['clue']
        data['pion']['truth'] / data['pion']['clue']
    """
    data = {}
    for ptype, filename in config_files.items():
        print(f"Loading {ptype:>4s}  —  {filename}")
        data[ptype] = {
            'truth': load_tree(path, filename, TRUTH_BRANCH),
            'clue':  load_tree(path, filename, CLUE_BRANCH),
        }
        n_truth = len(data[ptype]['truth']['vertices_indexes'])
        n_clue  = len(data[ptype]['clue' ]['vertices_indexes'])
        print(f"  truth events: {n_truth}   clue events: {n_clue}")
    return data


def assign_lcs_to_particles(arrays, event_idx):
    """
    Assign layer clusters to particles via dominant multiplicity fraction.
    Used for truth (simtrackstersCP).

    Returns
    -------
    lc_coords : {lc_idx: (x, y, z, energy)}
    assigned  : {particle_idx: [lc_idx, ...]}
    """
    lc_fractions = {}
    lc_coords    = {}
    n_particles  = len(arrays['vertices_indexes'][event_idx])

    for particle_idx in range(n_particles):
        idxs  = ak.to_numpy(arrays['vertices_indexes'     ][event_idx][particle_idx])
        xs    = ak.to_numpy(arrays['vertices_x'           ][event_idx][particle_idx])
        ys    = ak.to_numpy(arrays['vertices_y'           ][event_idx][particle_idx])
        zs    = ak.to_numpy(arrays['vertices_z'           ][event_idx][particle_idx])
        ens   = ak.to_numpy(arrays['vertices_energy'      ][event_idx][particle_idx])
        mults = ak.to_numpy(arrays['vertices_multiplicity'][event_idx][particle_idx])

        for idx, x, y, z, e, m in zip(idxs, xs, ys, zs, ens, mults):
            if m <= 0:
                continue
            lc_fractions.setdefault(idx, {})[particle_idx] = 1.0 / m
            if idx not in lc_coords:
                lc_coords[idx] = (x, y, z, float(e))

    assigned = {p: [] for p in range(n_particles)}
    for lc_idx, fracs in lc_fractions.items():
        dominant = max(fracs, key=fracs.get)
        assigned[dominant].append(lc_idx)

    return lc_coords, assigned


def build_df_lc(raw, particle_types=None):
    """
    One row per LC per event for both truth and CLUE3D sources.
    Truth LCs are deduplicated via dominant-particle assignment.
    CLUE3D LCs are deduplicated across tracksters (first-trackster-wins).

    Columns: global_event_id, event_idx, particle_type, source,
             lc_idx, x, y, z, energy
    """
    if particle_types is None:
        particle_types = PARTICLE_TYPES

    rows       = []
    n_events   = min(len(raw[p]['truth']['vertices_indexes']) for p in particle_types)
    global_eid = 0

    for event_idx in range(n_events):
        for ptype in particle_types:
            base = dict(global_event_id=global_eid, event_idx=event_idx,
                        particle_type=ptype)

            # truth LCs — deduplicated via dominant-particle assignment
            lc_coords, assigned = assign_lcs_to_particles(raw[ptype]['truth'], event_idx)
            for lc_idx in {lc for lcs in assigned.values() for lc in lcs}:
                x, y, z, e = lc_coords[lc_idx]
                rows.append({**base, 'source': 'truth',
                             'lc_idx': lc_idx, 'x': x, 'y': y, 'z': z, 'energy': e})

            # CLUE3D LCs — deduplicated across tracksters (first-trackster-wins)
            clue = raw[ptype]['clue']
            n_ts = len(clue['vertices_indexes'][event_idx])
            seen = set()
            for ts_id in range(n_ts):
                idxs = ak.to_numpy(clue['vertices_indexes'][event_idx][ts_id])
                xs   = ak.to_numpy(clue['vertices_x'      ][event_idx][ts_id])
                ys   = ak.to_numpy(clue['vertices_y'      ][event_idx][ts_id])
                zs   = ak.to_numpy(clue['vertices_z'      ][event_idx][ts_id])
                ens  = ak.to_numpy(clue['vertices_energy' ][event_idx][ts_id])
                for lc_idx, x, y, z, e in zip(idxs, xs, ys, zs, ens):
                    if lc_idx not in seen:
                        seen.add(lc_idx)
                        rows.append({**base, 'source': 'clue',
                                     'lc_idx': lc_idx, 'x': x, 'y': y,
                                     'z': z, 'energy': e})

            global_eid += 1

    return pd.DataFrame(rows).astype({
        'global_event_id': 'int32', 'event_idx': 'int32',
        'lc_idx': 'int32',
        'x': 'float32', 'y': 'float32', 'z': 'float32', 'energy': 'float32',
    })


def build_lc_energy_lookup(df_lc):
    """Build {global_event_id: {lc_idx: energy}} from truth LCs."""
    return (
        df_lc[df_lc['source'] == 'truth']
        .groupby('global_event_id')
        .apply(lambda g: dict(zip(g['lc_idx'], g['energy'].astype(float))))
        .to_dict()
    )
