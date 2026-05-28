"""
Data loading and preprocessing for CLUEstering parameter tuning.

Loads truth information (simtrackstersCP) from two ROOT files — electron-electron
and electron-pion events — and interleaves them into a single event list.

Each event is represented as:
    {
        "particle_type": "ee" | "epion",
        "sim_showers": [
            {
                "shower_id":   int,
                "lc_indexes":  np.ndarray (int64),
                "lc_x":        np.ndarray (float32),
                "lc_y":        np.ndarray (float32),
                "lc_z":        np.ndarray (float32),
                "lc_energy":   np.ndarray (float32),
                "true_energy": float,
            },
            ...  # always 2 showers per event
        ],
        "all_lcs": {
            "indexes": np.ndarray (int64),
            "x":       np.ndarray (float32),
            "y":       np.ndarray (float32),
            "z":       np.ndarray (float32),
            "energy":  np.ndarray (float32),
            "subdet":  np.ndarray (str),   # "CEE" or "CHE" per LC
        }
    }
"""

import uproot
import numpy as np
import awkward as ak

from config import DATA_DIR, FILES, TRUTH_BRANCH, BRANCHES, CEE_Z_BOUNDARY


# ── ROOT loading ───────────────────────────────────────────────────────────────

def _highest_cycle(file, branch_name):
    """Return the tree key with the highest cycle number for branch_name."""
    matching = [k for k in file.keys() if k.startswith(branch_name)]
    if not matching:
        raise ValueError(
            f"Branch '{branch_name}' not found. "
            f"Available keys: {list(file.keys())[:10]}"
        )
    def _cycle(k):
        return int(k.split(';')[1]) if ';' in k else 0
    return file[max(matching, key=_cycle)]


def load_events(data_dir=None, files=None, truth_branch=None, branches=None):
    """
    Load, resolve and interleave events from the ee and epion ROOT files.

    Returns
    -------
    events : list of per-event dicts (alternating ee / epion)
        [ee_0, epion_0, ee_1, epion_1, ...]
    """
    if data_dir    is None: data_dir     = DATA_DIR
    if files       is None: files        = FILES
    if truth_branch is None: truth_branch = TRUTH_BRANCH
    if branches    is None: branches     = BRANCHES

    raw = {}
    for ptype, filename in files.items():
        path = f"{data_dir}/{filename}"
        f    = uproot.open(path)
        tree = _highest_cycle(f, truth_branch)
        raw[ptype] = tree.arrays(branches)
        n = len(raw[ptype]['vertices_indexes'])
        print(f"  {ptype:>5s}  {n} events  <-  {path}")

    n_events = min(len(raw[p]['vertices_indexes']) for p in files)
    print(f"Using {n_events} events per type → {2 * n_events} total (interleaved)\n")

    events = []
    for ev_idx in range(n_events):
        for ptype in ('ee', 'epion'):          # fixed order for reproducibility
            events.append(_build_event(raw[ptype], ev_idx, ptype))

    return events


# ── Per-event builder ──────────────────────────────────────────────────────────

def _build_event(arrays, event_idx, particle_type):
    """
    Build the per-event data structure for one event from one particle type.

    Steps:
      1. Extract raw shower data.
      2. Resolve ambiguous LC assignments via dominant 1/multiplicity.
      3. Build sim_showers list.
      4. Build all_lcs dict (union of all LCs, with subdet labels).
    """
    n_showers = len(arrays['vertices_indexes'][event_idx])

    # ── Step 1: extract raw per-shower arrays ──────────────────────────────────
    showers_raw = []
    for sh_id in range(n_showers):
        idxs  = ak.to_numpy(arrays['vertices_indexes'     ][event_idx][sh_id])
        xs    = ak.to_numpy(arrays['vertices_x'           ][event_idx][sh_id])
        ys    = ak.to_numpy(arrays['vertices_y'           ][event_idx][sh_id])
        zs    = ak.to_numpy(arrays['vertices_z'           ][event_idx][sh_id])
        ens   = ak.to_numpy(arrays['vertices_energy'      ][event_idx][sh_id])
        mults = ak.to_numpy(arrays['vertices_multiplicity'][event_idx][sh_id])
        showers_raw.append(
            dict(shower_id=sh_id, idxs=idxs, xs=xs, ys=ys,
                 zs=zs, ens=ens, mults=mults)
        )

    # ── Step 2: resolve ambiguous LC assignments ───────────────────────────────
    # For each LC index, keep it only in the shower where 1/multiplicity is largest.
    lc_best_frac  = {}   # lc_idx -> best 1/multiplicity seen so far
    lc_best_shower = {}  # lc_idx -> shower_id of the winner
    lc_coord       = {}  # lc_idx -> (x, y, z, energy)  [first occurrence wins]

    for sh in showers_raw:
        for i, lc_idx in enumerate(sh['idxs']):
            m = float(sh['mults'][i])
            if m <= 0:
                continue
            frac = 1.0 / m
            if lc_idx not in lc_best_frac or frac > lc_best_frac[lc_idx]:
                lc_best_frac[lc_idx]   = frac
                lc_best_shower[lc_idx] = sh['shower_id']
            if lc_idx not in lc_coord:
                lc_coord[lc_idx] = (float(sh['xs'][i]), float(sh['ys'][i]),
                                    float(sh['zs'][i]), float(sh['ens'][i]))

    # Group resolved LC indexes by shower
    shower_lcs = {sh['shower_id']: [] for sh in showers_raw}
    for lc_idx, winner in lc_best_shower.items():
        shower_lcs[winner].append(lc_idx)

    # ── Step 3: build sim_showers ──────────────────────────────────────────────
    sim_showers = []
    for sh in showers_raw:
        lc_list = sorted(shower_lcs[sh['shower_id']])   # deterministic order
        lc_arr  = np.array(lc_list, dtype=np.int64)

        if len(lc_arr) > 0:
            coords  = np.array([lc_coord[idx] for idx in lc_arr], dtype=np.float32)
            lc_x, lc_y, lc_z, lc_e = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
        else:
            lc_x = lc_y = lc_z = lc_e = np.empty(0, dtype=np.float32)

        sim_showers.append({
            'shower_id':   sh['shower_id'],
            'lc_indexes':  lc_arr,
            'lc_x':        lc_x,
            'lc_y':        lc_y,
            'lc_z':        lc_z,
            'lc_energy':   lc_e,
            'true_energy': float(lc_e.sum()),
        })

    # ── Step 4: build all_lcs ─────────────────────────────────────────────────
    # Union of all LC indexes that survived the multiplicity filter.
    all_lc_sorted = np.array(sorted(lc_coord.keys()), dtype=np.int64)
    coords_all    = np.array([lc_coord[idx] for idx in all_lc_sorted], dtype=np.float32)

    all_x = coords_all[:, 0]
    all_y = coords_all[:, 1]
    all_z = coords_all[:, 2]
    all_e = coords_all[:, 3]

    # ── 2-way subdetector assignment ──────────────────────────────────────────
    abs_z  = np.abs(all_z)
    subdet = np.where(abs_z < CEE_Z_BOUNDARY, 'CEE', 'CHE')

    return {
        'particle_type': particle_type,
        'sim_showers':   sim_showers,
        'all_lcs': {
            'indexes': all_lc_sorted,
            'x':       all_x,
            'y':       all_y,
            'z':       all_z,
            'energy':  all_e,
            'subdet':  subdet,
        },
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_events(events):
    """
    Sanity-check the preprocessed event list.  Raises AssertionError on failure.
    """
    for i, ev in enumerate(events):
        assert len(ev['sim_showers']) == 2, (
            f"Event {i}: expected 2 sim_showers, got {len(ev['sim_showers'])}"
        )

        # No LC appears in more than one shower
        seen = {}
        for sh in ev['sim_showers']:
            for lc_idx in sh['lc_indexes']:
                assert lc_idx not in seen, (
                    f"Event {i}: LC {lc_idx} appears in showers "
                    f"{seen[lc_idx]} and {sh['shower_id']}"
                )
                seen[lc_idx] = sh['shower_id']

        # Every shower has positive true energy
        for sh in ev['sim_showers']:
            assert sh['true_energy'] > 0, (
                f"Event {i} shower {sh['shower_id']}: true_energy == 0"
            )

        # Only valid subdet labels
        unique_subdets = set(ev['all_lcs']['subdet'])
        assert unique_subdets <= {'CEE', 'CHE'}, (
            f"Event {i}: unexpected subdet values: {unique_subdets}"
        )

    print(f"Validation passed: {len(events)} events OK")
