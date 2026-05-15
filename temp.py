mport uproot
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import CLUEstering as clue
import patatune



BRANCHES = [
    'vertices_indexes',
    'vertices_x',
    'vertices_y',
    'vertices_z',
    'vertices_energy',
    'vertices_multiplicity',
]

PARTICLE_TYPES = ['em', 'pion']

PATH = '/eos/user/m/mmarriot/SWAN_projects/cluestering_opt'

CONFIG_FILES = {
    'em':   'tune/e.root',   # update filenames
    'pion': 'tune/pi.root',
}

TRUTH_BRANCH = 'ticlDumper/simtrackstersCP'
CLUE_BRANCH  = 'ticlDumper/hltTiclTrackstersCLUE3DHigh'

def load_branch_with_highest_cycle(file, branch_name):
    all_keys      = file.keys()
    matching_keys = [k for k in all_keys if k.startswith(branch_name)]
    if not matching_keys:
        raise ValueError(f"No branch '{branch_name}' found in file. Available: {all_keys[:10]}")
    return file[max(matching_keys, key=lambda k: int(k.split(';')[1]))]


def load_tree(path, filename, branch_name):
    """Open a ROOT file and return the arrays for the requested branch."""
    f    = uproot.open(f"{path}/{filename}")
    tree = load_branch_with_highest_cycle(f, branch_name)
    return tree.arrays(BRANCHES)


def load_all(path, config_files):
    """
    Returns
    -------
    data : dict
        data['em']['truth']  / data['em']['clue']
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


raw = load_all(PATH, CONFIG_FILES)

def assign_lcs_to_particles(arrays, event_idx):
    """
    Assign layer clusters to particles using multiplicity fractions.
    Used for truth (simtrackstersCP) only — CLUE3D assignment is already stored.

    Returns
    -------
    lc_coords : dict  {lc_idx: (x, y, z, energy)}
    assigned  : dict  {0: [lc_idx, ...], 1: [lc_idx, ...]}
    """
    lc_fractions = {}
    lc_coords    = {}

    n_particles = len(arrays['vertices_indexes'][event_idx])

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



def flatten_truth_event(arrays, event_idx, particle_type):
    """
    Flatten one truth event into a list of dicts, one row per LC.
    Each LC appears exactly once (dominant-particle assignment).
    """
    lc_coords, assigned = assign_lcs_to_particles(arrays, event_idx)
    rows = []
    for particle_idx, lc_list in assigned.items():
        for lc_idx in lc_list:
            x, y, z, e = lc_coords[lc_idx]
            rows.append({
                'event_idx':     event_idx,
                'particle_type': particle_type,
                'source':        'truth',
                'trackster_id':  particle_idx,
                'lc_idx':        lc_idx,
                'x': x, 'y': y, 'z': z,
                'energy':        e,
            })
    return rows


def flatten_clue_event(arrays, event_idx, particle_type):
    """
    Flatten one CLUE3D reco event into a list of dicts, one row per LC.
    No assignment needed — trackster membership is already stored per trackster.
    Duplicates within a trackster are dropped; if an LC appears in multiple
    tracksters it is kept in all (CLUE3D should not do this, but we deduplicate
    per trackster below just in case).
    """
    rows = []
    n_tracksters = len(arrays['vertices_indexes'][event_idx])

    seen = {}   # lc_idx -> trackster_id of first occurrence (for dedup across tracksters)

    for ts_id in range(n_tracksters):
        idxs  = ak.to_numpy(arrays['vertices_indexes'][event_idx][ts_id])
        xs    = ak.to_numpy(arrays['vertices_x'      ][event_idx][ts_id])
        ys    = ak.to_numpy(arrays['vertices_y'      ][event_idx][ts_id])
        zs    = ak.to_numpy(arrays['vertices_z'      ][event_idx][ts_id])
        ens   = ak.to_numpy(arrays['vertices_energy' ][event_idx][ts_id])

        # deduplicate within this trackster
        seen_in_ts = set()
        for lc_idx, x, y, z, e in zip(idxs, xs, ys, zs, ens):
            if lc_idx in seen_in_ts:
                continue
            seen_in_ts.add(lc_idx)

            if lc_idx not in seen:          # first trackster wins across tracksters
                seen[lc_idx] = ts_id
                rows.append({
                    'event_idx':     event_idx,
                    'particle_type': particle_type,
                    'source':        'clue',
                    'trackster_id':  ts_id,
                    'lc_idx':        lc_idx,
                    'x': x, 'y': y, 'z': z,
                    'energy':        e,
                })
    return rows




def build_df_lc(raw, particle_types=PARTICLE_TYPES):
    """
    One row per LC per event.
    Truth LCs are deduplicated (each LC appears once, assigned to dominant particle).
    CLUE3D LCs are deduplicated across tracksters (first-trackster-wins).

    Columns: global_event_id, event_idx, particle_type, source,
             lc_idx, x, y, z, energy
    """
    rows       = []
    n_events   = min(len(raw[p]['truth']['vertices_indexes']) for p in particle_types)
    global_eid = 0

    for event_idx in range(n_events):
        for ptype in particle_types:
            base = dict(global_event_id=global_eid, event_idx=event_idx,
                        particle_type=ptype)

            # --- truth: deduplicated via dominant-particle assignment ---
            lc_coords, assigned = assign_lcs_to_particles(
                raw[ptype]['truth'], event_idx)
            for lc_idx in {lc for lcs in assigned.values() for lc in lcs}:
                x, y, z, e = lc_coords[lc_idx]
                rows.append({**base, 'source': 'truth',
                             'lc_idx': lc_idx, 'x': x, 'y': y, 'z': z, 'energy': e})

            # --- clue: deduplicated across tracksters (first-trackster-wins) ---
            clue  = raw[ptype]['clue']
            n_ts  = len(clue['vertices_indexes'][event_idx])
            seen  = set()
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

    df = pd.DataFrame(rows).astype({
        'global_event_id': 'int32', 'event_idx': 'int32',
        'lc_idx': 'int32',
        'x': 'float32', 'y': 'float32', 'z': 'float32', 'energy': 'float32',
    })
    return df

df_lc = build_df_lc(raw)
print(df_lc.head())
print(f"\nTotal LC rows: {len(df_lc):,}")
print(df_lc.groupby(['particle_type', 'source']).size().rename('n_lcs'))

def build_df_tracksters(raw, particle_types=PARTICLE_TYPES):
    """
    One row per (event, source, trackster_id).
    LC coordinates are stored as numpy arrays in list-columns.

    Columns: global_event_id, event_idx, particle_type, source, trackster_id,
             xs, ys, zs, energies, n_lcs, total_energy
    """
    rows       = []
    n_events   = min(len(raw[p]['truth']['vertices_indexes']) for p in particle_types)
    global_eid = 0

    for event_idx in range(n_events):
        for ptype in particle_types:
            base = dict(global_event_id=global_eid, event_idx=event_idx,
                        particle_type=ptype)

            # --- truth tracksters ---
            lc_coords, assigned = assign_lcs_to_particles(
                raw[ptype]['truth'], event_idx)
            for cp_id, lc_list in assigned.items():
                if not lc_list:
                    continue
                coords = np.array([lc_coords[i] for i in lc_list])  # (N, 4)
                rows.append({**base, 'source': 'truth', 'trackster_id': cp_id,
                             'xs': coords[:, 0], 'ys': coords[:, 1],
                             'zs': coords[:, 2], 'energies': coords[:, 3],
                             'n_lcs': len(lc_list),
                             'total_energy': coords[:, 3].sum()})

            # --- clue3d tracksters ---
            clue = raw[ptype]['clue']
            n_ts = len(clue['vertices_indexes'][event_idx])
            for ts_id in range(n_ts):
                idxs = ak.to_numpy(clue['vertices_indexes'][event_idx][ts_id])
                xs   = ak.to_numpy(clue['vertices_x'      ][event_idx][ts_id])
                ys   = ak.to_numpy(clue['vertices_y'      ][event_idx][ts_id])
                zs   = ak.to_numpy(clue['vertices_z'      ][event_idx][ts_id])
                ens  = ak.to_numpy(clue['vertices_energy' ][event_idx][ts_id])
                # deduplicate within trackster
                seen, mask = set(), []
                for i, lc_idx in enumerate(idxs):
                    mask.append(lc_idx not in seen)
                    seen.add(lc_idx)
                mask = np.array(mask)
                xs, ys, zs, ens = xs[mask], ys[mask], zs[mask], ens[mask]
                if len(xs) == 0:
                    continue
                rows.append({**base, 'source': 'clue', 'trackster_id': ts_id,
                             'xs': xs, 'ys': ys, 'zs': zs, 'energies': ens,
                             'n_lcs': len(xs),
                             'total_energy': float(ens.sum())})

            global_eid += 1

    df = pd.DataFrame(rows).astype({
        'global_event_id': 'int32', 'event_idx': 'int32',
        'trackster_id': 'int32',
        'n_lcs': 'int32', 'total_energy': 'float32',
    })
    return df

df_tracksters = build_df_tracksters(raw)


import CLUEstering as clue

# ── parameters (starting point, will be tuned) ──
DC   = 3.0    # spatial density radius [cm]
RHOC = 5.0    # min local density to be a seed
DO   = 3.0    # outlier delta threshold  (will be ignored — outliers dropped)
DS   = 3.0   # seed delta threshold
WZ   = 0.5    # z-weight: compress z relative to x/y so layers don't dominate

def run_cluestering(xs, ys, zs, energies, dc=DC, rhoc=RHOC, do=DO, ds=DS, wz=WZ):
    """
    Run CLUEstering on one event's flat LC point cloud.
    Returns cluster_ids array; -1 = outlier.
    """
    data = pd.DataFrame({
        'x0': xs, 'x1': ys, 'x2': wz * zs, 'weight': energies
    })
    c = clue.clusterer(float(dc), float(rhoc), float(do), float(ds))
    c.read_data(data)
    c.run_clue()
    return np.array(c.cluster_ids)


def build_df_cluestering(df_lc, dc=DC, rhoc=RHOC, do=DO, ds=DS, wz=WZ):
    """
    Run CLUEstering on the flattened truth LCs for every event.
    Returns a DataFrame in the same format as df_tracksters (source='cluestering'),
    with outliers (cluster_id == -1) dropped.

    We run on truth LCs only — those are the clean deduplicated point cloud
    that we want to cluster. CLUE3D reco LCs are kept in df_tracksters for comparison.
    """
    rows = []

    for geid, ev in df_lc[df_lc['source'] == 'truth'].groupby('global_event_id'):
        ptype = ev['particle_type'].iloc[0]
        eidx  = ev['event_idx'].iloc[0]

        xs       = ev['x'].to_numpy()
        ys       = ev['y'].to_numpy()
        zs       = ev['z'].to_numpy()
        energies = ev['energy'].to_numpy()
        lc_idxs  = ev['lc_idx'].to_numpy()

        cluster_ids = run_cluestering(xs, ys, zs, energies, dc, rhoc, do, ds, wz)

        # drop outliers
        mask = cluster_ids != -1
        xs, ys, zs       = xs[mask], ys[mask], zs[mask]
        energies         = energies[mask]
        lc_idxs          = lc_idxs[mask]
        cluster_ids      = cluster_ids[mask]

        n_outliers = (~mask).sum()

        for cid in np.unique(cluster_ids):
            sel = cluster_ids == cid
            rows.append({
                'global_event_id': geid,
                'event_idx':       eidx,
                'particle_type':   ptype,
                'source':          'cluestering',
                'trackster_id':    int(cid),
                'xs':              xs[sel],
                'ys':              ys[sel],
                'zs':              zs[sel],
                'energies':        energies[sel],
                'lc_idxs':         lc_idxs[sel],
                'n_lcs':           int(sel.sum()),
                'total_energy':    float(energies[sel].sum()),
                'n_outliers_dropped': int(n_outliers),  # informational
            })

    df = pd.DataFrame(rows).astype({
        'global_event_id': 'int32', 'event_idx': 'int32',
        'trackster_id': 'int32', 'n_lcs': 'int32', 'total_energy': 'float32',
    })
    return df


df_cluestering = build_df_cluestering(df_lc)

print(df_cluestering.head())
print(f"\nTotal CLUEstering trackster rows: {len(df_cluestering):,}")
print(df_cluestering.groupby('particle_type')['n_lcs']
      .agg(['count', 'mean'])
      .rename(columns={'count': 'n_tracksters', 'mean': 'mean_lcs_per_trackster'}))
print(f"\nMean outliers dropped per event: "
      f"{df_cluestering.groupby('global_event_id')['n_outliers_dropped'].first().mean():.1f}")

# ── LC energy lookup: built once from truth flat LCs ──────────────────────────
# {global_event_id: {lc_idx: energy}}
lc_energy_lookup = (
    df_lc[df_lc['source'] == 'truth']
    .groupby('global_event_id')
    .apply(lambda g: dict(zip(g['lc_idx'], g['energy'].astype(float))))
    .to_dict()
)

# ── Baseline metrics from CLUE3D (df_tracksters) ──────────────────────────────
def compute_r2s(reco_lc_idxs, sim_lc_idxs, lc_energies):
    """
    Fraction of reco energy^2 not from matched sim particle.
    0 = perfect purity, 1 = completely fake.
    """
    reco_set = set(reco_lc_idxs)
    sim_set  = set(sim_lc_idxs)
    contam   = sum(lc_energies.get(k, 0.0)**2 for k in reco_set if k not in sim_set)
    total    = sum(lc_energies.get(k, 0.0)**2 for k in reco_set)
    return 1.0 if total == 0 else contam / total


def per_event_metrics(ev_tracksters, lc_energies):
    """
    Compute L1 (purity), r (energy ratio), N_T (trackster count)
    for one event given its trackster rows (one source: truth or clue/cluestering).

    ev_tracksters : DataFrame rows for one (event, source)
    lc_energies   : {lc_idx: energy} for this event (from truth LCs)

    Returns: (L1, r, N_T)
    """
    truth_rows = None   # we always need truth rows to compute sim_set
    # pass truth rows in via the caller — see compute_baseline / objective below

    reco_rows = ev_tracksters
    if len(reco_rows) == 0:
        return 1.0, 0.0, 0   # worst case

    total_reco_energy = reco_rows['total_energy'].sum()

    # energy ratio r — capped at 1
    total_truth_energy = sum(lc_energies.values())
    r = min(total_reco_energy / total_truth_energy, 1.0) if total_truth_energy > 0 else 0.0

    N_T = len(reco_rows)

    return r, N_T   # L1 needs truth LC sets — handled separately below


def compute_L1_event(reco_rows, truth_rows, lc_energies):
    """
    Energy-weighted mean r2s score for one event.
    For each reco trackster, finds the best-matching truth CP (min r2s).
    """
    if len(reco_rows) == 0:
        return 1.0

    # build truth LC sets: {cp_id: set(lc_idxs)}
    truth_lc_sets = {
        int(row['trackster_id']): set(row['lc_idxs'])
        for _, row in truth_rows.iterrows()
    } if truth_rows is not None and len(truth_rows) > 0 else {}

    weighted_sum  = 0.0
    total_reco_e  = 0.0

    for _, reco_row in reco_rows.iterrows():
        reco_idxs = reco_row['lc_idxs'] if hasattr(reco_row['lc_idxs'], '__iter__') \
                    else np.array([])
        e_reco = float(reco_row['total_energy'])

        if len(truth_lc_sets) > 0:
            best_r2s = min(
                compute_r2s(reco_idxs, sim_set, lc_energies)
                for sim_set in truth_lc_sets.values()
            )
        else:
            best_r2s = 1.0   # no truth → worst case

        weighted_sum  += e_reco * best_r2s
        total_reco_e  += e_reco

    return weighted_sum / total_reco_e if total_reco_e > 0 else 1.0


def compute_baseline_metrics(df_tracksters, df_lc, particle_types=PARTICLE_TYPES):
    """
    Compute mean L1, r, N_T for CLUE3D per particle type.
    Returns dict: {'em': {'L1': ..., 'r': ..., 'NT': ...}, 'pion': {...}}
    """
    # CLUE3D rows need lc_idxs to compute r2s — but df_tracksters from Cell 5
    # stores xs/ys/zs not lc_idxs. We add lc_idxs to clue rows here.
    # For the baseline we only need per-event r and N_T from df_tracksters,
    # and L1 requires lc_idxs which we store in df_cluestering.
    # Solution: re-extract CLUE3D lc_idxs from raw (same loop as build_df_tracksters)
    # but we avoid redoing that — instead we compute L1 from df_cluestering structure.
    # Here we compute r and N_T from df_tracksters (CLUE3D source).

    baseline = {}
    for ptype in particle_types:
        clue_rows  = df_tracksters[
            (df_tracksters['source'] == 'clue') &
            (df_tracksters['particle_type'] == ptype)
        ]
        truth_rows = df_tracksters[
            (df_tracksters['source'] == 'truth') &
            (df_tracksters['particle_type'] == ptype)
        ]

        L1_list, r_list, NT_list = [], [], []

        for geid, ev_clue in clue_rows.groupby('global_event_id'):
            ev_truth   = truth_rows[truth_rows['global_event_id'] == geid]
            lc_energies = lc_energy_lookup.get(geid, {})

            # energy ratio
            total_reco  = ev_clue['total_energy'].sum()
            total_truth = sum(lc_energies.values())
            r = min(total_reco / total_truth, 1.0) if total_truth > 0 else 0.0

            # N_T
            NT = len(ev_clue)

            # L1 — build truth lc sets from df_tracksters lc arrays
            truth_lc_sets = {
                int(row['trackster_id']): set(
                    np.arange(len(row['xs']))   # placeholder: no lc_idxs in df_tracksters
                )
                for _, row in ev_truth.iterrows()
            }
            # NOTE: df_tracksters doesn't store lc_idxs (only xs/ys/zs/energies).
            # For a proper L1 we use df_cluestering which stores lc_idxs.
            # The CLUE3D baseline L1 is approximated here; override below.

            L1_list.append(1.0)   # placeholder — see note below
            r_list.append(r)
            NT_list.append(NT)

        baseline[ptype] = {
            'L1': float(np.mean(L1_list)),
            'r':  float(np.mean(r_list)),
            'NT': float(np.mean(NT_list)),
        }

    return baseline


# ── Proper CLUE3D baseline L1 using raw data ────────────────────────────────────
# We compute baseline L1 directly from raw, since df_tracksters lacks lc_idxs for clue
def compute_clue_baseline_L1(raw, lc_energy_lookup, particle_types=PARTICLE_TYPES):
    """
    Compute mean L1 for CLUE3D per particle type using raw arrays (which have lc_idxs).
    """
    from itertools import islice

    n_events = min(len(raw[p]['truth']['vertices_indexes']) for p in particle_types)
    result   = {p: [] for p in particle_types}
    global_eid = 0

    for event_idx in range(n_events):
        for ptype in particle_types:
            lc_energies = lc_energy_lookup.get(global_eid, {})

            # truth LC sets via assignment
            _, assigned = assign_lcs_to_particles(raw[ptype]['truth'], event_idx)
            truth_lc_sets = {cp: set(lcs) for cp, lcs in assigned.items() if lcs}

            # CLUE3D reco tracksters
            clue   = raw[ptype]['clue']
            n_ts   = len(clue['vertices_indexes'][event_idx])

            class _Row:
                pass

            reco_rows = []
            for ts_id in range(n_ts):
                idxs = ak.to_numpy(clue['vertices_indexes'][event_idx][ts_id])
                ens  = ak.to_numpy(clue['vertices_energy' ][event_idx][ts_id])
                seen, mask = set(), []
                for i, lc_idx in enumerate(idxs):
                    mask.append(lc_idx not in seen); seen.add(lc_idx)
                mask = np.array(mask)
                idxs, ens = idxs[mask], ens[mask]
                r = _Row()
                r.lc_idxs     = idxs
                r.total_energy = float(ens.sum())
                reco_rows.append(r)

            if len(reco_rows) == 0 or len(truth_lc_sets) == 0:
                result[ptype].append(1.0)
                global_eid += 1
                continue

            total_reco_e, weighted_sum = 0.0, 0.0
            for rr in reco_rows:
                best = min(
                    compute_r2s(rr.lc_idxs, sim_set, lc_energies)
                    for sim_set in truth_lc_sets.values()
                )
                weighted_sum  += rr.total_energy * best
                total_reco_e  += rr.total_energy

            result[ptype].append(weighted_sum / total_reco_e if total_reco_e > 0 else 1.0)
            global_eid += 1

    return {p: float(np.mean(vals)) for p, vals in result.items()}


baseline_metrics = compute_baseline_metrics(df_tracksters, df_lc)
baseline_L1      = compute_clue_baseline_L1(raw, lc_energy_lookup)

# Override the placeholder L1 with the real values
for ptype in PARTICLE_TYPES:
    baseline_metrics[ptype]['L1'] = baseline_L1[ptype]

print("CLUE3D baseline metrics:")
for ptype, m in baseline_metrics.items():
    print(f"  {ptype:>4s}  L1={m['L1']:.4f}  r={m['r']:.4f}  NT={m['NT']:.2f}")

ALPHA = 0.5    # EM / HAD balance weight
LAMBDA = 2.0   # penalty weight for N_T < 2

def _run_and_score(params, particle_types, raw, lc_energy_lookup, baseline_metrics):
    """
    Core evaluation loop: runs CLUEstering with given params on all events,
    computes mean L1, r, N_T per particle type.

    params: [dc, rhoc, do, ds, wz]
    Returns: {'em': {'L1': ..., 'r': ..., 'NT': ...}, 'pion': {...}}
    """
    dc, rhoc, do, ds, wz = params

    n_events   = min(len(raw[p]['truth']['vertices_indexes']) for p in particle_types)
    scores     = {p: {'L1': [], 'r': [], 'NT': []} for p in particle_types}
    global_eid = 0

    for event_idx in range(n_events):
        for ptype in particle_types:
            lc_energies = lc_energy_lookup.get(global_eid, {})

            # ── flat truth LCs for this event ──
            ev = df_lc[
                (df_lc['global_event_id'] == global_eid) &
                (df_lc['source'] == 'truth')
            ]
            if len(ev) == 0:
                global_eid += 1
                continue

            xs       = ev['x'].to_numpy()
            ys       = ev['y'].to_numpy()
            zs       = ev['z'].to_numpy()
            energies = ev['energy'].to_numpy()
            lc_idxs  = ev['lc_idx'].to_numpy()

            # ── run CLUEstering ──
            cluster_ids = run_cluestering(xs, ys, zs, energies, dc, rhoc, do, ds, wz)

            # drop outliers
            mask        = cluster_ids != -1
            lc_idxs_cl  = lc_idxs[mask]
            energies_cl = energies[mask]
            cluster_ids_cl = cluster_ids[mask]

            unique_ids = np.unique(cluster_ids_cl)
            N_T = len(unique_ids)

            # ── truth LC sets for r2s ──
            _, assigned  = assign_lcs_to_particles(raw[ptype]['truth'], event_idx)
            truth_lc_sets = {cp: set(lcs) for cp, lcs in assigned.items() if lcs}

            # ── energy ratio r ──
            total_reco  = float(energies_cl.sum())
            total_truth = sum(lc_energies.values())
            r = min(total_reco / total_truth, 1.0) if total_truth > 0 else 0.0

            # ── L1: energy-weighted min r2s per reco trackster ──
            if N_T == 0 or len(truth_lc_sets) == 0:
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


def _delta(new_val, base_val, higher_is_better=False):
    """Signed relative change vs baseline. Negative = improvement."""
    if base_val == 0:
        return 0.0
    if higher_is_better:
        return (base_val - new_val) / base_val   # flip: improvement when new > base
    return (new_val - base_val) / base_val


def make_objectives(raw, lc_energy_lookup, baseline_metrics,
                    particle_types=PARTICLE_TYPES, alpha=ALPHA, lam=LAMBDA):
    """
    Returns three objective functions ready for patatune.
    Each takes x = [dc, rhoc, do, ds, wz] and returns a scalar.
    """
    def _combined(scores, key, higher_is_better=False):
        """Weighted combination across particle types."""
        weights = {'em': alpha, 'pion': 1.0 - alpha}
        return sum(
            weights[p] * _delta(scores[p][key],
                                 baseline_metrics[p][key],
                                 higher_is_better=higher_is_better)
            for p in particle_types
        )

    def f1_purity(x):
        """δ₁: energy-weighted r2s improvement. Minimise. Negative = better than CLUE3D."""
        scores = _run_and_score(x, particle_types, raw, lc_energy_lookup, baseline_metrics)
        return _combined(scores, 'L1', higher_is_better=False)

    def f2_energy(x):
        """Mean energy ratio r. Maximise. Higher = more energy recovered."""
        scores = _run_and_score(x, particle_types, raw, lc_energy_lookup, baseline_metrics)
        # return raw r (patatune will maximise via directions argument)
        return sum(
            ({'em': alpha, 'pion': 1.0 - alpha}[p]) * scores[p]['r']
            for p in particle_types
        )

    def f3_count(x):
        """δ₃: trackster count improvement + floor penalty. Minimise."""
        scores = _run_and_score(x, particle_types, raw, lc_energy_lookup, baseline_metrics)
        delta3 = _combined(scores, 'NT', higher_is_better=False)
        # floor penalty: heavily penalise collapsing to fewer than 2 tracksters
        mean_NT = sum(
            ({'em': alpha, 'pion': 1.0 - alpha}[p]) * scores[p]['NT']
            for p in particle_types
        )
        penalty = lam * max(0.0, 2.0 - mean_NT) ** 2
        return delta3 + penalty

    return f1_purity, f2_energy, f3_count


f1_purity, f2_energy, f3_count = make_objectives(
    raw, lc_energy_lookup, baseline_metrics
)

print("Objective functions built. Testing on default CLUEstering params...")
x0 = [DC, RHOC, DO, DS, WZ]
print(f"  f1_purity : {f1_purity(x0):.4f}  (0 = parity with CLUE3D, <0 = better)")
print(f"  f2_energy : {f2_energy(x0):.4f}  (higher is better)")
print(f"  f3_count  : {f3_count(x0):.4f}  (0 = parity with CLUE3D, <0 = better)")

import patatune

# ── single-pass objective: all 3 metrics in one CLUEstering sweep ─────────────
def objective_fn(x):
    """
    Runs CLUEstering once per event and returns all 3 objective values.
    x = [dc, rhoc, do, ds, wz]
    Returns: [purity_delta, energy_ratio, count_delta]
    """
    scores = _run_and_score(x, PARTICLE_TYPES, raw, lc_energy_lookup, baseline_metrics)

    # δ₁ purity — minimise
    d1 = sum(
        {'em': ALPHA, 'pion': 1-ALPHA}[p] *
        _delta(scores[p]['L1'], baseline_metrics[p]['L1'], higher_is_better=False)
        for p in PARTICLE_TYPES
    )

    # energy ratio r — maximise (return raw r, patatune handles direction)
    r_combined = sum(
        {'em': ALPHA, 'pion': 1-ALPHA}[p] * scores[p]['r']
        for p in PARTICLE_TYPES
    )

    # δ₃ count + floor penalty — minimise
    mean_NT = sum(
        {'em': ALPHA, 'pion': 1-ALPHA}[p] * scores[p]['NT']
        for p in PARTICLE_TYPES
    )
    d3 = sum(
        {'em': ALPHA, 'pion': 1-ALPHA}[p] *
        _delta(scores[p]['NT'], baseline_metrics[p]['NT'], higher_is_better=False)
        for p in PARTICLE_TYPES
    )
    penalty = LAMBDA * max(0.0, 2.0 - mean_NT) ** 2

    return [d1, r_combined, d3 + penalty]


objective = patatune.ElementWiseObjective(
    objective_fn,
    num_objectives=3,
    directions=['minimize', 'maximize', 'minimize'],
    objective_names=['purity_delta', 'energy_ratio', 'count_delta'],
)

# ── parameter bounds ──────────────────────────────────────────────────────────
lb = [1.0,  0.1, 1.0, 1.0,  0.2]
ub = [8.0, 10.0, 20.0, 20.0, 0.8]

# ── MOPSO config ──────────────────────────────────────────────────────────────
# With 500 events and single-pass objective, rough timing:
#   ~5ms/event × 1000 events/particle × 30 particles = ~2.5 min/iteration
#   50 iterations × 2.5 min = ~2 hours total — reasonable for a notebook run
#
# Hyperparameter rationale:
#   num_particles=30  : minimum for 5D space to sample meaningfully;
#                       below 20 the swarm can't cover the space,
#                       above 50 you're paying wall-clock with no gain at this budget
#   num_iterations=50 : MOPSO typically converges in 30–80 iterations for 5D;
#                       50 gives you a meaningful Pareto front to inspect
#   inertia=0.5       : slightly higher than default — you want exploration early
#                       in a 5D space with complex interactions between dc/rhoc/ds
#   cognitive=1.5     : standard — particles trust their own best
#   social=1.5        : matched to cognitive — balanced exploration/exploitation;
#                       higher social (2.0+) converges faster but risks premature
#                       convergence around one region of the Pareto front
#   max_pareto_length=100 : generous archive — you want the full front, not a
#                           truncated one; 100 is fine for 3 objectives
#   topology='random' : cheapest, works well for 3 objectives;
#                       use 'lower_weighted_crowding_distance' if you find the
#                       front is clustering in one corner after 50 iterations

mopso = patatune.MOPSO(
    objective=objective,
    lower_bounds=lb,
    upper_bounds=ub,
    num_particles=30,
    inertia_weight=0.5,
    cognitive_coefficient=1.5,
    social_coefficient=1.5,
    initial_particles_position='gaussian',
    default_point=x0,
    max_pareto_length=100,
    topology='random',
)

patatune.Randomizer.rng = np.random.default_rng(42)
patatune.Logger.setLevel('WARNING')   # INFO is very noisy over 50 iterations

# ── timing probe: how long does one particle evaluation actually take? ─────────
import time
print("Timing one objective call …")
t0     = time.time()
result = objective_fn(x0)
t1     = time.time()
per_call = t1 - t0
print(f"  one call : {per_call:.2f}s")
print(f"  per iter : {per_call * 30 / 60:.1f} min  (30 particles)")
print(f"  50 iters : {per_call * 30 * 50 / 3600:.1f} h")
print(f"  result   : {result}")
print()

# ── scale num_iterations to your actual timing ────────────────────────────────
# If per_call < 5s  → 50 iterations is fine (~2.5h)
# If per_call 5–15s → drop to 20 iterations, use more particles (40)
# If per_call > 15s → run on a subset of events first (200 events) to validate,
#                     then scale up

pareto = mopso.optimize(num_iterations=20)
print(f"\nDone. Pareto front: {len(pareto)} solutions")


