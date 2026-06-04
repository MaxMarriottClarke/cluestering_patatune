"""
Physics performance plots for CLUEstering parameter sets.

Ranks all solutions from pareto_front.csv by balanced score, then lets you
pick rows by rank number and produces 6 plots (purity, efficiency, number ratio
× EM and hadronic) with one curve per chosen parameter set.

Usage
-----
    # List available parameter sets and their rank:
    python physics_plots.py --results-dir results/ --list

    # Run with specific rows from the ranked table:
    python physics_plots.py --results-dir results/ --rows 1 3 5 \\
                            --output-dir /eos/user/m/mmarriot/php-plots/physics/

Metric definitions (all vs. simulated CP energy, split by particle type)
-------------------------------------------------------------------------
  Purity       : fraction of tracksters in bin with reco-to-sim score < 0.2.
                 Bin is the energy of the CP to which the trackster is assigned
                 (always by best reco-to-sim, regardless of score threshold).
  Efficiency   : fraction of CPs in bin for which at least one assigned
                 trackster has shared_energy / cp_energy > 0.5.
  Number ratio : (N reco tracksters) / (N sim CPs) per event, binned by
                 total LC energy summed over the whole event.
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

import objective as obj_module
from objective import _eval_event, _shared_energy, assign_tracksters_to_cps
from config import PARAM_NAMES, FILES
from data import load_events


# ── Style ─────────────────────────────────────────────────────────────────────

hep.style.use("CMS")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

OBJ_NAMES = ["R1_ee", "R2_ee", "R3_ee", "R1_pi", "R2_pi", "R3_pi"]
N_BINS    = 5

# Only count tracksters built from more than 2 layer clusters. Drops the tiny
# leakage blobs (mostly 1-2 LC CHE tracksters in ee events) that otherwise
# pollute purity / number ratio. Applied to both CLUEstering and CLUE3D.
MIN_TRACKSTER_LCS = 3


# ── Ranking ───────────────────────────────────────────────────────────────────

def _rank_solutions(results_dir):
    df = pd.read_csv(Path(results_dir) / "pareto_front.csv")
    f      = df[OBJ_NAMES].values
    f_norm = (f - f.min(0)) / (f.max(0) - f.min(0) + 1e-12)
    scores = f_norm.sum(axis=1)
    order  = np.argsort(scores)
    ranked = df.iloc[order].copy().reset_index(drop=True)
    ranked.insert(0, "rank",  range(1, len(ranked) + 1))
    ranked.insert(1, "score", scores[order])
    return ranked


def print_table(ranked):
    W = 155
    print("\n" + "=" * W)
    print("  All Pareto solutions — ranked by balanced score  (ratios < 1.0 beat CLUE3D)")
    print("=" * W)
    hdr_obj   = "  ".join(f"{n:>7}" for n in OBJ_NAMES)
    hdr_param = "  ".join(f"{n[:9]:>9}" for n in PARAM_NAMES)
    print(f"  {'#':>3}  {'score':>6}  {hdr_obj}  {hdr_param}")
    print("-" * W)
    for _, row in ranked.iterrows():
        obj_str   = "  ".join(f"{row[c]:>7.3f}" for c in OBJ_NAMES)
        param_str = "  ".join(f"{row[c]:>9.3f}" for c in PARAM_NAMES)
        print(f"  {int(row['rank']):>3}  {row['score']:>6.3f}  {obj_str}  {param_str}")
    print("=" * W + "\n")


# ── CLUEstering evaluation ────────────────────────────────────────────────────

def _run_events(events, cee_params, che_params, tag=''):
    """Run CLUEstering on a homogeneous list of events, return result dicts."""
    obj_module._EVENTS = events
    results = []
    n  = len(events)
    t0 = time.perf_counter()
    for i in range(n):
        results.append(_eval_event((i, list(cee_params), list(che_params))))
        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.perf_counter() - t0
            pfx = f"[{tag}] " if tag else ""
            print(f"  {pfx}{i+1}/{n} events  {elapsed:.1f}s  "
                  f"({elapsed/(i+1)*1000:.1f}ms/event)", flush=True)
    return results


# ── CLUE3D baseline loading ───────────────────────────────────────────────────

def load_clue3d_results(data_dir, ptype, events):
    """
    Load pre-computed CLUE3D tracksters from ROOT and compute assignments using
    exactly the same reco_to_sim_score / assign_tracksters_to_cps pipeline as
    CLUEstering, so the metrics are directly comparable.
    """
    import uproot
    import awkward as ak

    path = f"{data_dir}/{FILES[ptype]}"
    f    = uproot.open(path)
    tree = f['ticlDumper/hltTiclTrackstersCLUE3DHigh']

    vtx_idx = tree['vertices_indexes'].array()
    vtx_eng = tree['vertices_energy'].array()
    n_track = tree['NTracksters'].array()

    results = []
    for ev_idx, event in enumerate(events):
        n = int(n_track[ev_idx])
        tracksters = [
            {
                'lc_indexes':  ak.to_numpy(vtx_idx[ev_idx][t]).astype(np.int64),
                'lc_energies': ak.to_numpy(vtx_eng[ev_idx][t]).astype(np.float32),
            }
            for t in range(n)
        ]
        infeasible  = n < 2
        assignments = ({} if infeasible
                       else assign_tracksters_to_cps(tracksters, event['sim_showers']))
        results.append({
            'infeasible':    infeasible,
            'n_tracksters':  n,
            'tracksters':    tracksters,
            'assignments':   assignments,
            'sim_showers':   event['sim_showers'],
            'particle_type': ptype,
        })
    return results


# ── Raw data collection ───────────────────────────────────────────────────────

def collect_raw_data(events, results):
    """
    Flatten CLUEstering output into three DataFrames (one row per object).

    Returns
    -------
    tracksters_df : [ptype, cp_energy, score]        — one row per trackster
    cps_df        : [ptype, cp_energy, is_efficient]  — one row per sim CP
    events_df     : [ptype, total_ev_energy, n_reco, n_sim] — one row per event
    """
    t_rows, c_rows, e_rows = [], [], []

    for ev_idx, result in enumerate(results):
        if result['infeasible']:
            continue

        ptype       = result['particle_type']
        event       = events[ev_idx]
        sim_showers = result['sim_showers']
        tracksters  = result['tracksters']
        assignments = result['assignments']

        total_ev_energy = float(event['all_lcs']['energy'].sum())
        cp_by_id = {cp['shower_id']: cp for cp in sim_showers}

        # Keep only tracksters with > 2 layer clusters; everything below uses
        # this filtered set so metrics ignore the leakage blobs.
        kept = [t_id for t_id, t in enumerate(tracksters)
                if len(t['lc_indexes']) >= MIN_TRACKSTER_LCS]

        # CP efficiency: sum shared energies of all tracksters assigned by reco-to-sim score
        cp_shared_sum = {cp['shower_id']: 0.0 for cp in sim_showers}
        for t_id in kept:
            trackster  = tracksters[t_id]
            best_cp_id = assignments[t_id][0]
            cp = cp_by_id[best_cp_id]
            cp_shared_sum[best_cp_id] += _shared_energy(trackster, cp)

        # Trackster efficiency: for each CP, find the trackster with the greatest
        # shared LC energy, then require that trackster's reco-to-sim assignment
        # also points back to this CP (two-way match) and shared/cp_energy > 0.5.
        cp_max_shared  = {cp['shower_id']: 0.0  for cp in sim_showers}
        cp_best_t_id   = {cp['shower_id']: None for cp in sim_showers}
        for t_id in kept:
            trackster = tracksters[t_id]
            for cp in sim_showers:
                shared = _shared_energy(trackster, cp)
                if shared > cp_max_shared[cp['shower_id']]:
                    cp_max_shared[cp['shower_id']] = shared
                    cp_best_t_id[cp['shower_id']] = t_id
        cp_efficient = {}
        for cp in sim_showers:
            best_t = cp_best_t_id[cp['shower_id']]
            if best_t is None or cp['true_energy'] <= 0:
                cp_efficient[cp['shower_id']] = False
            else:
                cp_efficient[cp['shower_id']] = (
                    cp_max_shared[cp['shower_id']] / cp['true_energy'] > 0.5
                    and assignments[best_t][0] == cp['shower_id']
                )

        for cp in sim_showers:
            cp_eff = (cp_shared_sum[cp['shower_id']] / cp['true_energy']
                      if cp['true_energy'] > 0 else np.nan)
            c_rows.append({
                'ptype':         ptype,
                'cp_energy':     cp['true_energy'],
                'is_efficient':  int(cp_efficient[cp['shower_id']]),
                'cp_efficiency': cp_eff,
            })

        for t_id in kept:
            best_cp_id, score = assignments[t_id]
            cp = cp_by_id[best_cp_id]
            t_rows.append({
                'ptype':     ptype,
                'cp_energy': cp['true_energy'],
                'score':     score,
            })

        e_rows.append({
            'ptype':           ptype,
            'total_ev_energy': total_ev_energy,
            'n_reco':          len(kept),
            'n_sim':           len(sim_showers),
        })

    return pd.DataFrame(t_rows), pd.DataFrame(c_rows), pd.DataFrame(e_rows)


# ── Per-bin metric computation ────────────────────────────────────────────────

def _purity_per_bin(tracksters_df, ptype, edges):
    """sum(score < 0.2) / count per bin."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(len(centers), np.nan)
    errs = np.full(len(centers), np.nan)
    sub = tracksters_df[tracksters_df['ptype'] == ptype]
    if sub.empty:
        return centers, vals, errs
    sub = sub.copy()
    sub['bin'] = pd.cut(sub['cp_energy'], bins=edges, right=False,
                        include_lowest=True, labels=False)
    for i, grp in sub.groupby('bin', observed=False):
        n = len(grp)
        if n == 0:
            continue
        p = (grp['score'] < 0.2).sum() / n
        vals[int(i)] = p
        errs[int(i)] = np.sqrt(p * (1 - p) / n) if n > 1 else 0.0
    return centers, vals, errs


def _cp_efficiency_per_bin(cps_df, ptype, edges):
    """mean(sum_shared_energy / cp_energy) per CP per bin."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(len(centers), np.nan)
    errs = np.full(len(centers), np.nan)
    sub = cps_df[cps_df['ptype'] == ptype]
    if sub.empty:
        return centers, vals, errs
    sub = sub.copy()
    sub['bin'] = pd.cut(sub['cp_energy'], bins=edges, right=False,
                        include_lowest=True, labels=False)
    for i, grp in sub.groupby('bin', observed=False):
        n = len(grp)
        if n == 0:
            continue
        vals[int(i)] = grp['cp_efficiency'].mean()
        errs[int(i)] = grp['cp_efficiency'].std() / np.sqrt(n) if n > 1 else 0.0
    return centers, vals, errs


def _efficiency_per_bin(cps_df, ptype, edges):
    """sum(is_efficient) / count per bin."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(len(centers), np.nan)
    errs = np.full(len(centers), np.nan)
    sub = cps_df[cps_df['ptype'] == ptype]
    if sub.empty:
        return centers, vals, errs
    sub = sub.copy()
    sub['bin'] = pd.cut(sub['cp_energy'], bins=edges, right=False,
                        include_lowest=True, labels=False)
    for i, grp in sub.groupby('bin', observed=False):
        n = len(grp)
        if n == 0:
            continue
        p = grp['is_efficient'].sum() / n
        vals[int(i)] = p
        errs[int(i)] = np.sqrt(p * (1 - p) / n) if n > 1 else 0.0
    return centers, vals, errs


def _number_ratio_per_bin(events_df, ptype, edges):
    """sum(n_reco) / sum(n_sim) per bin."""
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(len(centers), np.nan)
    errs = np.full(len(centers), np.nan)
    sub = events_df[events_df['ptype'] == ptype]
    if sub.empty:
        return centers, vals, errs
    sub = sub.copy()
    sub['bin'] = pd.cut(sub['total_ev_energy'], bins=edges, right=False,
                        include_lowest=True, labels=False)
    for i, grp in sub.groupby('bin', observed=False):
        if len(grp) == 0:
            continue
        vals[int(i)] = grp['n_reco'].sum() / grp['n_sim'].sum()
        per_ev       = grp['n_reco'] / grp['n_sim']
        errs[int(i)] = per_ev.std() / np.sqrt(len(grp)) if len(grp) > 1 else 0.0
    return centers, vals, errs


def _make_edges(all_data_list, ptype, metric):
    """Bin edges from 1st–99th percentile of the energy axis, pooled across all param sets."""
    energies = []
    for tracksters_df, cps_df, events_df in all_data_list:
        if metric == 'purity':
            col = tracksters_df.loc[tracksters_df['ptype'] == ptype, 'cp_energy']
        elif metric in ('efficiency', 'cp_efficiency'):
            col = cps_df.loc[cps_df['ptype'] == ptype, 'cp_energy']
        else:
            col = events_df.loc[events_df['ptype'] == ptype, 'total_ev_energy']
        energies.extend(col.tolist())
    if not energies:
        return np.linspace(0, 1, N_BINS + 1)
    lo, hi = np.percentile(energies, [1, 99])
    return np.linspace(lo, hi, N_BINS + 1)


# ── Plotting ──────────────────────────────────────────────────────────────────

COLORS = plt.cm.tab10.colors

_PTYPE_LABEL = {'ee': 'EM (ee)', 'pi': 'Hadronic (ππ)'}
_METRIC_META = {
    'purity':         ('Purity',                _purity_per_bin,
                       'Purity  (score < 0.2)',                  'CP energy [GeV]'),
    'efficiency':     ('Trackster Efficiency',  _efficiency_per_bin,
                       'Trackster Efficiency  (best shared E > 50 % & reco-to-sim match)', 'CP energy [GeV]'),
    'cp_efficiency':  ('CP Efficiency',         _cp_efficiency_per_bin,
                       'CP Efficiency  (Σ shared E / CP E)', 'CP energy [GeV]'),
    'number_ratio':   ('Number ratio',          _number_ratio_per_bin,
                       'N reco / N sim',                         'Total event LC energy [GeV]'),
}


def _one_fig(all_data_list, labels, ptype, metric, edges, output_dir):
    title, bin_fn, ylabel, xlabel = _METRIC_META[metric]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{title} — {_PTYPE_LABEL[ptype]}", fontsize=12, fontweight="bold")

    color_idx = 0
    for data, label in zip(all_data_list, labels):
        tracksters_df, cps_df, events_df = data
        centers, vals, errs = bin_fn(
            tracksters_df if metric == 'purity' else
            cps_df        if metric in ('efficiency', 'cp_efficiency') else events_df,
            ptype, edges,
        )
        valid = ~np.isnan(vals)
        is_clue3d = label.startswith('CLUE3D')
        ax.errorbar(
            centers[valid], vals[valid], yerr=errs[valid],
            fmt='s--' if is_clue3d else 'o-',
            label=label,
            color='black' if is_clue3d else COLORS[color_idx % len(COLORS)],
            markersize=4, capsize=3, linewidth=1.5,
            zorder=3 if is_clue3d else 2,
        )
        if not is_clue3d:
            color_idx += 1

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=8, loc="best")

    fname = f"{metric}_{'ee' if ptype == 'ee' else 'pi'}.png"
    path  = output_dir / fname
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved  {path.name}")
    plt.close(fig)


def make_plots(all_data_list, labels, output_dir):
    for ptype in ('ee', 'pi'):
        for metric in ('purity', 'efficiency', 'cp_efficiency', 'number_ratio'):
            edges = _make_edges(all_data_list, ptype, metric)
            _one_fig(all_data_list, labels, ptype, metric, edges, output_dir)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Physics performance plots for CLUEstering parameter sets.",
    )
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing pareto_front.csv (default: results)")
    parser.add_argument("--data-dir", default="Data",
                        help="Directory containing ROOT files (default: Data)")
    parser.add_argument("--output-dir",
                        default="/eos/user/m/mmarriot/php-plots/physics",
                        help="Output directory for plots")
    parser.add_argument("--rows", nargs="+", type=int,
                        help="Rank numbers to evaluate (1-indexed)")
    parser.add_argument("--n-events", type=int, default=None,
                        help="Max events per particle type (default: all)")
    parser.add_argument("--list", action="store_true",
                        help="Print the ranked table and exit")
    args = parser.parse_args()

    ranked = _rank_solutions(args.results_dir)

    if args.list:
        print_table(ranked)
        return

    if not args.rows:
        parser.error("Specify --rows or --list")

    bad = [r for r in args.rows if r < 1 or r > len(ranked)]
    if bad:
        parser.error(f"Row(s) {bad} out of range (1–{len(ranked)})")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output  → {output_dir}\n")

    # Load events — keep ee and pi fully separate from here on
    print("Loading events...", flush=True)
    t0 = time.perf_counter()
    all_events = load_events(data_dir=args.data_dir)
    print(f"Events loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    ee_events = [e for e in all_events if e['particle_type'] == 'ee']
    pi_events = [e for e in all_events if e['particle_type'] == 'pi']
    if args.n_events is not None:
        ee_events = ee_events[:args.n_events]
        pi_events = pi_events[:args.n_events]
    print(f"Using {len(ee_events)} ee + {len(pi_events)} pi events\n", flush=True)

    # CLUE3D baseline — computed once, added as first curve on every plot
    print("Loading CLUE3D baseline...", flush=True)
    t0 = time.perf_counter()
    c3d_ee = load_clue3d_results(args.data_dir, 'ee', ee_events)
    c3d_pi = load_clue3d_results(args.data_dir, 'pi', pi_events)
    c3d_ee_t, c3d_ee_c, c3d_ee_e = collect_raw_data(ee_events, c3d_ee)
    c3d_pi_t, c3d_pi_c, c3d_pi_e = collect_raw_data(pi_events, c3d_pi)
    clue3d_data = (
        pd.concat([c3d_ee_t, c3d_pi_t], ignore_index=True),
        pd.concat([c3d_ee_c, c3d_pi_c], ignore_index=True),
        pd.concat([c3d_ee_e, c3d_pi_e], ignore_index=True),
    )
    n_c3d_ee = sum(1 for r in c3d_ee if not r['infeasible'])
    n_c3d_pi = sum(1 for r in c3d_pi if not r['infeasible'])
    print(f"CLUE3D loaded in {time.perf_counter() - t0:.1f}s  "
          f"(feasible: {n_c3d_ee} ee, {n_c3d_pi} pi)\n", flush=True)

    # Run CLUEstering for each chosen row, collect DataFrames
    all_data_list = [clue3d_data]
    labels        = ['CLUE3D']

    for row_num in args.rows:
        row = ranked[ranked["rank"] == row_num].iloc[0]
        cee_params = [row[p] for p in PARAM_NAMES[:5]]
        che_params = [row[p] for p in PARAM_NAMES[5:]]
        label = f"Row {row_num}  (score={row['score']:.3f})"
        print(f"\nRunning CLUEstering for {label}...")

        print("  ee events:", flush=True)
        ee_results = _run_events(ee_events, cee_params, che_params, tag='ee')
        print("  pi events:", flush=True)
        pi_results = _run_events(pi_events, cee_params, che_params, tag='pi')

        ee_t, ee_c, ee_e = collect_raw_data(ee_events, ee_results)
        pi_t, pi_c, pi_e = collect_raw_data(pi_events, pi_results)

        tracksters_df = pd.concat([ee_t, pi_t], ignore_index=True)
        cps_df        = pd.concat([ee_c, pi_c], ignore_index=True)
        events_df     = pd.concat([ee_e, pi_e], ignore_index=True)

        all_data_list.append((tracksters_df, cps_df, events_df))
        labels.append(label)

        n_ee = sum(1 for r in ee_results if not r['infeasible'])
        n_pi = sum(1 for r in pi_results if not r['infeasible'])
        print(f"  feasible: {n_ee} ee, {n_pi} pi")

    print("\nGenerating plots...")
    make_plots(all_data_list, labels, output_dir)

    total = len(list(output_dir.glob("*.png")))
    print(f"\nDone — {total} plots in {output_dir}")


if __name__ == "__main__":
    main()
