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
from objective import _eval_event, _shared_energy
from config import PARAM_NAMES
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

def _run_events(events, cee_params, che_params):
    """Run CLUEstering on all events with given params, return list of result dicts."""
    obj_module._EVENTS = events
    results = []
    t0 = time.perf_counter()
    for i in range(len(events)):
        results.append(_eval_event((i, list(cee_params), list(che_params))))
        if (i + 1) % 10 == 0 or (i + 1) == len(events):
            elapsed = time.perf_counter() - t0
            print(f"  {i+1}/{len(events)} events  {elapsed:.1f}s elapsed  "
                  f"({elapsed/(i+1)*1000:.1f}ms/event)", flush=True)
    return results


# ── Raw data collection ───────────────────────────────────────────────────────

def collect_raw_data(events, results):
    """
    Flatten CLUEstering output into three DataFrames (one row per object).

    Returns
    -------
    tracksters_df : [ptype, cp_energy, score]   — one row per trackster
    cps_df        : [ptype, cp_energy, is_efficient] — one row per sim CP
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

        # Per-CP efficiency: True if any assigned trackster has shared/cp > 0.5
        cp_efficient = {cp['shower_id']: False for cp in sim_showers}
        for t_id, trackster in enumerate(tracksters):
            best_cp_id = assignments[t_id][0]
            cp = cp_by_id[best_cp_id]
            if cp['true_energy'] > 0:
                if _shared_energy(trackster, cp) / cp['true_energy'] > 0.5:
                    cp_efficient[best_cp_id] = True

        for cp in sim_showers:
            c_rows.append({
                'ptype':        ptype,
                'cp_energy':    cp['true_energy'],
                'is_efficient': int(cp_efficient[cp['shower_id']]),
            })

        for t_id, (best_cp_id, score) in assignments.items():
            cp = cp_by_id[best_cp_id]
            t_rows.append({
                'ptype':     ptype,
                'cp_energy': cp['true_energy'],
                'score':     score,
            })

        e_rows.append({
            'ptype':           ptype,
            'total_ev_energy': total_ev_energy,
            'n_reco':          len(tracksters),
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
        elif metric == 'efficiency':
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
    'purity':       ('Purity',        _purity_per_bin,
                     'Purity  (score < 0.2)',      'CP energy [GeV]'),
    'efficiency':   ('Efficiency',    _efficiency_per_bin,
                     'Efficiency  (shared E > 50 %)', 'CP energy [GeV]'),
    'number_ratio': ('Number ratio',  _number_ratio_per_bin,
                     'N reco / N sim',              'Total event LC energy [GeV]'),
}


def _one_fig(all_data_list, labels, ptype, metric, edges, output_dir):
    title, bin_fn, ylabel, xlabel = _METRIC_META[metric]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"{title} — {_PTYPE_LABEL[ptype]}", fontsize=12, fontweight="bold")

    for i, (data, label) in enumerate(zip(all_data_list, labels)):
        tracksters_df, cps_df, events_df = data
        centers, vals, errs = bin_fn(
            tracksters_df if metric == 'purity' else
            cps_df        if metric == 'efficiency' else events_df,
            ptype, edges,
        )
        valid = ~np.isnan(vals)
        ax.errorbar(
            centers[valid], vals[valid], yerr=errs[valid],
            fmt="o-", label=label,
            color=COLORS[i % len(COLORS)],
            markersize=4, capsize=3, linewidth=1.5,
        )

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
        for metric in ('purity', 'efficiency', 'number_ratio'):
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

    # Load events
    print("Loading events...", flush=True)
    t0 = time.perf_counter()
    events = load_events(data_dir=args.data_dir)
    print(f"Events loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    if args.n_events is not None:
        ee = [e for e in events if e['particle_type'] == 'ee'][:args.n_events]
        pi = [e for e in events if e['particle_type'] == 'pi'][:args.n_events]
        events = [e for pair in zip(ee, pi) for e in pair]
        print(f"Using {len(ee)} ee + {len(pi)} pi = {len(events)} events\n", flush=True)

    # Run CLUEstering for each chosen row, collect DataFrames
    all_data_list = []
    labels        = []

    for row_num in args.rows:
        row = ranked[ranked["rank"] == row_num].iloc[0]
        cee_params = [row[p] for p in PARAM_NAMES[:5]]
        che_params = [row[p] for p in PARAM_NAMES[5:]]
        label = f"Row {row_num}  (score={row['score']:.3f})"
        print(f"\nRunning CLUEstering for {label}...")

        results = _run_events(events, cee_params, che_params)
        tracksters_df, cps_df, events_df = collect_raw_data(events, results)
        all_data_list.append((tracksters_df, cps_df, events_df))
        labels.append(label)

        n_ee = sum(1 for r in results if not r['infeasible'] and r['particle_type'] == 'ee')
        n_pi = sum(1 for r in results if not r['infeasible'] and r['particle_type'] == 'pi')
        print(f"  feasible events: {n_ee} ee, {n_pi} pi")

    print("\nGenerating plots...")
    make_plots(all_data_list, labels, output_dir)

    total = len(list(output_dir.glob("*.png")))
    print(f"\nDone — {total} plots in {output_dir}")


if __name__ == "__main__":
    main()
