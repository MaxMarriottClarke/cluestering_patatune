"""
Validation and visualisation of CLUEstering MOPSO Pareto results.

Reads from RESULTS_DIR (default: results/) and writes plots to OUTPUT_DIR.
Both directories can be overridden on the command line:

    python validate_results.py --results-dir results_che_newf3 \\
                               --output-dir /eos/.../cluestering_che_newf3
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mplhep as hep
from scipy.stats import spearmanr

hep.style.use("CMS")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

RESULTS_DIR = Path("results_che_newf3")
OUTPUT_DIR  = Path("/eos/user/m/mmarriot/php-plots/cluestering/parameters2")

PARAM_NAMES = [
    "density_radius_cee", "min_density_cee", "outlier_distance_cee",
    "seeding_distance_cee", "w_z_cee",
    "density_radius_che", "min_density_che", "outlier_distance_che",
    "seeding_distance_che", "w_z_che",
]
OBJ_NAMES = ["R1_ee", "R2_ee", "R3_ee", "R1_pi", "R2_pi", "R3_pi"]
OBJ_LABELS = [
    "F1 impurity (ee)", "F2 eff. deficit (ee)", "F3 fragmentation (ee)",
    "F1 impurity (ππ)", "F2 eff. deficit (ππ)", "F3 fragmentation (ππ)",
]

PARAM_LABEL = {
    "density_radius_cee":   r"$d_c$",
    "min_density_cee":      r"$\rho_\mathrm{min}$",
    "outlier_distance_cee": r"$d_m$",
    "seeding_distance_cee": r"$d_o$",
    "w_z_cee":              r"$w_z$",
    "density_radius_che":   r"$d_c$",
    "min_density_che":      r"$\rho_\mathrm{min}$",
    "outlier_distance_che": r"$d_m$",
    "seeding_distance_che": r"$d_o$",
    "w_z_che":              r"$w_z$",
}

CLUE3D_COLOR = "#9b59b6"


def save(fig, name):
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved  {path.name}")
    plt.close(fig)


def load_results():
    df = pd.read_csv(RESULTS_DIR / "pareto_front.csv")
    with open(RESULTS_DIR / "run_info.json") as f:
        info = json.load(f)
    df_cee = pd.read_csv(RESULTS_DIR / "staged_cee_pareto.csv")
    with open(RESULTS_DIR / "staged_cee_run_info.json") as f:
        info_cee = json.load(f)
    df_che = pd.read_csv(RESULTS_DIR / "staged_che_pareto.csv")
    with open(RESULTS_DIR / "staged_che_run_info.json") as f:
        info_che = json.load(f)
    return df, info, df_cee, info_cee, df_che, info_che


def rank_solutions(df):
    """Rank all solutions by normalised sum of all 6 objectives (lower = better)."""
    f = df[OBJ_NAMES].values
    f_norm = (f - f.min(0)) / (f.max(0) - f.min(0) + 1e-12)
    scores = f_norm.sum(axis=1)
    order = np.argsort(scores)
    ranked = df.iloc[order].copy().reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    ranked.insert(1, "score", scores[order])
    return ranked, scores


def print_all(ranked):
    W = 155
    print("\n" + "═" * W)
    print("  All Pareto solutions — ranked by balanced score  (ratios < 1.0 beat CLUE3D)")
    print("═" * W)
    hdr_obj   = "  ".join(f"{n:>7}" for n in OBJ_NAMES)
    hdr_param = "  ".join(f"{n[:9]:>9}" for n in PARAM_NAMES)
    print(f"  {'#':>3}  {'score':>6}  {hdr_obj}  {hdr_param}")
    print("─" * W)
    for _, row in ranked.iterrows():
        obj_str   = "  ".join(f"{row[c]:>7.3f}" for c in OBJ_NAMES)
        param_str = "  ".join(f"{row[c]:>9.3f}" for c in PARAM_NAMES)
        print(f"  {int(row['rank']):>3}  {row['score']:>6.3f}  {obj_str}  {param_str}")
    print("─" * W)
    print(f"  {'C3D':>3}  {'—':>6}  " + "  ".join(f"{'1.000':>7}" for _ in OBJ_NAMES)
          + "  (CLUE3D reference — all ratios = 1.0 by definition)")
    print("═" * W + "\n")


# ── Plot 1: Objective space ───────────────────────────────────────────────────

def _staged_scores(df, obj_cols):
    f = df[obj_cols].values
    f_norm = (f - f.min(0)) / (f.max(0) - f.min(0) + 1e-12)
    return f_norm.sum(axis=1)


def fig_objectives(df_cee, df_che):
    col_pairs  = [(0, 1), (0, 2), (1, 2)]
    col_titles = ["F1 vs F2", "F1 vs F3", "F2 vs F3"]

    ee_objs = OBJ_NAMES[:3]
    pi_objs = OBJ_NAMES[3:]
    rows = [
        ("ee", df_cee, ee_objs, OBJ_LABELS[:3]),
        ("ππ", df_che, pi_objs, OBJ_LABELS[3:]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(
        "Staged Pareto fronts — objective space  (ratio to CLUE3D, lower = better)",
        fontsize=13, fontweight="bold")

    sc = None
    for row_idx, (label, df, obj_cols, obj_labels) in enumerate(rows):
        f      = df[obj_cols].values
        scores = _staged_scores(df, obj_cols)
        ranks  = np.argsort(np.argsort(scores)) + 1

        for col_idx, ((a, b), title) in enumerate(zip(col_pairs, col_titles)):
            ax = axes[row_idx, col_idx]
            xi, xj = f[:, a], f[:, b]

            sc = ax.scatter(xi, xj, c=scores, cmap="viridis_r",
                            s=55, alpha=0.85, zorder=3,
                            linewidths=0.4, edgecolors="white")

            top20_mask = ranks <= 20
            for k in np.where(top20_mask)[0]:
                ax.annotate(str(ranks[k]), (xi[k], xj[k]),
                            fontsize=6, ha="center", va="center",
                            color="white", fontweight="bold", zorder=5)

            ax.axvline(1.0, color=CLUE3D_COLOR, lw=1.4, ls="--", alpha=0.7, zorder=2)
            ax.axhline(1.0, color=CLUE3D_COLOR, lw=1.4, ls="--", alpha=0.7, zorder=2)
            ax.scatter(1.0, 1.0, marker="P", s=200, color=CLUE3D_COLOR,
                       edgecolors="k", linewidths=0.8, zorder=6, label="CLUE3D")

            ax.set_xlim(0, min(max(xi.max(), 1.0) * 1.05, 10))
            ax.set_ylim(0, min(max(xj.max(), 1.0) * 1.05, 10))
            ax.set_xlabel(obj_labels[a], fontsize=10)
            ax.set_ylabel(obj_labels[b], fontsize=10)
            ax.legend(fontsize=8, loc="upper right")

            if row_idx == 0:
                ax.set_title(title, fontsize=11)

        axes[row_idx, 0].annotate(
            label, xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-50, 0), textcoords="offset points",
            fontsize=13, fontweight="bold", va="center", ha="right")

    fig.subplots_adjust(right=0.87, left=0.1)
    cax = fig.add_axes([0.89, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Balanced score (lower = better)", fontsize=9)
    return fig


# ── Plot 2: Parameter space ───────────────────────────────────────────────────

def fig_parameters(df_cee, info_cee, df_che, info_che):
    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    fig.suptitle("Parameter Distributions across the Pareto Front",
                 fontsize=13, fontweight="bold")

    colors = {"CEE": "#4a90d9", "CHE": "#e88c2a"}
    rows = [
        ("CEE", PARAM_NAMES[:5], df_cee, info_cee),
        ("CHE", PARAM_NAMES[5:], df_che, info_che),
    ]

    for row_idx, (subdet, param_slice, df, info) in enumerate(rows):
        c = colors[subdet]
        lower = np.array(info.get("lower_bounds", [np.nan] * 5), dtype=float)
        upper = np.array(info.get("upper_bounds", [np.nan] * 5), dtype=float)

        for col, pname in enumerate(param_slice):
            ax = axes[row_idx, col]
            vals = df[pname].values
            lb, ub = lower[col], upper[col]

            if vals.std() > 1e-6:
                vp = ax.violinplot(vals, positions=[0], showmedians=False,
                                   showextrema=False, widths=0.7)
                for body in vp["bodies"]:
                    body.set_facecolor(c)
                    body.set_alpha(0.4)

            jitter = np.random.default_rng(42).uniform(-0.18, 0.18, len(vals))
            ax.scatter(jitter, vals, s=14, alpha=0.55, color=c,
                       zorder=3, linewidths=0)

            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(0, q25, q75, color="navy", lw=5, alpha=0.4, zorder=4)
            ax.scatter(0, q50, s=55, color="navy", zorder=5)

            if not (np.isnan(lb) or np.isnan(ub)):
                ax.axhline(lb, color="grey", lw=0.9, ls=":", alpha=0.6)
                ax.axhline(ub, color="grey", lw=0.9, ls=":", alpha=0.6)
                ax.set_ylim(lb - 0.1 * (ub - lb), ub + 0.1 * (ub - lb))

            ax.set_title(PARAM_LABEL[pname], fontsize=11)
            ax.set_xlim(-0.6, 0.6)
            ax.set_xticks([])

            ax.text(0.97, 0.97,
                    f"med={q50:.2f}\n[{q25:.2f},  {q75:.2f}]",
                    transform=ax.transAxes, va="top", ha="right", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        axes[row_idx, 0].set_ylabel(f"{subdet}\nValue", fontsize=10)

    fig.tight_layout()
    return fig


# ── Plot 3: Parameter–objective correlation matrix ───────────────────────────

PARAM_ROW_LABELS = [
    r"$d_c$ (CEE)",          r"$\rho_\mathrm{min}$ (CEE)", r"$d_m$ (CEE)",
    r"$d_o$ (CEE)",          r"$w_z$ (CEE)",
    r"$d_c$ (CHE)",          r"$\rho_\mathrm{min}$ (CHE)", r"$d_m$ (CHE)",
    r"$d_o$ (CHE)",          r"$w_z$ (CHE)",
]

OBJ_COL_LABELS = [
    r"$F_1$ (ee)", r"$F_2$ (ee)", r"$F_3$ (ee)",
    r"$F_1$ (ππ)", r"$F_2$ (ππ)", r"$F_3$ (ππ)",
]


def fig_correlation(df_cee, df_che):
    n_params = len(PARAM_NAMES)   # 10
    n_objs   = len(OBJ_NAMES)     # 6

    rho  = np.full((n_params, n_objs), np.nan)
    pval = np.full((n_params, n_objs), np.nan)

    # Top-left block: CEE params (rows 0-4) vs ee objectives (cols 0-2)
    for i, p in enumerate(PARAM_NAMES[:5]):
        for j, o in enumerate(OBJ_NAMES[:3]):
            r, pv = spearmanr(df_cee[p], df_cee[o])
            rho[i, j]  = r
            pval[i, j] = pv

    # Bottom-right block: CHE params (rows 5-9) vs pi objectives (cols 3-5)
    for i, p in enumerate(PARAM_NAMES[5:]):
        for j, o in enumerate(OBJ_NAMES[3:]):
            r, pv = spearmanr(df_che[p], df_che[o])
            rho[5 + i, 3 + j]  = r
            pval[5 + i, 3 + j] = pv

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.suptitle(
        "Spearman Correlation: parameters vs. objectives  (* = p < 0.05)",
        fontsize=12, fontweight="bold")

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#d8d8d8")
    rho_masked = np.ma.masked_where(np.isnan(rho), rho)
    im = ax.imshow(rho_masked, cmap=cmap, vmin=-1, vmax=1, aspect="auto")

    for i in range(n_params):
        for j in range(n_objs):
            if np.isnan(rho[i, j]):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=10, color="#aaaaaa")
            else:
                star = "*" if pval[i, j] < 0.05 else ""
                ax.text(j, i, f"{rho[i, j]:.2f}{star}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(rho[i, j]) > 0.5 else "black")

    ax.set_xticks(range(n_objs))
    ax.set_xticklabels(OBJ_COL_LABELS, fontsize=10)
    ax.set_yticks(range(n_params))
    ax.set_yticklabels(PARAM_ROW_LABELS, fontsize=10)

    ax.set_xticks(np.arange(-0.5, n_objs), minor=True)
    ax.set_yticks(np.arange(-0.5, n_params), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Dividers separating the two blocks
    ax.axhline(4.5, color="black", lw=2.0)
    ax.axvline(2.5, color="black", lw=2.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman ρ", fontsize=10)

    fig.tight_layout()
    return fig


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global RESULTS_DIR, OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Validate and plot CLUEstering MOPSO Pareto results.",
    )
    parser.add_argument(
        '--results-dir', default=str(RESULTS_DIR),
        help=f'Directory containing pareto_front.csv and run_info.json '
             f'(default: {RESULTS_DIR})',
    )
    parser.add_argument(
        '--output-dir', default=str(OUTPUT_DIR),
        help=f'Directory for plots and ranked CSV (default: {OUTPUT_DIR})',
    )
    args = parser.parse_args()

    RESULTS_DIR = Path(args.results_dir)
    OUTPUT_DIR  = Path(args.output_dir)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Results → {RESULTS_DIR}")
    print(f"Output  → {OUTPUT_DIR}\n")

    df, info, df_cee, info_cee, df_che, info_che = load_results()
    print(f"Loaded {len(df)} joint Pareto solutions, "
          f"{len(df_cee)} staged-CEE, {len(df_che)} staged-CHE\n")

    ranked, scores = rank_solutions(df)
    print_all(ranked)

    ranked.to_csv(OUTPUT_DIR / "all_solutions_ranked.csv", index=False)
    print(f"  saved  all_solutions_ranked.csv\n")

    print("Generating plots...")
    save(fig_objectives(df_cee, df_che),                  "01_objectives.png")
    save(fig_parameters(df_cee, info_cee, df_che, info_che), "02_parameters.png")
    save(fig_correlation(df_cee, df_che),                 "03_correlation.png")

    total = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\nDone — {total} plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
