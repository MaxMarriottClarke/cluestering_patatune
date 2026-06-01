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

hep.style.use("CMS")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

RESULTS_DIR = Path("results")
OUTPUT_DIR  = Path("/eos/user/m/mmarriot/php-plots/cluestering")

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
    "density_radius_cee":   r"$\rho_r^\mathrm{CEE}$",
    "min_density_cee":      r"$\rho_\mathrm{min}^\mathrm{CEE}$",
    "outlier_distance_cee": r"$d_\mathrm{out}^\mathrm{CEE}$",
    "seeding_distance_cee": r"$d_\mathrm{seed}^\mathrm{CEE}$",
    "w_z_cee":              r"$w_z^\mathrm{CEE}$",
    "density_radius_che":   r"$\rho_r^\mathrm{CHE}$",
    "min_density_che":      r"$\rho_\mathrm{min}^\mathrm{CHE}$",
    "outlier_distance_che": r"$d_\mathrm{out}^\mathrm{CHE}$",
    "seeding_distance_che": r"$d_\mathrm{seed}^\mathrm{CHE}$",
    "w_z_che":              r"$w_z^\mathrm{CHE}$",
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
    return df, info


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

def fig_objectives(df, scores):
    """
    3 panels: F1/F2/F3 — each plots ee vs pi value.
    Points coloured by balanced rank; top-20 labelled with rank number.
    CLUE3D sits at (1, 1) in every panel.
    """
    f = df[OBJ_NAMES].values
    ranks = np.argsort(np.argsort(scores)) + 1  # per-row rank in original df order

    pairs  = [(0, 3), (1, 4), (2, 5)]
    titles = ["F1: Impurity", "F2: Efficiency deficit", "F3: Fragmentation"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hep.cms.label("Preliminary", data=False, ax=axes[0],
                  com=None, lumi=None, loc=0, fontsize=11)
    fig.suptitle(
        "Pareto Front — objective space  (ratio to CLUE3D, < 1 = better)\n"
        "Colour = balanced score (dark = best).  Labels = rank for top-20.",
        fontsize=12, fontweight="bold", y=1.05)

    sc = None
    for ax, (i, j), title in zip(axes, pairs, titles):
        xi, xj = f[:, i], f[:, j]

        sc = ax.scatter(xi, xj, c=scores, cmap="viridis_r",
                        s=55, alpha=0.85, zorder=3,
                        linewidths=0.4, edgecolors="white")

        # Label top-20 with rank numbers
        top20_mask = ranks <= 20
        for k in np.where(top20_mask)[0]:
            ax.annotate(str(ranks[k]), (xi[k], xj[k]),
                        fontsize=6, ha="center", va="center",
                        color="white", fontweight="bold", zorder=5)

        # CLUE3D at (1, 1)
        ax.axvline(1.0, color=CLUE3D_COLOR, lw=1.4, ls="--", alpha=0.7, zorder=2)
        ax.axhline(1.0, color=CLUE3D_COLOR, lw=1.4, ls="--", alpha=0.7, zorder=2)
        ax.scatter(1.0, 1.0, marker="P", s=200, color=CLUE3D_COLOR,
                   edgecolors="k", linewidths=0.8, zorder=6, label="CLUE3D")

        ax.set_xlabel(OBJ_LABELS[i], fontsize=10)
        ax.set_ylabel(OBJ_LABELS[j], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.013, pad=0.02)
    cbar.set_label("Balanced score (lower = better)", fontsize=9)
    fig.tight_layout()
    return fig


# ── Plot 2: Parameter space ───────────────────────────────────────────────────

def fig_parameters(df, info):
    """
    2 rows (CEE, CHE) × 5 cols — violin + strip plot per parameter.
    Grey dotted lines show the optimisation bounds.
    """
    lower = np.array(info.get("lower_bounds", [np.nan] * 10), dtype=float)
    upper = np.array(info.get("upper_bounds", [np.nan] * 10), dtype=float)

    fig, axes = plt.subplots(2, 5, figsize=(18, 8))
    hep.cms.label("Preliminary", data=False, ax=axes[0, 0],
                  com=None, lumi=None, loc=0, fontsize=10)
    fig.suptitle("Parameter Distributions across the Pareto Front",
                 fontsize=13, fontweight="bold", y=1.02)

    colors = {"CEE": "#4a90d9", "CHE": "#e88c2a"}

    for row, (subdet, param_slice, offset) in enumerate([
        ("CEE", PARAM_NAMES[:5], 0),
        ("CHE", PARAM_NAMES[5:], 5),
    ]):
        c = colors[subdet]
        for col, pname in enumerate(param_slice):
            ax = axes[row, col]
            gi = offset + col
            vals = df[pname].values
            lb, ub = lower[gi], upper[gi]

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
            ax.set_ylabel("Value" if col == 0 else "")

            ax.text(0.97, 0.97,
                    f"med={q50:.2f}\n[{q25:.2f},  {q75:.2f}]",
                    transform=ax.transAxes, va="top", ha="right", fontsize=7,
                    bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        axes[row, 0].set_ylabel(f"{subdet}\nValue", fontsize=10)

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

    df, info = load_results()
    print(f"Loaded {len(df)} Pareto solutions\n")

    ranked, scores = rank_solutions(df)
    print_all(ranked)

    ranked.to_csv(OUTPUT_DIR / "all_solutions_ranked.csv", index=False)
    print(f"  saved  all_solutions_ranked.csv\n")

    print("Generating plots...")
    save(fig_objectives(df, scores), "01_objectives.png")
    save(fig_parameters(df, info),   "02_parameters.png")

    total = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\nDone — {total} plots in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
