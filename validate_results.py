"""
Validation and visualisation of CLUEstering MOPSO Pareto results.

Reads from results/ and writes all plots to OUTPUT_DIR.

Dependencies: matplotlib, numpy, scipy, pandas, seaborn, mplhep
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.spatial import ConvexHull
import mplhep as hep

# ── Style ──────────────────────────────────────────────────────────────────────
hep.style.use("CMS")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#fafafa",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
})

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")
OUTPUT_DIR  = Path("/eos/user/m/mmarriot/php-plots/cluestering")

PARAM_NAMES = [
    "density_radius_cee",  "min_density_cee",  "outlier_distance_cee",
    "seeding_distance_cee", "w_z_cee",
    "density_radius_che",  "min_density_che",  "outlier_distance_che",
    "seeding_distance_che", "w_z_che",
]
OBJ_NAMES  = ["R1_ee", "R2_ee", "R3_ee", "R1_pi", "R2_pi", "R3_pi"]
OBJ_LABELS = [
    "F1 impurity (ee)",        "F2 eff. deficit (ee)",        "F3 fragmentation (ee)",
    "F1 impurity (ππ)",       "F2 eff. deficit (ππ)",       "F3 fragmentation (ππ)",
]
OBJ_UNITS  = ["ratio / CLUE3D"] * 6
# Grouping for 2-D projections: same metric, cross particle-type
OBJ_PAIRS  = [(0,3), (1,4), (2,5)]   # (ee metric, pi metric) per F1/F2/F3
OBJ_PAIR_TITLES = [
    ("F1: Impurity  ee vs ππ",        "F1 ee (ratio)", "F1 ππ (ratio)"),
    ("F2: Efficiency deficit  ee vs ππ", "F2 ee (ratio)", "F2 ππ (ratio)"),
    ("F3: Fragmentation  ee vs ππ",   "F3 ee (ratio)", "F3 ππ (ratio)"),
]

PARAM_LABELS = {
    "density_radius_cee":    r"$\rho_r^\mathrm{CEE}$",
    "min_density_cee":       r"$\rho_\mathrm{min}^\mathrm{CEE}$",
    "outlier_distance_cee":  r"$d_\mathrm{out}^\mathrm{CEE}$",
    "seeding_distance_cee":  r"$d_\mathrm{seed}^\mathrm{CEE}$",
    "w_z_cee":               r"$w_z^\mathrm{CEE}$",
    "density_radius_che":    r"$\rho_r^\mathrm{CHE}$",
    "min_density_che":       r"$\rho_\mathrm{min}^\mathrm{CHE}$",
    "outlier_distance_che":  r"$d_\mathrm{out}^\mathrm{CHE}$",
    "seeding_distance_che":  r"$d_\mathrm{seed}^\mathrm{CHE}$",
    "w_z_che":               r"$w_z^\mathrm{CHE}$",
}
PARAM_LABELS_SHORT = {
    "density_radius_cee":    "ρ_r (CEE)",
    "min_density_cee":       "ρ_min (CEE)",
    "outlier_distance_cee":  "d_out (CEE)",
    "seeding_distance_cee":  "d_seed (CEE)",
    "w_z_cee":               "w_z (CEE)",
    "density_radius_che":    "ρ_r (CHE)",
    "min_density_che":       "ρ_min (CHE)",
    "outlier_distance_che":  "d_out (CHE)",
    "seeding_distance_che":  "d_seed (CHE)",
    "w_z_che":               "w_z (CHE)",
}

BEST_LABELS  = [
    "Min R1_ee",    "Min R2_ee",    "Min R3_ee",
    "Min R1_pi", "Min R2_pi", "Min R3_pi",
    "Balanced\n(min norm. sum)",
]
BEST_COLORS  = ["#e6194b", "#3cb44b", "#4363d8",
                "#f032e6", "#f58231", "#42d4f4",
                "#000000"]
BEST_MARKERS = ["*", "D", "^", "P", "s", "X", "o"]

CLUE3D_COLOR  = "#9b59b6"
CLUE3D_MARKER = "P"          # thick plus
CLUE3D_LABEL  = "CLUE3D\n(baseline)"

# ── CLUE3D baseline ────────────────────────────────────────────────────────────
# CEE/CHE boundary must match config.py
_CEE_Z_BOUNDARY = 364.55
_DATA_DIR       = Path("Data")
_FILES          = {"ee": "e_energy.root", "pi": "pi_energy.root"}
_CLUE3D_BRANCH  = "ticlDumper/hltTiclTrackstersCLUE3DHigh"
_CLUE3D_CACHE   = RESULTS_DIR / "clue3d_baseline.json"


# ── I/O helpers ────────────────────────────────────────────────────────────────

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


def compute_clue3d_baseline():
    """
    Compute F1, F2, F3 for CLUE3D using the identical scoring logic as the
    optimisation (objective.py).  The scoring primitives are inlined here so
    that objective.py (which imports CLUEstering and would SIGILL on import on
    some nodes) is never touched.

    CLUE3D runs globally, not split by subdetector.  Each trackster is assigned
    to CEE or CHE by energy-majority using the same CEE_Z_BOUNDARY so F3 is
    computed identically.

    Returns dict with keys f1, f2, f3.
    """
    import uproot
    import awkward as ak

    # ── Load truth (simtrackstersCP) from ROOT directly — no CLUEstering ────────
    TRUTH_BRANCH = "ticlDumper/simtrackstersCP"
    TRUTH_COLS   = ["vertices_indexes", "vertices_x", "vertices_y",
                    "vertices_z", "vertices_energy", "vertices_multiplicity"]

    def _highest_cycle(file, branch):
        keys = [k for k in file.keys() if k.startswith(branch)]
        return file[max(keys, key=lambda k: int(k.split(";")[1]) if ";" in k else 0)]

    print("  Computing CLUE3D baseline — loading truth + CLUE3D tracksters...")

    # Load both files
    truth_raw  = {}
    clue3d_raw = {}
    for ptype, fname in _FILES.items():
        path = str(_DATA_DIR / fname)
        f = uproot.open(path)
        truth_raw [ptype] = _highest_cycle(f, TRUTH_BRANCH).arrays(TRUTH_COLS)
        clue3d_raw[ptype] = _highest_cycle(f, _CLUE3D_BRANCH).arrays(
            ["vertices_indexes", "vertices_energy", "vertices_z"]
        )

    n_per_type = min(len(truth_raw[p]["vertices_indexes"]) for p in _FILES)

    # ── Scoring primitives (identical to objective.py) ───────────────────────────
    def _reco_to_sim_score(lc_indexes, lc_energies, cp_set):
        total = float(np.asarray(lc_energies).sum())
        if total == 0:
            return 0.0
        impure = sum(float(e) for idx, e in zip(lc_indexes, lc_energies)
                     if idx not in cp_set)
        return impure / total

    def _assign(tracksters, sim_showers):
        assignments = {}
        for t_id, t in enumerate(tracksters):
            best_cp, best_score = None, np.inf
            for cp in sim_showers:
                cp_set = set(cp["lc_indexes"].tolist())
                s = _reco_to_sim_score(t["lc_indexes"], t["lc_energies"], cp_set)
                if s < best_score:
                    best_score, best_cp = s, cp["shower_id"]
            assignments[t_id] = (best_cp, best_score)
        return assignments

    def _shared_energy(t, cp):
        cp_set = set(cp["lc_indexes"].tolist())
        return float(sum(float(e) for idx, e in zip(t["lc_indexes"], t["lc_energies"])
                         if idx in cp_set))

    def _f1(results):
        per = []
        for r in results:
            if r["infeasible"]: return np.inf
            per.append(max(score for _, score in r["assignments"].values()))
        return float(np.mean(per))

    def _f2(results):
        per = []
        for r in results:
            if r["infeasible"]: return np.inf
            cp_shared = {cp["shower_id"]: 0.0 for cp in r["sim_showers"]}
            for t_id, t in enumerate(r["tracksters"]):
                best_cp_id = r["assignments"][t_id][0]
                cp = next(c for c in r["sim_showers"] if c["shower_id"] == best_cp_id)
                cp_shared[best_cp_id] += _shared_energy(t, cp)
            effs = [cp_shared[cp["shower_id"]] / cp["true_energy"]
                    if cp["true_energy"] > 0 else 0.0
                    for cp in r["sim_showers"]]
            per.append(1.0 - min(effs))
        return float(np.mean(per))

    def _f3(results):
        per = []
        for r in results:
            if r["infeasible"]: return np.inf
            counts = {}
            for t_id, (best_cp_id, _) in r["assignments"].items():
                subdet = r["tracksters"][t_id]["subdet"]
                key = (best_cp_id, subdet)
                counts[key] = counts.get(key, 0) + 1
            per.append(float(sum(max(0, c - 1) for c in counts.values())))
        return float(np.mean(per))

    # ── Build truth sim_showers per event (identical resolution to data.py) ──────
    def _build_sim_showers(arrays, ev_idx):
        n_sh = len(arrays["vertices_indexes"][ev_idx])
        lc_best_frac   = {}
        lc_best_shower = {}
        lc_coord       = {}
        for sh_id in range(n_sh):
            idxs  = ak.to_numpy(arrays["vertices_indexes"     ][ev_idx][sh_id]).astype(np.int64)
            ens   = ak.to_numpy(arrays["vertices_energy"      ][ev_idx][sh_id]).astype(np.float32)
            mults = ak.to_numpy(arrays["vertices_multiplicity"][ev_idx][sh_id]).astype(np.float32)
            xs    = ak.to_numpy(arrays["vertices_x"           ][ev_idx][sh_id]).astype(np.float32)
            ys    = ak.to_numpy(arrays["vertices_y"           ][ev_idx][sh_id]).astype(np.float32)
            zs    = ak.to_numpy(arrays["vertices_z"           ][ev_idx][sh_id]).astype(np.float32)
            for i, lc_idx in enumerate(idxs):
                m = float(mults[i])
                if m <= 0:
                    continue
                frac = 1.0 / m
                if lc_idx not in lc_best_frac or frac > lc_best_frac[lc_idx]:
                    lc_best_frac[lc_idx]   = frac
                    lc_best_shower[lc_idx] = sh_id
                if lc_idx not in lc_coord:
                    lc_coord[lc_idx] = (float(xs[i]), float(ys[i]),
                                        float(zs[i]), float(ens[i]))

        shower_lcs = {sh_id: [] for sh_id in range(n_sh)}
        for lc_idx, winner in lc_best_shower.items():
            shower_lcs[winner].append(lc_idx)

        sim_showers = []
        for sh_id in range(n_sh):
            lc_list = sorted(shower_lcs[sh_id])
            lc_arr  = np.array(lc_list, dtype=np.int64)
            if len(lc_arr) > 0:
                coords = np.array([lc_coord[idx] for idx in lc_arr], dtype=np.float32)
                lc_e   = coords[:, 3]
            else:
                lc_e = np.empty(0, dtype=np.float32)
            sim_showers.append({
                "shower_id":   sh_id,
                "lc_indexes":  lc_arr,
                "true_energy": float(lc_e.sum()),
            })
        return sim_showers

    # ── Evaluate every event, tracking per-type results separately ───────────────
    results_by_type = {"ee": [], "pi": [], "combined": []}

    for ev_idx in range(n_per_type):
        for ptype in ["ee", "pi"]:
            sim_showers = _build_sim_showers(truth_raw[ptype], ev_idx)

            arr = clue3d_raw[ptype]
            tracksters = []
            for tr_i in range(len(arr["vertices_indexes"][ev_idx])):
                idxs = ak.to_numpy(arr["vertices_indexes"][ev_idx][tr_i]).astype(np.int64)
                ens  = ak.to_numpy(arr["vertices_energy" ][ev_idx][tr_i]).astype(np.float32)
                zs   = ak.to_numpy(arr["vertices_z"      ][ev_idx][tr_i]).astype(np.float32)
                cee_e  = float(ens[np.abs(zs) < _CEE_Z_BOUNDARY].sum())
                che_e  = float(ens[np.abs(zs) >= _CEE_Z_BOUNDARY].sum())
                tracksters.append({
                    "lc_indexes":  idxs,
                    "lc_energies": ens,
                    "subdet":      "CEE" if cee_e >= che_e else "CHE",
                })

            n          = len(tracksters)
            infeasible = n < 2
            result = {
                "infeasible":   infeasible,
                "n_tracksters": n,
                "tracksters":   tracksters,
                "assignments":  {} if infeasible else _assign(tracksters, sim_showers),
                "sim_showers":  sim_showers,
            }
            results_by_type[ptype].append(result)
            results_by_type["combined"].append(result)

    out = {}
    for key, results in results_by_type.items():
        f1 = _f1(results)
        f2 = _f2(results)
        f3 = _f3(results)
        n_infeasible = sum(1 for r in results if r["infeasible"])
        n_tr_vals    = [r["n_tracksters"] for r in results]
        out[key] = {
            "f1": f1, "f2": f2, "f3": f3,
            "n_events":    len(results),
            "n_infeasible": n_infeasible,
            "mean_n_tracksters": float(np.mean(n_tr_vals)),
        }

    for key in ["ee", "pi", "combined"]:
        d = out[key]
        print(f"  CLUE3D [{key:>8s}] ({d['n_events']} events, "
              f"mean NTr={d['mean_n_tracksters']:.1f}):  "
              f"F1={d['f1']:.4f}  F2={d['f2']:.4f}  F3={d['f3']:.3f}")

    # Flat keys for backward compat + nested per_type
    return {
        "f1": out["combined"]["f1"],
        "f2": out["combined"]["f2"],
        "f3": out["combined"]["f3"],
        "per_type": {
            "ee":    out["ee"],
            "pi": out["pi"],
        },
    }


def load_clue3d_baseline(recompute=False):
    """Return cached baseline dict, computing it first if necessary."""
    if not recompute and _CLUE3D_CACHE.exists():
        with open(_CLUE3D_CACHE) as fh:
            baseline = json.load(fh)
        print(f"  Loaded CLUE3D baseline from cache: "
              f"F1={baseline['f1']:.4f}  F2={baseline['f2']:.4f}  F3={baseline['f3']:.3f}")
        return baseline
    baseline = compute_clue3d_baseline()
    with open(_CLUE3D_CACHE, "w") as fh:
        json.dump(baseline, fh, indent=2)
    print(f"  Saved baseline to {_CLUE3D_CACHE}")
    return baseline


def best_solutions(df):
    f      = df[OBJ_NAMES].values
    f_norm = (f - f.min(0)) / (f.max(0) - f.min(0) + 1e-12)
    indices = (
        [int(np.argmin(f[:, k])) for k in range(len(OBJ_NAMES))]
        + [int(np.argmin(f_norm.sum(1)))]
    )
    return dict(zip(BEST_LABELS, indices))


# ── Figure 1: 2-D Pareto projections with KDE contours ────────────────────────

def fig_pareto_2d(df, best, baseline=None):
    """
    Three panels: same metric for ee (x) vs ππ (y), one panel per F1/F2/F3.
    The CLUE3D reference point is always (1, 1) in every panel — shown as a
    dashed crosshair.  Solutions in the lower-left quadrant beat CLUE3D on
    both particle types for that metric.
    """
    f = df[OBJ_NAMES].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    hep.cms.label("Preliminary", data=False, ax=axes[0],
                  com=None, lumi=None, loc=0, fontsize=11)
    fig.suptitle("Pareto Front — ee vs ππ per metric  (ratio to CLUE3D, < 1 = better)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (i, j), (title, xl, yl) in zip(axes, OBJ_PAIRS, OBJ_PAIR_TITLES):
        xi, xj = f[:, i], f[:, j]

        # KDE contour background
        try:
            k   = gaussian_kde(np.vstack([xi, xj]))
            xg  = np.linspace(xi.min(), xi.max(), 80)
            yg  = np.linspace(xj.min(), xj.max(), 80)
            Xg, Yg = np.meshgrid(xg, yg)
            Z   = k(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contourf(Xg, Yg, Z, levels=8, cmap="Blues", alpha=0.35)
            ax.contour( Xg, Yg, Z, levels=8, colors="steelblue", alpha=0.4, linewidths=0.6)
        except Exception:
            pass

        # All Pareto points
        sc = ax.scatter(xi, xj, c=np.arange(len(df)), cmap="viridis",
                        s=45, alpha=0.85, zorder=3, linewidths=0.3, edgecolors="white")

        # Convex hull
        try:
            pts = np.column_stack([xi, xj])
            hull = ConvexHull(pts)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            ax.plot(pts[hull_pts, 0], pts[hull_pts, 1],
                    "k--", lw=1.0, alpha=0.4, zorder=2)
        except Exception:
            pass

        # Best solutions
        for (label, idx), mk, bc in zip(best.items(), BEST_MARKERS, BEST_COLORS):
            ax.scatter(xi[idx], xj[idx], marker=mk, s=200, zorder=6,
                       color=bc, edgecolors="k", linewidths=0.8)

        # CLUE3D reference at (1, 1) — always exact
        ax.axvline(1.0, color=CLUE3D_COLOR, lw=1.5, ls="--", alpha=0.8, zorder=4)
        ax.axhline(1.0, color=CLUE3D_COLOR, lw=1.5, ls="--", alpha=0.8, zorder=4)
        ax.scatter(1.0, 1.0, marker=CLUE3D_MARKER, s=220, zorder=8,
                   color=CLUE3D_COLOR, edgecolors="k", linewidths=0.9)
        ax.text(1.0, 1.0, " CLUE3D", fontsize=7.5, color=CLUE3D_COLOR,
                va="bottom", fontweight="bold", zorder=9)

        # Shade the "beats CLUE3D on both types" quadrant
        xlim = ax.get_xlim(); ylim = ax.get_ylim()
        ax.axvspan(xlim[0], 1.0, ymin=0,
                   ymax=(1.0 - ylim[0]) / (ylim[1] - ylim[0]) if ylim[1] > ylim[0] else 0.5,
                   alpha=0.06, color="green", zorder=0)

        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.set_title(title, fontsize=10, pad=4)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.015, pad=0.02)
    cbar.set_label("Pareto solution index", fontsize=9)

    legend_handles = [
        Line2D([0], [0], marker=mk, color="w", markerfacecolor=bc,
               markeredgecolor="k", markersize=9,
               label=label.replace("\n", " "))
        for (label, _), mk, bc in zip(best.items(), BEST_MARKERS, BEST_COLORS)
    ] + [
        Line2D([0], [0], marker=CLUE3D_MARKER, color="w",
               markerfacecolor=CLUE3D_COLOR, markeredgecolor="k",
               markersize=10, label="CLUE3D (1, 1) reference")
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=8)
    fig.tight_layout()
    return fig


# ── Figure 2: 3-D Pareto front ────────────────────────────────────────────────

def fig_pareto_3d(df, best, baseline=None):
    """Two 3-D views: left = ee objectives (indices 0,1,2), right = ππ (3,4,5)."""
    f   = df[OBJ_NAMES].values
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor("white")
    fig.suptitle("Pareto Front — 3-D  (ratio to CLUE3D, reference point = (1,1,1))",
                 fontsize=13, fontweight="bold")

    for col, (idx_slice, ptype_label) in enumerate([
        (slice(0, 3), "ee events"),
        (slice(3, 6), "ππ events"),
    ]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        fv = f[:, idx_slice]   # shape (N, 3)

        norm = plt.Normalize(fv[:, 1].min(), fv[:, 1].max())
        sc   = ax.scatter(fv[:, 0], fv[:, 1], fv[:, 2],
                          c=fv[:, 1], cmap="plasma", norm=norm,
                          s=40, alpha=0.8, depthshade=True)

        # Shadow projections
        f_min = fv.min(0)
        for dim in range(3):
            shadow = fv.copy()
            shadow[:, dim] = f_min[dim]
            ax.scatter(shadow[:, 0], shadow[:, 1], shadow[:, 2],
                       c=fv[:, 1], cmap="plasma", norm=norm,
                       s=10, alpha=0.15, depthshade=False)

        # CLUE3D reference at (1, 1, 1)
        ax.scatter(1, 1, 1, marker=CLUE3D_MARKER, s=280, color=CLUE3D_COLOR,
                   edgecolors="k", linewidths=0.9, zorder=11,
                   label="CLUE3D (1,1,1)")

        # Best solutions
        for (label, idx), mk, bc in zip(best.items(), BEST_MARKERS, BEST_COLORS):
            ax.scatter(*fv[idx], marker=mk, s=200, color=bc,
                       edgecolors="k", linewidths=0.8, zorder=10)

        fig.colorbar(sc, ax=ax, shrink=0.5, pad=0.08,
                     label="F2 ratio" if col == 0 else "R2_pi")
        ax.set_xlabel("F1 ratio", labelpad=8, fontsize=9)
        ax.set_ylabel("F2 ratio", labelpad=8, fontsize=9)
        ax.set_zlabel("F3 ratio", labelpad=8, fontsize=9)
        ax.set_title(ptype_label, fontsize=11)
        ax.legend(loc="upper left", fontsize=7, framealpha=0.9)

    fig.tight_layout()
    return fig


# ── Figure 3: Objective distributions with KDE ────────────────────────────────

def fig_objective_distributions(df, baseline=None):
    """2×3 grid: top row = ee objectives, bottom = ππ.  Ratio=1 line = CLUE3D."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    hep.cms.label("Preliminary", data=False, ax=axes[0, 0],
                  com=None, lumi=None, loc=0, fontsize=10)
    fig.suptitle("Objective Distributions  (ratio to CLUE3D, vertical line at 1.0 = CLUE3D)",
                 fontsize=13, fontweight="bold", y=1.01)

    row_labels = ["ee events", "ππ events"]
    palette_row = [["#e6194b", "#3cb44b", "#4363d8"],
                   ["#f032e6", "#f58231", "#42d4f4"]]

    for row in range(2):
        for col in range(3):
            ax     = axes[row, col]
            obj_i  = row * 3 + col
            color  = palette_row[row][col]
            vals   = df[OBJ_NAMES[obj_i]].values

            sns.histplot(vals, bins=25, ax=ax, color=color, alpha=0.6,
                         edgecolor="white", stat="density")
            kde_x = np.linspace(vals.min(), vals.max(), 300)
            kde_y = gaussian_kde(vals)(kde_x)
            ax.plot(kde_x, kde_y, color=color, lw=2.0)

            p16, p50, p84 = np.percentile(vals, [16, 50, 84])
            ax.axvspan(p16, p84, alpha=0.12, color=color)
            ax.axvline(p50,       color=color,   lw=1.8, ls="-",
                       label=f"median {p50:.3f}")
            ax.axvline(vals.min(), color="navy",  lw=1.4, ls="--",
                       label=f"min {vals.min():.3f}")

            # CLUE3D reference: always at 1.0 by construction
            ax.axvline(1.0, color=CLUE3D_COLOR, lw=2.2, ls="-.",
                       label="CLUE3D = 1.0", zorder=5)

            ax.plot(vals, np.full_like(vals, -0.01 * kde_y.max()), "|",
                    color=color, alpha=0.4, markersize=5)

            ax.set_xlabel(OBJ_LABELS[obj_i], fontsize=9)
            ax.set_ylabel("Density" if col == 0 else "")
            ax.legend(fontsize=7.5, framealpha=0.9)
            ax.set_title(f"{row_labels[row]}  ·  {OBJ_LABELS[obj_i]}", fontsize=9)
            ax.text(0.97, 0.97,
                    f"n={len(vals)}\nmin={vals.min():.3f}\nmax={vals.max():.3f}",
                    transform=ax.transAxes, va="top", ha="right", fontsize=7.5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

    fig.tight_layout()
    return fig


# ── Figure 4: Parameter violin plots (seaborn) ────────────────────────────────

def fig_param_violin(df, info):
    default = info.get("default_params", None)
    lower   = np.array(info.get("lower_bounds", [None] * 10), dtype=float)
    upper   = np.array(info.get("upper_bounds", [None] * 10), dtype=float)

    for subdet, params, indices, title, fname in [
        ("CEE", PARAM_NAMES[:5], range(5),  "CEE Parameters", "04a_params_violin_cee.png"),
        ("CHE", PARAM_NAMES[5:], range(5,10), "CHE Parameters", "04b_params_violin_che.png"),
    ]:
        offset = 0 if subdet == "CEE" else 5
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        hep.cms.label("Preliminary", data=False, ax=axes[0],
                      com=None, lumi=None, loc=0, fontsize=10)
        fig.suptitle(f"Parameter Distributions — {subdet}", fontsize=14,
                     fontweight="bold", y=1.02)

        for ax, pname, gi in zip(axes, params, indices):
            vals = df[pname].values
            lb, ub = lower[gi], upper[gi]

            # Violin + strip
            vp = ax.violinplot(vals, positions=[0], showmedians=False,
                               showextrema=False, widths=0.8)
            for body in vp["bodies"]:
                body.set_facecolor("#4a90d9")
                body.set_edgecolor("steelblue")
                body.set_alpha(0.55)

            # Individual points (jittered)
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(jitter, vals, s=12, alpha=0.5, color="steelblue",
                       zorder=3, linewidths=0)

            # Quartile box
            q25, q50, q75 = np.percentile(vals, [25, 50, 75])
            ax.vlines(0, q25, q75, color="navy", lw=6, alpha=0.5, zorder=4)
            ax.scatter(0, q50, s=60, color="navy", zorder=5)

            # Default
            if default is not None:
                ax.axhline(default[gi], color="#e6194b", lw=2.5, ls="--",
                           zorder=6, label="default")
                ax.scatter(0, default[gi], s=90, color="#e6194b", marker="D",
                           zorder=7)

            # Bounds
            ax.axhline(lb, color="grey", lw=1.0, ls=":", alpha=0.7)
            ax.axhline(ub, color="grey", lw=1.0, ls=":", alpha=0.7)
            ax.text(-0.45, lb + 0.01*(ub-lb), "lower\nbound", fontsize=6,
                    color="grey", va="bottom")
            ax.text(-0.45, ub - 0.01*(ub-lb), "upper\nbound", fontsize=6,
                    color="grey", va="top")

            # Stats
            ax.text(0.97, 0.5,
                    f"med={q50:.2f}\nIQR=[{q25:.2f},\n{q75:.2f}]",
                    transform=ax.transAxes, va="center", ha="right",
                    fontsize=7.5, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

            ax.set_title(PARAM_LABELS[pname], fontsize=12)
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(lb - 0.08*(ub-lb), ub + 0.08*(ub-lb))
            ax.set_xticks([])
            ax.set_ylabel("Value")

        if default is not None:
            fig.legend(
                handles=[Line2D([0], [0], color="#e6194b", lw=2.5, ls="--",
                                marker="D", markersize=7,
                                label="Current default params")],
                loc="lower right", fontsize=10
            )
        fig.tight_layout()
        save(fig, fname)


# ── Figure 5: Correlation clustermap ─────────────────────────────────────────

def fig_correlation_clustermap(df):
    cols  = PARAM_NAMES + OBJ_NAMES
    # Drop constant columns — zero-variance makes correlation undefined (NaN)
    valid_cols = [c for c in cols if df[c].std() > 1e-10]
    dropped    = set(cols) - set(valid_cols)
    if dropped:
        print(f"  [clustermap] dropping constant columns: {dropped}")
    corr  = df[valid_cols].corr()

    # Rename for display
    rename = {**PARAM_LABELS_SHORT,
              "F1_purity": "F1: Impurity",
              "F2_efficiency_deficit": "F2: Eff. deficit",
              "F3_fragmentation": "F3: Fragmentation"}
    corr.index   = [rename.get(c, c) for c in corr.index]
    corr.columns = [rename.get(c, c) for c in corr.columns]

    g = sns.clustermap(
        corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        annot=True, fmt=".2f", annot_kws={"size": 7},
        figsize=(14, 12), linewidths=0.4, linecolor="white",
        dendrogram_ratio=0.12, cbar_pos=(0.02, 0.82, 0.03, 0.15),
    )
    g.ax_heatmap.set_xticklabels(
        g.ax_heatmap.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    g.ax_heatmap.set_yticklabels(
        g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8)
    g.figure.suptitle(
        "Hierarchically-Clustered Pearson Correlation\n(params + objectives)",
        y=1.02, fontsize=13, fontweight="bold")

    return g.figure


# ── Figure 6: Parallel coordinates ────────────────────────────────────────────

def fig_parallel_coords(df, best, colour_by=1, baseline=None):
    """colour_by: index into OBJ_NAMES used to colour background lines."""
    all_cols  = PARAM_NAMES + OBJ_NAMES
    all_labels = ([PARAM_LABELS_SHORT[p] for p in PARAM_NAMES]
                  + ["F1: Impurity", "F2: Eff. deficit", "F3: Fragmentation"])

    # Normalise to [0,1] per column using Pareto front range
    col_min = df[all_cols].min()
    col_max = df[all_cols].max()
    data_norm = (df[all_cols] - col_min) / (col_max - col_min + 1e-12)

    colour_vals = data_norm[OBJ_NAMES[colour_by]].values
    cmap        = plt.cm.plasma
    n_ax        = len(all_cols)

    fig, ax = plt.subplots(figsize=(20, 6))
    fig.suptitle(
        f"Parallel Coordinates — all parameters + objectives "
        f"(colour = {OBJ_LABELS[colour_by]})",
        fontsize=13, fontweight="bold")

    # Background: all Pareto solutions
    for i in range(len(df)):
        ax.plot(range(n_ax), data_norm.iloc[i].values,
                color=cmap(colour_vals[i]), alpha=0.25, lw=0.9, zorder=2)

    # Best solutions on top
    ls_cycle = ["-", "--", "-.", ":"]
    for (label, idx), ls, bc in zip(best.items(), ls_cycle, BEST_COLORS):
        ax.plot(range(n_ax), data_norm.iloc[idx].values,
                color=bc, lw=2.8, ls=ls, zorder=5,
                label=label.replace("\n", " "))
        ax.scatter(range(n_ax), data_norm.iloc[idx].values,
                   color=bc, s=30, zorder=6)

    # CLUE3D baseline — only the objective axes are defined; param axes are N/A
    if baseline is not None:
        bv_raw  = np.array([baseline["f1"], baseline["f2"], baseline["f3"]])
        obj_start = len(PARAM_NAMES)
        # Normalise CLUE3D objectives using the same Pareto-front scale
        bv_norm = (bv_raw - col_min[OBJ_NAMES].values) / (
            col_max[OBJ_NAMES].values - col_min[OBJ_NAMES].values + 1e-12
        )
        ax.plot(range(obj_start, n_ax), bv_norm,
                color=CLUE3D_COLOR, lw=3.0, ls="-", zorder=7,
                label="CLUE3D (baseline)")
        ax.scatter(range(obj_start, n_ax), bv_norm,
                   marker=CLUE3D_MARKER, color=CLUE3D_COLOR,
                   s=100, zorder=8, edgecolors="k", linewidths=0.7)
        for k, bval_raw in zip(range(obj_start, n_ax), bv_raw):
            ax.text(k, bv_norm[k - obj_start] + 0.04, f"{bval_raw:.3f}",
                    ha="center", fontsize=7, color=CLUE3D_COLOR, fontweight="bold")

    # Vertical axis lines with tick labels
    for k in range(n_ax):
        raw_col = all_cols[k]
        raw_min = df[raw_col].min()
        raw_max = df[raw_col].max()
        ax.axvline(k, color="grey", lw=0.6, alpha=0.6, zorder=1)
        ax.text(k, -0.06, f"{raw_min:.2f}", ha="center", va="top",
                fontsize=6.5, color="grey")
        ax.text(k,  1.06, f"{raw_max:.2f}", ha="center", va="bottom",
                fontsize=6.5, color="grey")

    # Shaded objective region
    ax.axvspan(len(PARAM_NAMES) - 0.5, n_ax - 0.5,
               alpha=0.08, color="gold", zorder=0, label="Objectives")

    ax.set_xticks(range(n_ax))
    ax.set_xticklabels(all_labels, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Normalised value")
    ax.set_ylim(-0.12, 1.20)
    ax.set_xlim(-0.4, n_ax - 0.6)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(
        df[OBJ_NAMES[colour_by]].min(), df[OBJ_NAMES[colour_by]].max()))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.012, pad=0.01)
    cbar.set_label(OBJ_LABELS[colour_by], fontsize=9)

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    return fig


# ── Figure 7: Best solutions — radar chart ───────────────────────────────────

def fig_radar(df, best):
    """Radar chart of all 10 parameters for the 4 best solutions, normalised."""
    n_params = len(PARAM_NAMES)
    angles   = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles  += angles[:1]                   # close the polygon

    labels_radar = [PARAM_LABELS_SHORT[p] for p in PARAM_NAMES]

    fig, axes = plt.subplots(1, 4, figsize=(18, 5),
                              subplot_kw=dict(polar=True))
    fig.suptitle("Best Solutions — Normalised Parameter Radar",
                 fontsize=14, fontweight="bold")

    lower = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5, 0.1])
    upper = np.array([5.0, 10.0, 10.0, 10.0, 1.0, 8.0, 15.0, 15.0, 15.0, 1.0])

    for ax, (label, idx), color in zip(axes, best.items(), BEST_COLORS):
        row    = df.iloc[idx][PARAM_NAMES].values.astype(float)
        normed = (row - lower) / (upper - lower)
        vals   = normed.tolist() + [normed[0]]

        ax.plot(angles, vals, color=color, lw=2.5)
        ax.fill(angles, vals, color=color, alpha=0.25)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels_radar, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=6, color="grey")
        ax.grid(color="grey", alpha=0.3)

        fvals = df.iloc[idx][OBJ_NAMES].values
        ax.set_title(
            f"{label.replace(chr(10), ' ')}\n"
            f"F1={fvals[0]:.4f}  F2={fvals[1]:.4f}  F3={fvals[2]:.2f}",
            fontsize=9, pad=18, color=color, fontweight="bold")

    fig.tight_layout()
    return fig


# ── Figure 8: Seaborn pair plots — params coloured by each objective ──────────

def fig_pairplot(df):
    for obj, label, fname in zip(
        OBJ_NAMES, OBJ_LABELS,
        ["08a_pairplot_F1.png", "08b_pairplot_F2.png", "08c_pairplot_F3.png"]
    ):
        # Use normalised objective as hue via continuous colormap
        plot_df = df[PARAM_NAMES + [obj]].copy()
        plot_df.columns = (
            [PARAM_LABELS_SHORT[p] for p in PARAM_NAMES] + [label]
        )

        # Bin the objective into 4 quantile groups for pairplot colouring
        plot_df["_q"] = pd.qcut(plot_df[label], q=4,
                                 labels=["Q1 (best)", "Q2", "Q3", "Q4 (worst)"])
        palette = sns.color_palette("RdYlGn_r", 4)

        g = sns.pairplot(
            plot_df.drop(columns=[label]),
            hue="_q",
            palette=palette,
            plot_kws=dict(alpha=0.5, s=18, edgecolor="none"),
            diag_kind="kde",
            diag_kws=dict(fill=True, alpha=0.45),
            corner=True,
        )
        g.figure.suptitle(
            f"Parameter Pair Plot — coloured by {label} quartile",
            y=1.02, fontsize=13, fontweight="bold")
        g._legend.set_title(f"{label} quartile")
        save(g.figure, fname)


# ── Figure 9: CEE vs CHE scatter matrix with marginals ───────────────────────

def fig_cee_vs_che(df, best):
    param_base  = ["density_radius", "min_density", "outlier_distance",
                   "seeding_distance", "w_z"]
    nice_labels = [r"$\rho_r$", r"$\rho_\mathrm{min}$",
                   r"$d_\mathrm{out}$", r"$d_\mathrm{seed}$", r"$w_z$"]

    f3     = df["F3_fragmentation"].values
    f3norm = (f3 - f3.min()) / (f3.max() - f3.min() + 1e-12)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    hep.cms.label("Preliminary", data=False, ax=axes[0],
                  com=None, lumi=None, loc=0, fontsize=10)
    fig.suptitle("CEE vs CHE Parameter Values (colour = F3 fragmentation)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, base, label in zip(axes, param_base, nice_labels):
        x = df[f"{base}_cee"].values
        y = df[f"{base}_che"].values

        # KDE background
        try:
            k   = gaussian_kde(np.vstack([x, y]))
            xg  = np.linspace(x.min(), x.max(), 60)
            yg  = np.linspace(y.min(), y.max(), 60)
            Xg, Yg = np.meshgrid(xg, yg)
            Z   = k(np.vstack([Xg.ravel(), Yg.ravel()])).reshape(Xg.shape)
            ax.contourf(Xg, Yg, Z, levels=6, cmap="Greys", alpha=0.35)
        except Exception:
            pass

        sc = ax.scatter(x, y, c=f3norm, cmap="plasma", s=35, alpha=0.85,
                        zorder=3, linewidths=0.3, edgecolors="white")

        # Best solutions
        for (lbl, idx), mk, bc in zip(best.items(), BEST_MARKERS, BEST_COLORS):
            ax.scatter(x[idx], y[idx], marker=mk, s=200, color=bc,
                       edgecolors="k", linewidths=0.7, zorder=5)

        # y = x diagonal
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5, label="CEE=CHE")

        ax.set_xlabel(f"{label} CEE")
        ax.set_ylabel(f"{label} CHE")
        ax.set_aspect("auto")

    cbar = fig.colorbar(sc, ax=axes, fraction=0.013, pad=0.02)
    cbar.set_label("F3 fragmentation (norm.)")
    fig.tight_layout()
    return fig


# ── Figure 10: Objective trade-off analysis ───────────────────────────────────

def fig_tradeoff(df, best, baseline=None):
    """
    Show how each pair of objectives trades off, with the solution density
    shown as a 2-D histogram in the background and best solutions annotated.
    """
    pairs = [(0, 1), (0, 2), (1, 2)]
    f     = df[OBJ_NAMES].values
    bv    = [baseline["f1"], baseline["f2"], baseline["f3"]] if baseline else None

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("Objective Trade-off Analysis — 2-D Density",
                 fontsize=14, fontweight="bold")

    for ax, (i, j) in zip(axes, pairs):
        xi, xj = f[:, i], f[:, j]

        # 2-D histogram background
        h = ax.hist2d(xi, xj, bins=20, cmap="YlOrRd",
                      density=True, zorder=1)
        fig.colorbar(h[3], ax=ax, label="Density")

        ax.scatter(xi, xj, s=25, color="white", alpha=0.5, zorder=3,
                   linewidths=0.3, edgecolors="grey")

        for (label, idx), mk, bc in zip(best.items(), BEST_MARKERS, BEST_COLORS):
            ax.scatter(xi[idx], xj[idx], marker=mk, s=240, color=bc,
                       edgecolors="k", linewidths=0.9, zorder=6,
                       label=label.replace("\n", " "))

        # CLUE3D baseline
        if bv is not None:
            ax.scatter(bv[i], bv[j], marker=CLUE3D_MARKER, s=300,
                       color=CLUE3D_COLOR, edgecolors="k", linewidths=0.9, zorder=7,
                       label="CLUE3D (baseline)")

        # Spearman correlation
        from scipy.stats import spearmanr
        rho, pval = spearmanr(xi, xj)
        ax.text(0.03, 0.97,
                f"Spearman ρ = {rho:.3f}\np = {pval:.3f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.85))

        ax.set_xlabel(OBJ_LABELS[i] + f"\n{OBJ_UNITS[i]}")
        ax.set_ylabel(OBJ_LABELS[j] + f"\n{OBJ_UNITS[j]}")

    legend_handles = [
        Line2D([0], [0], marker=mk, color="w", markerfacecolor=bc,
               markeredgecolor="k", markersize=10, label=label.replace("\n", " "))
        for (label, _), mk, bc in zip(best.items(), BEST_MARKERS, BEST_COLORS)
    ]
    if bv is not None:
        legend_handles.append(
            Line2D([0], [0], marker=CLUE3D_MARKER, color="w",
                   markerfacecolor=CLUE3D_COLOR, markeredgecolor="k",
                   markersize=11, label="CLUE3D (baseline)")
        )
    fig.legend(handles=legend_handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.10), frameon=True, fontsize=9)
    fig.tight_layout()
    return fig


# ── Figure 11: Summary text panel ────────────────────────────────────────────

def fig_summary(df, info, best, baseline=None):
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.axis("off")
    hep.cms.label("Preliminary", data=False, ax=ax, com=None, lumi=None,
                  loc=0, fontsize=11)
    fig.suptitle("Optimisation Run Summary", fontsize=14, fontweight="bold")

    lines = [
        f"{'Events:':<25} {info.get('n_events','?')}  "
        f"({info.get('n_ee','?')} ee + {info.get('n_pi','?')} ππ)",
        f"{'Particles:':<25} {info.get('num_particles','?')}",
        f"{'Iterations:':<25} {info.get('num_iterations','?')}",
        f"{'Pareto archive:':<25} {info.get('pareto_size', len(df))} solutions",
        f"{'Wall time:':<25} {info.get('wall_time_h','?'):.2f} h",
        "",
        "Objectives are ratios to CLUE3D — value < 1.0 means better than CLUE3D",
        "─" * 72,
        f"{'Objective':<34}  {'min':>8}  {'mean':>8}  {'max':>8}  {'σ':>8}",
        "─" * 72,
    ]
    for col, label in zip(OBJ_NAMES, OBJ_LABELS):
        v = df[col].values
        lines.append(
            f"  {label:<32}  {v.min():8.4f}  {v.mean():8.4f}  "
            f"{v.max():8.4f}  {v.std():8.4f}"
        )
    lines += ["─" * 72, "", "Best solutions  (ratio to CLUE3D):", "─" * 72]
    short_obj = ["R1ee", "R2ee", "R3ee", "R1pi", "R2pi", "R3pi"]
    for (label, idx) in best.items():
        row  = df.iloc[idx]
        vals = "  ".join(f"{o}={row[c]:.3f}"
                         for o, c in zip(short_obj, OBJ_NAMES))
        lines.append(f"  {label.replace(chr(10),' '):<28}  {vals}")

    ax.text(0.03, 0.88, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4ff", alpha=0.9))
    fig.tight_layout()
    return fig


# ── Figure 12: Top-10 balanced solutions table ───────────────────────────────

def top10_solutions(df, n=10):
    """Return the top-n solutions ranked by normalised sum of all 3 objectives."""
    f      = df[OBJ_NAMES].values
    f_norm = (f - f.min(0)) / (f.max(0) - f.min(0) + 1e-12)
    scores = f_norm.sum(axis=1)
    idx    = np.argsort(scores)[:n]
    ranked = df.iloc[idx].copy()
    ranked.insert(0, "rank",        range(1, len(ranked) + 1))
    ranked.insert(1, "score_norm",  scores[idx])
    return ranked


def fig_top10_table(df, baseline=None):
    """
    Table of top-10 balanced solutions.  Objectives are ratios (< 1 = better
    than CLUE3D).  A reference row at ratio=1.0 marks the CLUE3D level.
    """
    ranked = top10_solutions(df, n=10)
    obj_display   = OBJ_NAMES
    param_display = PARAM_NAMES
    all_display   = ["rank"] + obj_display + param_display

    disp = ranked[all_display].copy()
    for c in obj_display:
        disp[c] = disp[c].map(lambda v: f"{v:.3f}")
    for c in param_display:
        disp[c] = disp[c].map(lambda v: f"{v:.3f}")

    cell_text = disp.values.tolist()
    # CLUE3D reference row: all ratios = 1.0 by definition
    cell_text.append(
        ["CLUE3D"] + ["1.000"] * len(obj_display) + ["—"] * len(param_display)
    )

    col_labels = (["#"] +
                  ["R1\nee", "R2\nee", "R3\nee",
                   "R1\nππ", "R2\nππ", "R3\nππ"] +
                  [PARAM_LABELS_SHORT[p].replace(" ", "\n") for p in param_display])

    n_data_rows = len(cell_text)
    fig, ax = plt.subplots(figsize=(26, 0.45 * (n_data_rows + 2) + 1.5))
    ax.axis("off")
    fig.suptitle(
        "Top-10 Balanced Solutions  (ranked by min normalised sum of all 6 ratio objectives)"
        "  ·  Ratios < 1.0 beat CLUE3D",
        fontsize=12, fontweight="bold", y=1.01)

    tbl = ax.table(cellText=cell_text, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1.0, 2.0)

    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == n_data_rows:      # CLUE3D reference row
            cell.set_facecolor("#e8d5f5")
            cell.set_text_props(color=CLUE3D_COLOR, fontweight="bold")
        elif row == 1:
            cell.set_facecolor("#fdf6e3")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
        # Colour-code ratio values: green < 1, red > 1
        if row > 0 and row < n_data_rows and 1 <= col <= len(obj_display):
            try:
                v = float(cell.get_text().get_text())
                cell.set_facecolor("#d4efdf" if v < 1.0 else "#fadbd8")
            except ValueError:
                pass

    fig.tight_layout()
    return fig


def print_top10(df, baseline=None):
    """Print a readable top-10 table to stdout."""
    ranked = top10_solutions(df, n=10)
    W = 130
    print("\n" + "═" * W)
    print("  Top-10 balanced Pareto solutions  "
          "(ranked by min normalised sum — ratios < 1.0 beat CLUE3D)")
    print("═" * W)
    obj_hdr = "  ".join(f"{n:>8}" for n in ["R1_ee","R2_ee","R3_ee","R1_pi","R2_pi","R3_pi"])
    print(f"  {'#':>3}  {obj_hdr}  "
          f"{'ρr_CEE':>7}  {'ρmin_CEE':>9}  {'dout_CEE':>9}  {'dseed_CEE':>10}  {'wz_CEE':>7}"
          f"  {'ρr_CHE':>7}  {'ρmin_CHE':>9}  {'dout_CHE':>9}  {'dseed_CHE':>10}  {'wz_CHE':>7}")
    print("─" * W)
    for _, row in ranked.iterrows():
        obj_vals = "  ".join(f"{row[c]:>8.3f}" for c in OBJ_NAMES)
        print(
            f"  {int(row['rank']):>3}  {obj_vals}  "
            f"{row['density_radius_cee']:>7.3f}  "
            f"{row['min_density_cee']:>9.3f}  "
            f"{row['outlier_distance_cee']:>9.3f}  "
            f"{row['seeding_distance_cee']:>10.3f}  "
            f"{row['w_z_cee']:>7.3f}  "
            f"{row['density_radius_che']:>7.3f}  "
            f"{row['min_density_che']:>9.3f}  "
            f"{row['outlier_distance_che']:>9.3f}  "
            f"{row['seeding_distance_che']:>10.3f}  "
            f"{row['w_z_che']:>7.3f}"
        )
    print("─" * W)
    print(f"  {'C3D':>3}  " + "  ".join(f"{'1.000':>8}" for _ in OBJ_NAMES)
          + "  (CLUE3D reference — all ratios = 1.0 by definition)")
    print("═" * W + "\n")


# ── Figure 13: CLUE3D per-particle-type breakdown ────────────────────────────

def fig_clue3d_pertype(baseline):
    """
    Bar chart comparing CLUE3D F1/F2/F3 broken down by ee vs ππ events,
    alongside the combined value.  Makes clear how much the pion events drive
    the fragmentation score.
    """
    if baseline is None or "per_type" not in baseline:
        return None

    pt  = baseline["per_type"]
    labels  = ["ee (2e)", "ππ", "Combined"]
    f1_vals = [pt["ee"]["f1"],    pt["pi"]["f1"],    baseline["f1"]]
    f2_vals = [pt["ee"]["f2"],    pt["pi"]["f2"],    baseline["f2"]]
    f3_vals = [pt["ee"]["f3"],    pt["pi"]["f3"],    baseline["f3"]]
    ntr_vals= [pt["ee"]["mean_n_tracksters"],
               pt["pi"]["mean_n_tracksters"],
               0.5*(pt["ee"]["mean_n_tracksters"] + pt["pi"]["mean_n_tracksters"])]

    x     = np.arange(len(labels))
    width = 0.26
    colors = ["#e6194b", "#3cb44b", "#4363d8"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hep.cms.label("Preliminary", data=False, ax=axes[0],
                  com=None, lumi=None, loc=0, fontsize=10)
    fig.suptitle("CLUE3D Baseline — Metrics by Particle Type",
                 fontsize=14, fontweight="bold", y=1.02)

    for ax, vals, label, unit, color in zip(
        axes,
        [f1_vals, f2_vals, f3_vals],
        OBJ_LABELS, OBJ_UNITS,
        ["#e6194b", "#3cb44b", "#4363d8"]
    ):
        bars = ax.bar(x, vals, width=0.55, color=[CLUE3D_COLOR, "#c39bd3", "#9b59b6"],
                      edgecolor="k", linewidth=0.7, zorder=3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v * 1.02,
                    f"{v:.4f}" if label != OBJ_LABELS[2] else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        # Annotate with mean NTracksters
        for i, (xi, ntr) in enumerate(zip(x, ntr_vals)):
            ax.text(xi, -0.08 * max(vals), f"<Ntr>={ntr:.1f}",
                    ha="center", va="top", fontsize=7.5, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(f"{label}\n{unit}")
        ax.set_ylim(0, max(vals) * 1.25)
        ax.set_title(label, fontsize=11)

    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output → {OUTPUT_DIR}\n")

    df, info = load_results()
    print(f"Loaded {len(df)} Pareto solutions\n")

    best = best_solutions(df)
    print("Best solutions:")
    for label, idx in best.items():
        row = df.iloc[idx]
        print(f"  [{idx:3d}] {label.replace(chr(10),' '):<35s} "
              f"F1={row['F1_purity']:.4f}  "
              f"F2={row['F2_efficiency_deficit']:.4f}  "
              f"F3={row['F3_fragmentation']:.2f}")

    print_top10(df)

    print("Generating plots...")
    save(fig_summary(df, info, best),                        "00_summary.png")
    save(fig_pareto_2d(df, best),                            "01_pareto_2d_projections.png")
    save(fig_pareto_3d(df, best),                            "02_pareto_3d.png")
    save(fig_objective_distributions(df),                    "03_objective_distributions.png")
    fig_param_violin(df, info)                               # saves 04a + 04b internally
    save(fig_correlation_clustermap(df),                     "05_correlation_clustermap.png")
    save(fig_parallel_coords(df, best, colour_by=1),         "06_parallel_coordinates.png")
    save(fig_radar(df, best),                                "07_radar_best_solutions.png")
    fig_pairplot(df)                                         # saves 08a/b/c internally
    save(fig_cee_vs_che(df, best),                           "09_cee_vs_che.png")
    save(fig_tradeoff(df, best),                             "10_tradeoff_density.png")
    save(fig_top10_table(df),                                "12_top10_balanced.png")

    # Save best solutions as CSV
    rows = df.iloc[list(best.values())].copy()
    rows.insert(0, "selection", [l.replace("\n", " ") for l in best])
    rows.to_csv(OUTPUT_DIR / "best_solutions.csv", index=False)
    print(f"  saved  best_solutions.csv")

    # Save full top-10 as CSV
    top10_solutions(df, n=10).to_csv(OUTPUT_DIR / "top10_balanced.csv", index=False)
    print(f"  saved  top10_balanced.csv")

    total = len(list(OUTPUT_DIR.glob("*.png")))
    print(f"\nDone — {total} plots written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
