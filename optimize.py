"""
Entry point for CLUEstering parameter optimisation.

Usage
-----
    python optimize.py

All tunable knobs (data paths, CLUEstering defaults, MOPSO settings, objective
weights) live in config.py.  The objective function lives in objective.py.

Output
------
    pareto_front.csv   — Pareto-optimal parameter sets and objective values
"""

import csv
import time

import numpy as np
import patatune

from config import (
    PATH, CONFIG_FILES, PARTICLE_TYPES,
    DEFAULT_PARAMS, PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS,
    NUM_PARTICLES, NUM_ITERATIONS, INERTIA, COGNITIVE, SOCIAL,
    MAX_PARETO, TOPOLOGY, RANDOM_SEED,
)
from data import load_all, build_df_lc, build_lc_energy_lookup
from metrics import compute_baseline
from objective import make_objective_fn


def main():
    # ── load data ──────────────────────────────────────────────────────────────
    print("=== Loading data ===")
    raw              = load_all(PATH, CONFIG_FILES)
    df_lc            = build_df_lc(raw)
    lc_energy_lookup = build_lc_energy_lookup(df_lc)
    print(f"LC rows: {len(df_lc):,}\n")

    # ── CLUE3D baseline ────────────────────────────────────────────────────────
    print("=== Computing CLUE3D baseline ===")
    baseline = compute_baseline(raw, lc_energy_lookup)
    for ptype, m in baseline.items():
        print(f"  {ptype:>4s}  L1={m['L1']:.4f}  r={m['r']:.4f}  NT={m['NT']:.2f}")
    print()

    # ── objective ──────────────────────────────────────────────────────────────
    objective_fn = make_objective_fn(raw, df_lc, lc_energy_lookup, baseline)

    objective = patatune.ElementWiseObjective(
        objective_fn,
        num_objectives=3,
        directions=['minimize', 'maximize', 'minimize'],
        objective_names=['purity_delta', 'energy_ratio', 'count_delta'],
    )

    # ── timing probe ───────────────────────────────────────────────────────────
    print("=== Timing probe (default params) ===")
    t0       = time.time()
    sample   = objective_fn(DEFAULT_PARAMS)
    per_call = time.time() - t0
    print(f"  one call : {per_call:.2f}s")
    print(f"  per iter : {per_call * NUM_PARTICLES / 60:.1f} min  ({NUM_PARTICLES} particles)")
    print(f"  total    : {per_call * NUM_PARTICLES * NUM_ITERATIONS / 3600:.1f} h  ({NUM_ITERATIONS} iters)")
    print(f"  result   : purity_delta={sample[0]:.4f}  energy_ratio={sample[1]:.4f}  count_delta={sample[2]:.4f}")
    print()

    # ── MOPSO ──────────────────────────────────────────────────────────────────
    print("=== Running MOPSO ===")
    patatune.Randomizer.rng = np.random.default_rng(RANDOM_SEED)
    patatune.Logger.setLevel('WARNING')

    mopso = patatune.MOPSO(
        objective=objective,
        lower_bounds=LOWER_BOUNDS,
        upper_bounds=UPPER_BOUNDS,
        num_particles=NUM_PARTICLES,
        inertia_weight=INERTIA,
        cognitive_coefficient=COGNITIVE,
        social_coefficient=SOCIAL,
        initial_particles_position='gaussian',
        default_point=DEFAULT_PARAMS,
        max_pareto_length=MAX_PARETO,
        topology=TOPOLOGY,
    )

    pareto = mopso.optimize(num_iterations=NUM_ITERATIONS)
    print(f"\nDone. Pareto front: {len(pareto)} solutions\n")

    # ── results ────────────────────────────────────────────────────────────────
    obj_names = ['purity_delta', 'energy_ratio', 'count_delta']
    header    = PARAM_NAMES + obj_names

    print("=== Pareto front ===")
    print("  ".join(f"{h:>14s}" for h in header))
    rows = _extract_pareto_rows(pareto)
    for row in rows:
        print("  ".join(f"{v:14.6f}" for v in row))

    with open('pareto_front.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} solutions → pareto_front.csv")


def _extract_pareto_rows(pareto):
    """Extract (params + objectives) from each Pareto solution."""
    rows = []
    for sol in pareto:
        # patatune solution objects expose .position and .objectives
        pos  = list(sol.position)
        objs = list(sol.objectives)
        rows.append(pos + objs)
    return rows


if __name__ == '__main__':
    main()
