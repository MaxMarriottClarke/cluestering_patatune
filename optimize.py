"""
Entry point for CLUEstering parameter optimisation.

Runs a single MOPSO over the joint 10-parameter space [CEE×5, CHE×5].
CLUEstering is run separately on CEE and CHE LCs per event, but quality
is scored globally across both subdetectors.

Usage
-----
    python optimize.py

Output
------
    pareto_front.csv          — Pareto-optimal solutions (params + objectives)
    pareto_front.json         — same, structured with named keys
    pareto_positions.npy      — (N, 10) array of parameter vectors
    pareto_fitnesses.npy      — (N, 3)  array of [F1, F2, F3] values
"""

import csv
import json

import numpy as np
import patatune

from config import (
    PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, DEFAULT_PARAMS,
    NUM_PARTICLES, NUM_ITERATIONS, INERTIA, COGNITIVE, SOCIAL,
    MAX_PARETO, TOPOLOGY, RANDOM_SEED,
)
from data import load_events, validate_events
from objective import make_objective

_OBJ_NAMES = ['F1_purity', 'F2_efficiency_deficit', 'F3_fragmentation']


def main():
    # ── load and validate data ──────────────────────────────────────────────────
    print("=== Loading data ===")
    events = load_events()
    validate_events(events)
    print()

    # ── build objective ─────────────────────────────────────────────────────────
    obj_fn = make_objective(events)

    objective = patatune.ElementWiseObjective(
        obj_fn,
        num_objectives=3,
        directions=['minimize', 'minimize', 'minimize'],
        objective_names=_OBJ_NAMES,
    )

    # ── run MOPSO ──────────────────────────────────────────────────────────────
    print("=== Running MOPSO (joint CEE+CHE, 10 parameters) ===")
    print(f"  particles={NUM_PARTICLES}  iterations={NUM_ITERATIONS}")
    print(f"  params: {PARAM_NAMES[:5]}  (CEE)")
    print(f"          {PARAM_NAMES[5:]}  (CHE)\n")

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
    print(f"\nDone — {len(pareto)} Pareto-optimal solutions\n")

    # ── print summary ───────────────────────────────────────────────────────────
    header = PARAM_NAMES + _OBJ_NAMES
    print("=== Pareto front ===")
    print("  ".join(f"{h:>22s}" for h in header))
    for sol in pareto:
        vals = list(sol.position) + list(sol.objectives)
        print("  ".join(f"{v:22.6f}" for v in vals))

    # ── save results ────────────────────────────────────────────────────────────
    print("\n=== Saving results ===")
    _save_csv(pareto)
    _save_json(pareto)
    _save_npy(pareto)
    print("All done.")


# ── Output helpers ─────────────────────────────────────────────────────────────

def _save_csv(pareto):
    header = PARAM_NAMES + _OBJ_NAMES
    rows   = [[*sol.position, *sol.objectives] for sol in pareto]
    with open('pareto_front.csv', 'w', newline='') as f:
        csv.writer(f).writerow(header)
        csv.writer(f).writerows(rows)
    print(f"  pareto_front.csv  ({len(rows)} solutions)")


def _save_json(pareto):
    archive = [
        {
            'params':     dict(zip(PARAM_NAMES,  sol.position.tolist())),
            'objectives': dict(zip(_OBJ_NAMES,   sol.objectives.tolist())),
        }
        for sol in pareto
    ]
    with open('pareto_front.json', 'w') as f:
        json.dump(archive, f, indent=2)
    print("  pareto_front.json")


def _save_npy(pareto):
    np.save('pareto_positions.npy', np.array([sol.position   for sol in pareto]))
    np.save('pareto_fitnesses.npy',  np.array([sol.objectives for sol in pareto]))
    print("  pareto_positions.npy / pareto_fitnesses.npy")


if __name__ == '__main__':
    main()
