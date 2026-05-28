"""
Entry point for CLUEstering parameter optimisation.

Runs a single MOPSO over the joint 10-parameter space [CEE×5, CHE×5].
CLUEstering is run separately on each subdetector per event, but quality
is scored globally across both.

Usage
-----
    python optimize.py

Output  (written to OUTPUT_DIR, default 'results/')
------
    pareto_front.csv
    pareto_front.json
    pareto_positions.npy      (N, 15) parameter vectors
    pareto_fitnesses.npy      (N, 3)  [F1, F2, F3] values
    run_info.json             wall time, event counts, settings used
"""

# ── Immediate startup confirmation (printed before slow package imports) ────────
# EOS-hosted packages take ~80s to load from the network on first access.
# This line confirms Python is alive and the redirect is working.
import sys as _sys, time as _time
print(f"=== optimize.py: Python started at {_time.strftime('%H:%M:%S')} — loading packages... ===",
      flush=True)

import csv
import json
import os
import time

import numpy as np
import patatune

from config import (
    PARAM_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, DEFAULT_PARAMS,
    N_EVENTS, N_JOBS,
    NUM_PARTICLES, NUM_ITERATIONS, INERTIA, COGNITIVE, SOCIAL,
    MAX_PARETO, TOPOLOGY, RANDOM_SEED,
    OUTPUT_DIR,
)
from data import load_events, validate_events
from objective import make_objective

_OBJ_NAMES = ['F1_purity', 'F2_efficiency_deficit', 'F3_fragmentation']


def _subsample_events(events, n):
    """
    Return n events keeping the ee/epion balance exactly 50/50.
    Events are assumed to be interleaved [ee_0, epion_0, ee_1, ...].
    """
    if n is None or n >= len(events):
        return events
    n = n - (n % 2)   # round down to nearest even number
    ee    = [e for e in events if e['particle_type'] == 'ee'   ][:n // 2]
    epion = [e for e in events if e['particle_type'] == 'epion'][:n // 2]
    return [e for pair in zip(ee, epion) for e in pair]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── load and validate data ─────────────────────────────────────────────────
    print("=== Loading data ===", flush=True)
    all_events = load_events()
    validate_events(all_events)
    print(flush=True)

    events = _subsample_events(all_events, N_EVENTS)
    n_ee    = sum(1 for e in events if e['particle_type'] == 'ee')
    n_epion = sum(1 for e in events if e['particle_type'] == 'epion')
    print(f"Using {len(events)} events for optimisation  "
          f"({n_ee} ee + {n_epion} epion)\n", flush=True)

    # ── timing probe (single-process, 20 events) ──────────────────────────────
    print("=== Timing probe ===", flush=True)
    probe_events = _subsample_events(all_events, 20)
    probe_obj    = make_objective(probe_events, n_jobs=1)   # always single for probe
    t0           = time.time()
    probe_result = probe_obj(DEFAULT_PARAMS)
    t_call       = time.time() - t0

    t_per_particle = t_call / len(probe_events) * len(events) / N_JOBS
    t_iter         = t_per_particle * NUM_PARTICLES
    t_total        = t_iter * NUM_ITERATIONS

    print(f"  probe call (20 events, 1 worker) : {t_call:.2f}s")
    print(f"  scaled to {len(events)} events / {N_JOBS} workers : "
          f"{t_per_particle:.1f}s per particle")
    print(f"  estimated per iteration : {t_iter / 60:.1f} min  ({NUM_PARTICLES} particles)")
    print(f"  estimated total         : {t_total / 3600:.1f} h  ({NUM_ITERATIONS} iters)")
    print(f"  probe result            : F1={probe_result[0]:.4f}  "
          f"F2={probe_result[1]:.4f}  F3={probe_result[2]:.4f}\n", flush=True)

    # ── build objective (parallel) ─────────────────────────────────────────────
    print(f"Initialising {N_JOBS} worker processes...", flush=True)
    obj_fn = make_objective(events, n_jobs=N_JOBS)
    print(f"Workers ready.\n", flush=True)

    objective = patatune.ElementWiseObjective(
        obj_fn,
        num_objectives=3,
        directions=['minimize', 'minimize', 'minimize'],
        objective_names=_OBJ_NAMES,
    )

    # ── run MOPSO ─────────────────────────────────────────────────────────────
    print(f"=== Running MOPSO ===")
    print(f"  10 parameters: {PARAM_NAMES[:5]}  (CEE)")
    print(f"                 {PARAM_NAMES[5:]}  (CHE)")
    print(f"  particles={NUM_PARTICLES}  iterations={NUM_ITERATIONS}\n", flush=True)

    patatune.Randomizer.rng = np.random.default_rng(RANDOM_SEED)
    patatune.Logger.setLevel('INFO')

    # Disable patatune's automatic pickle checkpoint.
    # After each iteration patatune calls FileManager.save_pickle(mopso, ...)
    # which tries to serialise the entire MOPSO object — including our objective
    # closure — with dill.  multiprocessing.Pool cannot be pickled, so the job
    # crashes.  Setting saving_pickle_enabled=False makes save_state() a no-op.
    patatune.FileManager.saving_pickle_enabled = False

    t_start = time.time()

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

    pareto    = mopso.optimize(num_iterations=NUM_ITERATIONS)
    wall_time = time.time() - t_start

    # Clean up worker processes
    if obj_fn._pool is not None:
        obj_fn._pool.close()
        obj_fn._pool.join()

    print(f"\nDone in {wall_time / 3600:.2f}h — {len(pareto)} Pareto-optimal solutions\n")

    # ── print summary ─────────────────────────────────────────────────────────
    print("=== Pareto front ===")
    header = PARAM_NAMES + _OBJ_NAMES
    print("  ".join(f"{h:>24s}" for h in header))
    for sol in sorted(pareto, key=lambda s: s.fitness[0]):
        vals = list(sol.position) + list(sol.fitness)
        print("  ".join(f"{v:24.6f}" for v in vals))

    # ── save ──────────────────────────────────────────────────────────────────
    print(f"\n=== Saving to {OUTPUT_DIR}/ ===")
    _save_csv(pareto)
    _save_json(pareto)
    _save_npy(pareto)
    _save_run_info(pareto, wall_time, len(events), n_ee, n_epion)
    print("All done.")


# ── Output helpers ─────────────────────────────────────────────────────────────

def _path(filename):
    return os.path.join(OUTPUT_DIR, filename)


def _save_csv(pareto):
    header = PARAM_NAMES + _OBJ_NAMES
    rows   = [[*sol.position, *sol.fitness] for sol in pareto]
    with open(_path('pareto_front.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  pareto_front.csv  ({len(rows)} solutions)")


def _save_json(pareto):
    archive = [
        {
            'params':     dict(zip(PARAM_NAMES, sol.position.tolist())),
            'objectives': dict(zip(_OBJ_NAMES,  sol.fitness.tolist())),
        }
        for sol in pareto
    ]
    with open(_path('pareto_front.json'), 'w') as f:
        json.dump(archive, f, indent=2)
    print("  pareto_front.json")


def _save_npy(pareto):
    np.save(_path('pareto_positions.npy'), np.array([sol.position for sol in pareto]))
    np.save(_path('pareto_fitnesses.npy'),  np.array([sol.fitness  for sol in pareto]))
    print("  pareto_positions.npy / pareto_fitnesses.npy")


def _save_run_info(pareto, wall_time, n_events, n_ee, n_epion):
    info = {
        'n_events':       n_events,
        'n_ee':           n_ee,
        'n_epion':        n_epion,
        'num_particles':  NUM_PARTICLES,
        'num_iterations': NUM_ITERATIONS,
        'pareto_size':    len(pareto),
        'wall_time_h':    round(wall_time / 3600, 3),
        'param_names':    PARAM_NAMES,
        'lower_bounds':   LOWER_BOUNDS,
        'upper_bounds':   UPPER_BOUNDS,
        'default_params': DEFAULT_PARAMS,
        'random_seed':    RANDOM_SEED,
    }
    with open(_path('run_info.json'), 'w') as f:
        json.dump(info, f, indent=2)
    print("  run_info.json")


if __name__ == '__main__':
    main()
