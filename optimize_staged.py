"""
Staged two-phase CLUEstering parameter optimisation.

  Phase 1  CEE  —  5 CEE params, ee events only, 3 objectives [R1_ee, R2_ee, R3_ee]
                   CHE params fixed at DEFAULT_PARAMS[5:]
  Phase 2  CHE  —  5 CHE params, pi events only, 3 objectives [R1_pi, R2_pi, R3_pi]
                   CEE params fixed at balanced-best solution from Phase 1

Rationale
---------
Electrons shower almost entirely in CEE, so Phase 1 finds CEE params with a clean
signal uncorrupted by hadronic performance.  Pion showers span both subdets, but the
hadronic objectives are dominated by CHE; fixing the CEE params found in Phase 1 lets
Phase 2 freely explore CHE without the search space doubling.

Outputs (all in OUTPUT_DIR = results/)
-------
  staged_cee_pareto.csv    Phase 1 Pareto front: 5 CEE params + [R1_ee, R2_ee, R3_ee]
  staged_cee_run_info.json
  staged_che_pareto.csv    Phase 2 Pareto front: 5 CHE params + [R1_pi, R2_pi, R3_pi]
  staged_che_run_info.json
  pareto_front.csv         Combined 10-param + 6-objective CSV for validate_results.py
                           (CEE row is the Phase 1 balanced best, fixed for all rows;
                            ee objectives are constant, pi objectives vary)
  run_info.json            validate_results.py-compatible run info
"""

import sys as _sys, time as _time
print(f"=== optimize_staged.py: Python started at {_time.strftime('%H:%M:%S')} "
      f"— loading packages... ===", flush=True)

import csv
import json
import os
import time

import numpy as np
import patatune

from config import (
    PARAM_NAMES, OBJ_NAMES, LOWER_BOUNDS, UPPER_BOUNDS, DEFAULT_PARAMS,
    N_EVENTS, N_JOBS,
    NUM_PARTICLES, NUM_ITERATIONS, INERTIA, COGNITIVE, SOCIAL,
    MAX_PARETO, TOPOLOGY, RANDOM_SEED,
    OUTPUT_DIR, CLUE3D_BASELINES,
)
from data import load_events, validate_events
from objective import make_staged_objective

# ── Staged parameter/objective slices ─────────────────────────────────────────
CEE_PARAM_NAMES = PARAM_NAMES[:5]
CHE_PARAM_NAMES = PARAM_NAMES[5:]
CEE_OBJ_NAMES   = ['R1_ee', 'R2_ee', 'R3_ee']
CHE_OBJ_NAMES   = ['R1_pi', 'R2_pi', 'R3_pi']
CEE_LOWER       = LOWER_BOUNDS[:5]
CEE_UPPER       = UPPER_BOUNDS[:5]
CHE_LOWER       = LOWER_BOUNDS[5:]
CHE_UPPER       = UPPER_BOUNDS[5:]
CEE_DEFAULTS    = DEFAULT_PARAMS[:5]
CHE_DEFAULTS    = DEFAULT_PARAMS[5:]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _subsample_events(events, n):
    if n is None or n >= len(events):
        return events
    n = n - (n % 2)
    ee = [e for e in events if e['particle_type'] == 'ee'][:n // 2]
    pi = [e for e in events if e['particle_type'] == 'pi'][:n // 2]
    return [e for pair in zip(ee, pi) for e in pair]


def _balanced_best(pareto):
    """
    Return the index of the preferred Pareto solution.

    Preference order:
      1. Among solutions where ALL objectives < 1.0 (every metric beats CLUE3D),
         pick the one with the minimum normalised sum — iterate through candidates
         in ascending normalised-sum order and return the first that satisfies the
         all-below-1 constraint.
      2. If no such solution exists, fall back to global minimum normalised sum
         (same as before) and warn.
    """
    fitnesses = np.array([sol.fitness for sol in pareto])
    f_min  = fitnesses.min(0)
    f_max  = fitnesses.max(0)
    f_norm = (fitnesses - f_min) / (f_max - f_min + 1e-12)
    ranked = np.argsort(f_norm.sum(1))   # best normalised sum first

    for idx in ranked:
        if np.all(fitnesses[idx] < 1.0):
            return int(idx)

    # Fallback: no solution beats CLUE3D on all metrics
    fallback = int(ranked[0])
    print(f"  WARNING: no Phase 1 solution has all objectives < 1.0. "
          f"Falling back to min normalised sum (index {fallback}, "
          f"objectives: {fitnesses[fallback]}).")
    return fallback


def _path(filename):
    return os.path.join(OUTPUT_DIR, filename)


def _save_phase_csv(pareto, param_names, obj_names, path):
    rows = [[*sol.position, *sol.fitness] for sol in pareto]
    with open(path, 'w', newline='') as f:
        csv.writer(f).writerow(param_names + obj_names)
        csv.writer(f).writerows(rows)
    print(f"  {os.path.basename(path)}  ({len(rows)} solutions)")


def _save_combined_csv(cee_params, cee_fitness, che_pareto, path):
    """
    Build a combined 10-param + 6-objective CSV for validate_results.py.

    All rows share the same CEE params and ee objectives (Phase 1 balanced best).
    The CHE params and pi objectives come from the Phase 2 Pareto front.
    """
    header = PARAM_NAMES + OBJ_NAMES
    rows = []
    for sol in che_pareto:
        params = list(cee_params) + list(sol.position)
        objs   = list(cee_fitness) + list(sol.fitness)
        rows.append(params + objs)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {os.path.basename(path)}  ({len(rows)} solutions, combined for validate_results.py)")


def _run_mopso(label, obj_fn, param_names, obj_names, lower, upper, defaults):
    print(f"=== Phase: {label} ===")
    print(f"  {len(param_names)} parameters : {param_names}")
    print(f"  {len(obj_names)} objectives  : {obj_names}")
    print(f"  particles={NUM_PARTICLES}  iterations={NUM_ITERATIONS}\n", flush=True)

    objective = patatune.ElementWiseObjective(
        obj_fn,
        num_objectives=len(obj_names),
        directions=['minimize'] * len(obj_names),
        objective_names=obj_names,
    )
    patatune.FileManager.saving_pickle_enabled = False

    mopso = patatune.MOPSO(
        objective=objective,
        lower_bounds=lower,
        upper_bounds=upper,
        num_particles=NUM_PARTICLES,
        inertia_weight=INERTIA,
        cognitive_coefficient=COGNITIVE,
        social_coefficient=SOCIAL,
        initial_particles_position='gaussian',
        default_point=defaults,
        max_pareto_length=MAX_PARETO,
        topology=TOPOLOGY,
    )

    t0     = time.time()
    pareto = mopso.optimize(num_iterations=NUM_ITERATIONS)
    wall   = time.time() - t0
    print(f"\n  {label} done in {wall / 3600:.2f}h — {len(pareto)} Pareto solutions\n")
    return pareto, wall


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    patatune.Randomizer.rng = np.random.default_rng(RANDOM_SEED)
    patatune.Logger.setLevel('INFO')

    # ── Load and validate ──────────────────────────────────────────────────────
    print("=== Loading data ===", flush=True)
    all_events = load_events()
    validate_events(all_events)

    events = _subsample_events(all_events, N_EVENTS)
    n_ee   = sum(1 for e in events if e['particle_type'] == 'ee')
    n_pi   = sum(1 for e in events if e['particle_type'] == 'pi')
    print(f"Using {len(events)} events  ({n_ee} ee + {n_pi} pi)\n", flush=True)

    # ── Timing probe (CEE phase, 20 events, single worker) ────────────────────
    print("=== Timing probe ===", flush=True)
    probe_events = _subsample_events(all_events, 20)
    probe_obj    = make_staged_objective(probe_events, n_jobs=1, subdet='CEE',
                                         fixed_other_params=CHE_DEFAULTS)
    t0           = time.time()
    probe_result = probe_obj(CEE_DEFAULTS)
    t_call       = time.time() - t0
    n_probe_ee   = sum(1 for e in probe_events if e['particle_type'] == 'ee')
    t_scaled     = t_call / n_probe_ee * n_ee / N_JOBS
    print(f"  probe ({n_probe_ee} ee events, 1 worker)  : {t_call:.2f}s")
    print(f"  scaled to {n_ee} events / {N_JOBS} workers: {t_scaled:.1f}s per particle")
    print(f"  est. per iteration : {t_scaled * NUM_PARTICLES / 60:.1f} min")
    print(f"  est. total (× 2 phases) : {2 * t_scaled * NUM_PARTICLES * NUM_ITERATIONS / 3600:.1f} h")
    print(f"  probe result : " + "  ".join(
          f"{n}={v:.4f}" for n, v in zip(CEE_OBJ_NAMES, probe_result)), flush=True)
    print()

    # ── Phase 1: CEE params, ee events ────────────────────────────────────────
    obj_cee = make_staged_objective(events, n_jobs=N_JOBS, subdet='CEE',
                                    fixed_other_params=CHE_DEFAULTS)
    cee_pareto, cee_wall = _run_mopso(
        'Phase 1 — CEE (ee events only)',
        obj_cee, CEE_PARAM_NAMES, CEE_OBJ_NAMES,
        CEE_LOWER, CEE_UPPER, CEE_DEFAULTS,
    )
    if obj_cee._pool is not None:
        obj_cee._pool.close()
        obj_cee._pool.join()

    # Pick best balanced CEE solution to fix for Phase 2
    best_idx         = _balanced_best(cee_pareto)
    best_cee_params  = cee_pareto[best_idx].position
    best_cee_fitness = cee_pareto[best_idx].fitness

    print("Best balanced CEE solution (min normalised sum, used as fixed params in Phase 2):")
    for n, v in zip(CEE_PARAM_NAMES, best_cee_params):
        print(f"  {n} = {v:.4f}")
    print("  objectives: " + "  ".join(
          f"{n}={v:.4f}" for n, v in zip(CEE_OBJ_NAMES, best_cee_fitness)))
    print(flush=True)

    # ── Phase 2: CHE params, pi events, CEE fixed ─────────────────────────────
    obj_che = make_staged_objective(events, n_jobs=N_JOBS, subdet='CHE',
                                    fixed_other_params=best_cee_params)
    che_pareto, che_wall = _run_mopso(
        'Phase 2 — CHE (pi events only, CEE fixed)',
        obj_che, CHE_PARAM_NAMES, CHE_OBJ_NAMES,
        CHE_LOWER, CHE_UPPER, CHE_DEFAULTS,
    )
    if obj_che._pool is not None:
        obj_che._pool.close()
        obj_che._pool.join()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n=== Saving to {OUTPUT_DIR}/ ===")

    _save_phase_csv(cee_pareto, CEE_PARAM_NAMES, CEE_OBJ_NAMES,
                    _path('staged_cee_pareto.csv'))

    with open(_path('staged_cee_run_info.json'), 'w') as f:
        json.dump({
            'phase': 'CEE', 'particle_type': 'ee',
            'n_events': n_ee, 'pareto_size': len(cee_pareto),
            'wall_time_h': round(cee_wall / 3600, 3),
            'param_names': CEE_PARAM_NAMES, 'obj_names': CEE_OBJ_NAMES,
            'lower_bounds': CEE_LOWER, 'upper_bounds': CEE_UPPER,
            'default_params': CEE_DEFAULTS,
            'best_balanced_idx': best_idx,
            'best_balanced_params': best_cee_params.tolist(),
            'best_balanced_fitness': best_cee_fitness.tolist(),
            'clue3d_baselines': {'ee': CLUE3D_BASELINES['ee']},
        }, f, indent=2)
    print("  staged_cee_run_info.json")

    _save_phase_csv(che_pareto, CHE_PARAM_NAMES, CHE_OBJ_NAMES,
                    _path('staged_che_pareto.csv'))

    with open(_path('staged_che_run_info.json'), 'w') as f:
        json.dump({
            'phase': 'CHE', 'particle_type': 'pi',
            'n_events': n_pi, 'pareto_size': len(che_pareto),
            'wall_time_h': round(che_wall / 3600, 3),
            'param_names': CHE_PARAM_NAMES, 'obj_names': CHE_OBJ_NAMES,
            'lower_bounds': CHE_LOWER, 'upper_bounds': CHE_UPPER,
            'default_params': CHE_DEFAULTS,
            'fixed_cee_params': best_cee_params.tolist(),
            'clue3d_baselines': {'pi': CLUE3D_BASELINES['pi']},
        }, f, indent=2)
    print("  staged_che_run_info.json")

    _save_combined_csv(best_cee_params, best_cee_fitness, che_pareto,
                       _path('pareto_front.csv'))

    with open(_path('run_info.json'), 'w') as f:
        json.dump({
            'mode': 'staged',
            'n_events': len(events), 'n_ee': n_ee, 'n_pi': n_pi,
            'num_particles': NUM_PARTICLES, 'num_iterations': NUM_ITERATIONS,
            'pareto_size': len(che_pareto),
            'wall_time_h': round((cee_wall + che_wall) / 3600, 3),
            'cee_wall_time_h': round(cee_wall / 3600, 3),
            'che_wall_time_h': round(che_wall / 3600, 3),
            'param_names': PARAM_NAMES, 'obj_names': OBJ_NAMES,
            'lower_bounds': LOWER_BOUNDS, 'upper_bounds': UPPER_BOUNDS,
            'default_params': DEFAULT_PARAMS,
            'random_seed': RANDOM_SEED,
            'clue3d_baselines': CLUE3D_BASELINES,
            'fixed_cee_params': best_cee_params.tolist(),
            'fixed_cee_fitness': best_cee_fitness.tolist(),
        }, f, indent=2)
    print("  run_info.json")

    total_wall = cee_wall + che_wall
    print(f"\nAll done — total wall time {total_wall / 3600:.2f}h  "
          f"(CEE {cee_wall / 3600:.2f}h + CHE {che_wall / 3600:.2f}h)")


if __name__ == '__main__':
    main()
