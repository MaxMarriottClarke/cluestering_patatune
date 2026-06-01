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

import argparse
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_last_cee_params_and_fitness():
    """Load best CEE params and fitness from the most recent run in OUTPUT_DIR."""
    cee_info = os.path.join(OUTPUT_DIR, 'staged_cee_run_info.json')
    run_info  = os.path.join(OUTPUT_DIR, 'run_info.json')

    for path, p_key, f_key in [
        (cee_info, 'best_balanced_params', 'best_balanced_fitness'),
        (run_info,  'fixed_cee_params',     'fixed_cee_fitness'),
    ]:
        if os.path.exists(path):
            with open(path) as fh:
                info = json.load(fh)
            params  = np.array(info[p_key])
            fitness = np.array(info.get(f_key, [np.nan, np.nan, np.nan]))
            print(f"  source : {path}")
            return params, fitness

    print("  WARNING: no previous CEE results found — falling back to DEFAULT_PARAMS[:5]")
    return np.array(DEFAULT_PARAMS[:5]), np.full(3, np.nan)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    # ── CLI ───────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description='Staged CLUEstering parameter optimisation',
    )
    parser.add_argument(
        '--phase', choices=['both', 'cee', 'che'], default='both',
        help=(
            'Phases to run.  "both" (default): Phase 1 CEE on ee events then '
            'Phase 2 CHE on pi events.  "cee": Phase 1 only.  '
            '"che": Phase 2 only — CEE params loaded from the last run in '
            f'{OUTPUT_DIR}/ (the default results dir) unless --cee-params is given.'
        ),
    )
    parser.add_argument(
        '--cee-params', nargs=5, type=float,
        metavar=('DR_CEE', 'MD_CEE', 'OD_CEE', 'SD_CEE', 'WZ_CEE'),
        help='Fixed CEE params for --phase che '
             '(density_radius  min_density  outlier_distance  seeding_distance  w_z). '
             'If omitted, loaded automatically from the last run.',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help=f'Directory for output files (default: {OUTPUT_DIR} from config.py). '
             'Created if it does not exist.',
    )
    args = parser.parse_args()

    out_dir = args.output_dir or OUTPUT_DIR
    _path   = lambda fn: os.path.join(out_dir, fn)   # shadows module-level _path
    os.makedirs(out_dir, exist_ok=True)
    patatune.Randomizer.rng = np.random.default_rng(RANDOM_SEED)
    patatune.Logger.setLevel('INFO')

    # ── Load and validate ─────────────────────────────────────────────────────
    print("=== Loading data ===", flush=True)
    all_events = load_events()
    validate_events(all_events)

    events = _subsample_events(all_events, N_EVENTS)
    n_ee   = sum(1 for e in events if e['particle_type'] == 'ee')
    n_pi   = sum(1 for e in events if e['particle_type'] == 'pi')
    print(f"Using {len(events)} events  ({n_ee} ee + {n_pi} pi)\n", flush=True)

    # ── Resolve fixed CEE params for CHE-only run ─────────────────────────────
    best_cee_params  = None
    best_cee_fitness = None
    best_idx         = None

    if args.phase == 'che':
        if args.cee_params:
            best_cee_params  = np.array(args.cee_params)
            best_cee_fitness = np.full(3, np.nan)
            print("=== Fixed CEE params (from --cee-params) ===")
        else:
            print("=== Loading CEE params from last run ===")
            best_cee_params, best_cee_fitness = _load_last_cee_params_and_fitness()
        for n, v in zip(CEE_PARAM_NAMES, best_cee_params):
            print(f"  {n} = {v:.4f}")
        if not np.any(np.isnan(best_cee_fitness)):
            print("  objectives (R1_ee, R2_ee, R3_ee) : " +
                  "  ".join(f"{v:.4f}" for v in best_cee_fitness))
        print(flush=True)

    # ── Timing probe ──────────────────────────────────────────────────────────
    print("=== Timing probe ===", flush=True)
    probe_events = _subsample_events(all_events, 20)
    if args.phase == 'che':
        probe_subdet      = 'CHE'
        probe_fixed_other = best_cee_params
        probe_defaults    = CHE_DEFAULTS
        probe_obj_names   = CHE_OBJ_NAMES
        n_probe_ptype     = sum(1 for e in probe_events if e['particle_type'] == 'pi')
        n_scale           = n_pi
    else:
        probe_subdet      = 'CEE'
        probe_fixed_other = CHE_DEFAULTS
        probe_defaults    = CEE_DEFAULTS
        probe_obj_names   = CEE_OBJ_NAMES
        n_probe_ptype     = sum(1 for e in probe_events if e['particle_type'] == 'ee')
        n_scale           = n_ee

    probe_obj    = make_staged_objective(probe_events, n_jobs=1, subdet=probe_subdet,
                                         fixed_other_params=probe_fixed_other)
    t0           = time.time()
    probe_result = probe_obj(probe_defaults)
    t_call       = time.time() - t0
    if probe_obj._pool is not None:
        probe_obj._pool.close(); probe_obj._pool.join()

    t_scaled = t_call / max(n_probe_ptype, 1) * n_scale / N_JOBS
    n_phases = 2 if args.phase == 'both' else 1
    print(f"  probe ({n_probe_ptype} {probe_subdet} events, 1 worker) : {t_call:.2f}s")
    print(f"  scaled to {n_scale} events / {N_JOBS} workers           : {t_scaled:.1f}s per particle")
    print(f"  est. per iteration  : {t_scaled * NUM_PARTICLES / 60:.1f} min")
    print(f"  est. total ({n_phases} phase(s)) : {n_phases * t_scaled * NUM_PARTICLES * NUM_ITERATIONS / 3600:.1f} h")
    print(f"  probe result : " + "  ".join(
          f"{n}={v:.4f}" for n, v in zip(probe_obj_names, probe_result)), flush=True)
    print()

    # ── Phase 1: CEE params on ee events ──────────────────────────────────────
    cee_pareto = None
    cee_wall   = 0.0
    if args.phase in ('both', 'cee'):
        obj_cee = make_staged_objective(events, n_jobs=N_JOBS, subdet='CEE',
                                        fixed_other_params=CHE_DEFAULTS)
        cee_pareto, cee_wall = _run_mopso(
            'Phase 1 — CEE (ee events only)',
            obj_cee, CEE_PARAM_NAMES, CEE_OBJ_NAMES,
            CEE_LOWER, CEE_UPPER, CEE_DEFAULTS,
        )
        if obj_cee._pool is not None:
            obj_cee._pool.close(); obj_cee._pool.join()

        best_idx         = _balanced_best(cee_pareto)
        best_cee_params  = cee_pareto[best_idx].position
        best_cee_fitness = cee_pareto[best_idx].fitness

        print("Best balanced CEE solution (fixed params for Phase 2):")
        for n, v in zip(CEE_PARAM_NAMES, best_cee_params):
            print(f"  {n} = {v:.4f}")
        print("  objectives: " + "  ".join(
              f"{n}={v:.4f}" for n, v in zip(CEE_OBJ_NAMES, best_cee_fitness)))
        print(flush=True)

    # ── Phase 2: CHE params on pi events, CEE fixed ───────────────────────────
    che_pareto = None
    che_wall   = 0.0
    if args.phase in ('both', 'che'):
        obj_che = make_staged_objective(events, n_jobs=N_JOBS, subdet='CHE',
                                        fixed_other_params=best_cee_params)
        che_pareto, che_wall = _run_mopso(
            'Phase 2 — CHE (pi events only, CEE fixed)',
            obj_che, CHE_PARAM_NAMES, CHE_OBJ_NAMES,
            CHE_LOWER, CHE_UPPER, CHE_DEFAULTS,
        )
        if obj_che._pool is not None:
            obj_che._pool.close(); obj_che._pool.join()

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n=== Saving to {out_dir}/ ===")

    if cee_pareto is not None:
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

    if che_pareto is not None:
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
                'phase_run': args.phase,
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
    phase_detail = (f"  (CEE {cee_wall / 3600:.2f}h + CHE {che_wall / 3600:.2f}h)"
                    if args.phase == 'both' else "")
    print(f"\nAll done — total wall time {total_wall / 3600:.2f}h{phase_detail}")


if __name__ == '__main__':
    main()
