# cluestering_patatune

Multi-objective optimisation of [CLUEstering](https://github.com/cms-patatrack/CLUEstering) parameters against CLUE3D truth/reco data using [patatune](https://github.com/cms-patatrack/patatune) MOPSO.

## Quick start

```bash
python optimize.py
```

Loads the ROOT files, computes the CLUE3D baseline, runs MOPSO, and writes `pareto_front.csv`.

## File layout

| File | Purpose |
|---|---|
| `config.py` | **All tunable knobs** — data paths, default CLUEstering params, MOPSO settings, objective weights |
| `data.py` | ROOT loading, LC-to-particle assignment, DataFrame builders |
| `metrics.py` | `compute_r2s`, `compute_baseline` (L1 / r / N_T for CLUE3D) |
| `objective.py` | **Edit this to change the objective** — `_run_and_score`, `make_objective_fn` |
| `optimize.py` | Entry point — wires everything together and runs MOPSO |

## Changing the objective

The three objectives are defined inside `make_objective_fn` in `objective.py`:

| Index | Name | Direction | Meaning |
|---|---|---|---|
| 0 | `purity_delta` | minimise | δ(L1) vs CLUE3D; negative = purer than CLUE3D |
| 1 | `energy_ratio` | maximise | weighted mean energy recovery r |
| 2 | `count_delta` | minimise | δ(N_T) + floor penalty for N_T < 2 |

To add/remove objectives:
1. Modify `_run_and_score()` to compute the new per-event metric.
2. Change the return value of `objective_fn` inside `make_objective_fn()`.
3. Update `directions` and `objective_names` in `optimize.py` to match.

## Changing data or parameters

- **Different files / paths** → edit `PATH` and `CONFIG_FILES` in `config.py`
- **Different ROOT branch names** → edit `TRUTH_BRANCH` / `CLUE_BRANCH`
- **MOPSO budget** → edit `NUM_PARTICLES` / `NUM_ITERATIONS` (the timing probe at startup tells you the projected wall time)
- **EM/HAD balance** → edit `ALPHA` (EM weight; pion weight = 1 − ALPHA)
- **Parameter search range** → edit `LOWER_BOUNDS` / `UPPER_BOUNDS`
