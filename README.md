# Lunar Lander Optimization with IOH

Ready-to-run LunarLander optimization suite with multiple evolution strategies
and Differential Evolution variants, instrumented for IOH logging.

## Overview
- Gym wrapper for LunarLander (`GymProblem`) supporting discrete/continuous control and wind/gravity presets.
- DE family (rand/1, rand/2, best/1, best/2, rand-to-best/2, current-to-best/1, adaptive, mixed-strategy) plus (1+1)-ES baselines.
- IOH-integrated runner to benchmark any subset of algorithms and presets.
- Smoke-test script for quick sanity checks.

## Project Structure
```
├── problem.py                       # Gym wrapper used by all algorithms
├── adapters.py                      # Stable call signatures for algorithms
├── algorithms/
│   ├── adaptive_differential_evolution.py
│   ├── adaptive_mixed_de.py
│   ├── differential_evolution.py
│   ├── differential_evolution_rand2.py
│   ├── differential_evolution_rand_to_best2.py
│   ├── differential_evolution_best1.py
│   ├── differential_evolution_best2.py
│   ├── differential_evolution_current_to_best1.py
│   ├── one_plus_one_es.py
│   ├── one_plus_one_self_adaptive.py
│   ├── one_plus_one_combo.py
│   └── random_search.py
├── run_replicates_ioh_all.py        # IOH driver for batch runs
├── run_sample.sh                    # One-line smoke test
├── make_clean_submission.py         # Build clean submission zip
└── requirements.txt
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Usage

### Smoke test (small IOH run)
```bash
./run_sample.sh
```
Runs `adaptive_mixed_de` for 1 replicate with a small budget and writes IOH logs to `sample_ioh/`.

### Custom IOH runs
```bash
python run_replicates_ioh_all.py \
  --presets nominal_continuous \
  --algos adaptive_mixed_de,de,best1 \
  --reps 5 \
  --budget 2000 \
  --out-root IOH_runs \
  --clean
```

Key flags:
- `--presets`: env presets (`nominal`, `nominal_continuous`, `wind_med`, `wind_high`, `gravity_low`, `gravity_high`, `turbulence_high`, `wind_and_gravity`).
- `--algos`: comma-separated names from `adapters.py`.
- `--budget`: iterations per algorithm (one env eval per iteration).
- `--reps`: replicate count; seeds start at 1000.
- `--out-root`: IOH output folder (zipped automatically at end).
- `--clean`: remove existing `out-root` before running.

## GymProblem API (quick reference)

```python
GymProblem(
    continuous: bool = False,
    gravity: float = -10.0,
    enable_wind: bool = False,
    wind_power: float = 0.0,
    turbulence_power: float = 0.0,
)
```

- `.sample()`: draw from [-1, 1]^n where n = state_size × n_outputs.
- `.play_episode(x, **env_kwargs)`: run a rollout; returns `(fitness, rewards)`.
- `__call__(x)`: shorthand for `play_episode(x, **simulation_params)`.
- `.show(x)`: render an episode (`render_mode="human"`).

### Minimal example
```python
from problem import GymProblem
import numpy as np

problem = GymProblem(continuous=False)
x = problem.sample()
f, rewards = problem(x)
print("Fitness:", f)
```

## Build a clean submission zip

If you only need the essential files for grading/sharing:
```bash
python make_clean_submission.py --out submission_clean.zip
```
This collects the algorithms, adapters, runner scripts, and requirements into a single archive.
