# Clean IOH Submission Bundle

Minimal, ready-to-run package for the LunarLander IOH experiments. Only the
algorithms, problem wrapper, IOH runner, and requirements are included.

## Contents
- `problem.py`: Gym wrapper used by all algorithms
- `adapters.py`: stable call signatures for every algorithm
- Algorithms (in `algorithms/`): `adaptive_differential_evolution.py`,
  `adaptive_mixed_de.py`, `differential_evolution.py`, `differential_evolution_rand2.py`,
  `differential_evolution_rand_to_best2.py`, `differential_evolution_best1.py`,
  `differential_evolution_best2.py`, `differential_evolution_current_to_best1.py`,
  `one_plus_one_es.py`, `one_plus_one_self_adaptive.py`, `one_plus_one_combo.py`,
  `random_search.py`
- IOH driver and sample runner: `run_replicates_ioh_all.py`, `run_sample.sh`
- `requirements.txt`
- `make_clean_submission.py` (creates the zip)

## Quick start
1) Create a venv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Smoke test (1 rep, small budget, IOH logging to `sample_ioh/`):
```bash
./run_sample.sh
```

## Custom IOH run
Run any subset of algorithms/presets with IOH logging:
```bash
python run_replicates_ioh_all.py --presets nominal_continuous --algos adaptive_mixed_de --reps 1 --budget 200 --out-root sample_ioh --clean
```

## Build the clean submission zip
From the repo root:
```bash
python make_clean_submission.py --out submission_clean.zip
```
The zip will contain only the files listed above.
