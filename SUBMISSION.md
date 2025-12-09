Submission contents and instructions

Included files:
- Core source: `problem.py`, `run_replicates_ioh_all.py`, `random_search.py`, `adapters.py`, `adaptive_mixed_de.py`, and helper scripts.
- Utilities: `verify_continuous.py`, `instrumented_problem.py`, `eval_and_resim.py`, `resimulate_best_from_ioh.py`, `create_ioh_bundle.py`, `create_per_algo_zips.py` (if present), `viz.py`
- Documentation: `README.md`, `LICENSE`, `requirements.txt`, `run_sample.sh`.

Notes:
- The final report is not included — add `Report.tex` / PDF to this folder before creating the final ZIP.
- Large IOH output folders and zip archives are excluded intentionally.

Quick start:
1. Create a venv and install requirements:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the sample smoke test:
   ```bash
   ./run_sample.sh
   ```
3. When ready, create the submission ZIP (from the parent folder):
   ```bash
   zip -r LunarLander_submission_clean.zip LunarLander_submission_clean
   shasum -a 256 LunarLander_submission_clean.zip > LunarLander_submission_clean.zip.sha256
   ```
