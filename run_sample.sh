#!/usr/bin/env bash
set -euo pipefail

# Quick smoke test (small budget) — graders can run this after installing deps
python run_replicates_ioh_all.py --presets nominal_continuous --algos adaptive_mixed_de --reps 1 --budget 200 --out-root sample_ioh --clean

echo "Sample run complete. Check sample_ioh/ for outputs."