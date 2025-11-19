#!/usr/bin/env python3
"""Convert aggregated CSV to a NEVERGRAD/IOH-compatible CSV by adding required metadata columns.

Usage:
  python convert_to_nevergrad_csv.py --in aggregated_ioh_all.csv --out aggregated_nevergrad.csv

This script will add columns: function_class,function_id,instance,dimension and keep
algorithm,seed,evaluation,best_f. It will attempt to infer `dimension` from `problem.py` if available.
"""
import argparse
import pathlib
import csv
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='infile', default='aggregated_ioh_all.csv')
parser.add_argument('--out', dest='outfile', default='aggregated_nevergrad.csv')
args = parser.parse_args()

IN = pathlib.Path(args.infile)
OUT = pathlib.Path(args.outfile)

if not IN.exists():
    print(f'Input file not found: {IN}', file=sys.stderr)
    raise SystemExit(1)

# try to infer dimension from problem.GymProblem if available
dimension = ''
try:
    from problem import GymProblem
    gp = GymProblem()
    if hasattr(gp, 'n_variables'):
        dimension = int(gp.n_variables)
    else:
        dimension = ''
except Exception:
    dimension = ''

print(f'Infile: {IN} -> Outfile: {OUT}; inferred dimension={dimension}')

with IN.open() as inf, OUT.open('w', newline='') as outf:
    reader = csv.DictReader(inf)
    # determine available cols and write new header
    fieldnames = ['function_class', 'function_id', 'instance', 'dimension', 'algorithm', 'seed', 'evaluation', 'best_f']
    writer = csv.DictWriter(outf, fieldnames=fieldnames)
    writer.writeheader()
    for r in reader:
        # map values
        algorithm = r.get('algorithm') or r.get('source_root') or 'unknown'
        seed = r.get('seed', '')
        evaluation = r.get('evaluation', '')
        best_f = r.get('best_f', '')
        out = {
            'function_class': 'LunarLander-v3',
            'function_id': 1,
            'instance': 1,
            'dimension': dimension,
            'algorithm': algorithm,
            'seed': seed,
            'evaluation': evaluation,
            'best_f': best_f,
        }
        writer.writerow(out)

print('Wrote', OUT)
