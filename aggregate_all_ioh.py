#!/usr/bin/env python3
"""Aggregate all per-run IOH-style CSVs under any ioh* folders into one CSV.

Search pattern: any directory in the repo named starting with 'ioh' (e.g. ioh_runs, ioh_adaptive,
ioh_instrumented_batch). For each file matching 'seed_*.csv' under those folders we read the
'evaluation,best_f' data and append rows to a single aggregated CSV with columns:

  problem,source_root,algorithm,seed,evaluation,best_f

This script writes 'aggregated_ioh_all.csv' in the repo root.

It is safe to re-run; existing aggregated file will be overwritten.
"""
import csv
import pathlib
import argparse

ROOT = pathlib.Path(__file__).resolve().parent


def find_ioh_roots(root: pathlib.Path):
    # directories directly under repo root that start with 'ioh'
    return [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.startswith('ioh')]


def aggregate(roots, out_path: pathlib.Path):
    rows = []
    for root in roots:
        # look for seed_*.csv under immediate children (algorithm folders) and deeper
        for csv_file in sorted(root.rglob('seed_*.csv')):
            try:
                algorithm = csv_file.parent.name
                seed = csv_file.stem.split('_', 1)[-1]
                with csv_file.open() as fh:
                    reader = csv.DictReader(fh)
                    # expect evaluation and best_f columns
                    if 'evaluation' not in reader.fieldnames or 'best_f' not in reader.fieldnames:
                        print(f"Skipping {csv_file}: missing expected columns")
                        continue
                    for r in reader:
                        rows.append({
                            'problem': 'LunarLander-v3',
                            'source_root': root.name,
                            'algorithm': algorithm,
                            'seed': seed,
                            'evaluation': r['evaluation'],
                            'best_f': r['best_f']
                        })
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
    # write aggregated
    with out_path.open('w', newline='') as fh:
        fieldnames = ['problem', 'source_root', 'algorithm', 'seed', 'evaluation', 'best_f']
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', '-o', default=str(ROOT / 'aggregated_ioh_all.csv'))
    parser.add_argument('--zip', action='store_true', help='also create a zip of all ioh* folders')
    args = parser.parse_args()

    roots = find_ioh_roots(ROOT)
    if not roots:
        print('No ioh* folders found in repo root; nothing to aggregate.')
        raise SystemExit(1)

    out_path = pathlib.Path(args.out)
    n = aggregate(roots, out_path)
    print(f'Wrote {n} rows to {out_path}')

    if args.zip:
        import subprocess
        zip_name = ROOT / 'ioh_all_bundle.zip'
        # build list of directories to zip
        dirs = [str(p.name) for p in roots]
        cmd = ['zip', '-r', str(zip_name)] + dirs
        print('Creating zip bundle:', ' '.join(cmd))
        subprocess.run(cmd, cwd=str(ROOT))
        print('Wrote zip bundle to', zip_name)
