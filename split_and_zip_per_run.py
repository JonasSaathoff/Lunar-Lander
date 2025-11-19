#!/usr/bin/env python3
import csv
from pathlib import Path
from collections import defaultdict
import zipfile

root = Path('.').resolve()
infile = root / 'aggregated_nevergrad.csv'
out_zip = root / 'ioh_upload_per_run.zip'

if not infile.exists():
    print('aggregated_nevergrad.csv not found')
    raise SystemExit(1)

# collect rows per (function_class, algorithm, seed)
groups = defaultdict(list)

with infile.open('r', newline='') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        key = (row['function_class'], row['algorithm'], row['seed'])
        groups[key].append((int(row['evaluation']), row['best_f']))

# write each group as top-level CSV inside zip
with zipfile.ZipFile(out_zip, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for (func, algo, seed), vals in groups.items():
        # sort by evaluation
        vals.sort()
        fname = f"{func}__{algo}__seed_{seed}.csv"
        # create CSV content
        lines = ['evaluation,best_f']
        for ev, bf in vals:
            lines.append(f"{ev},{bf}")
        data = '\n'.join(lines) + '\n'
        z.writestr(fname, data)
        print('Added', fname, 'rows=', len(vals))

print('Created', out_zip, 'size=', out_zip.stat().st_size)
