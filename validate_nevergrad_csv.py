#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path
import sys

p = Path('aggregated_nevergrad.csv')
if not p.exists():
    print('File not found:', p)
    sys.exit(2)

required = ['function_class','function_id','instance','dimension','algorithm','seed','evaluation','best_f']

bad = False
with p.open('r', newline='') as fh:
    reader = csv.reader(fh)
    try:
        header = next(reader)
    except StopIteration:
        print('Empty file')
        sys.exit(2)

    print('Header columns:', header[:20])
    # check required present
    missing = [c for c in required if c not in header]
    if missing:
        print('Missing required columns:', missing)
        bad = True

    idx = {name: header.index(name) for name in header}

    # quick row checks
    per_group_eval = defaultdict(list)
    per_group_best = defaultdict(list)
    rownum = 1
    for row in reader:
        rownum += 1
        if len(row) != len(header):
            print(f'Row {rownum}: wrong number of columns ({len(row)} vs {len(header)})')
            bad = True
            if rownum < 20:
                print('  sample:', row)
            continue
        try:
            alg = row[idx['algorithm']]
            seed = row[idx['seed']]
            evalv = int(row[idx['evaluation']])
            bestv = float(row[idx['best_f']])
        except Exception as e:
            print(f'Row {rownum}: parsing error: {e} -- row sample: {row[:10]}')
            bad = True
            continue
        key = (alg, seed)
        per_group_eval[key].append((evalv, rownum))
        per_group_best[key].append((bestv, rownum))

    # Check monotonic evaluation per group
    for key, evlist in list(per_group_eval.items())[:10]:
        evs = [e for e,_ in evlist]
        if evs != sorted(evs):
            print('Evaluation not non-decreasing for', key, 'sample first values:', evs[:10])
            bad = True

    # Check best_f non-decreasing (best-so-far means non-decreasing if larger means better)
    for key, blist in list(per_group_best.items())[:10]:
        bs = [b for b,_ in blist]
        # We expect best_f to be non-decreasing (monotonic non-decreasing)
        for i in range(1, len(bs)):
            if bs[i] < bs[i-1] - 1e-9:
                print('best_f decreased for', key, 'at index', i, 'vals', bs[i-3:i+1])
                bad = True
                break

print('\nValidation result:')
if bad:
    print('Issues found. Please review the messages above.')
    sys.exit(1)
else:
    print('No obvious issues found. CSV looks structurally valid.')
    sys.exit(0)
