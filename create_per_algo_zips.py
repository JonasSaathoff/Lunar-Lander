#!/usr/bin/env python3
import sys
from pathlib import Path
import zipfile

root = Path('.').resolve()
ioh_dirs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith('ioh')]
if not ioh_dirs:
    print('No ioh* directories found in', root)
    sys.exit(0)

out_dir = root / 'ioh_bundle_per_algo'
out_dir.mkdir(exist_ok=True)

# Map algorithm name -> list of files
alg_files = {}
for ioh in ioh_dirs:
    for f in ioh.rglob('*.csv'):
        # algorithm name is the immediate parent directory (e.g., ioh_adaptive/adaptive_de/seed_0.csv)
        algo = f.parent.name
        alg_files.setdefault(algo, []).append(f)

if not alg_files:
    print('No CSV files found under ioh* directories')
    sys.exit(0)

created = []
for algo, files in sorted(alg_files.items()):
    zip_path = out_dir / f'{algo}.zip'
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            # store as algo/seed_xxx.csv inside zip
            arcname = f'{algo}/{f.name}'
            z.write(f, arcname)
    size = zip_path.stat().st_size
    created.append((zip_path, len(files), size))
    print(f'Created {zip_path} with {len(files)} files, size={size} bytes')

print('\nSummary:')
for p, count, size in created:
    kb = size/1024
    print(f'- {p.name}: {count} files, {kb:.1f} KB')

print('\nDone.')
