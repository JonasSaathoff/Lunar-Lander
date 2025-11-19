"""
Create an IOH-style bundle (IOH_Data/...) from simple per-run CSVs.

Expect input_dir structure:
  input_dir/
    adaptive_de/
      seed_1.csv
      seed_2.csv
    random/
      seed_1.csv
    ...

Each seed_*.csv must be two columns with header (evaluation,best_f) or similar.
The script writes a minimal IOH_Data bundle into output_dir and optionally zips it.

Usage (from project root):
  python create_ioh_bundle.py \
    --input IOH_Data_raw \
    --output IOH_Data_bundle \
    --func-id 3 --func-name Rastrigin --dim 2 --maximize 0 \
    --zip

"""
import os, csv, json, argparse, zipfile
from pathlib import Path

def read_run_csv(fn):
    """Return list of (eval, best_f) from CSV, tries to skip header."""
    rows = []
    with open(fn, newline='') as f:
        reader = csv.reader(f)
        first = True
        for r in reader:
            if not r:
                continue
            # try to parse numbers; skip header if non-numeric
            try:
                if first:
                    float(r[0]); float(r[1])
                rows.append((int(float(r[0])), float(r[1])))
            except Exception:
                # skip header / non-numeric first row
                pass
            first = False
    return rows

def make_ioh_bundle(input_dir, output_dir, func_id, func_name, dim, maximize):
    input_dir = Path(input_dir)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for algo_dir in sorted([d for d in input_dir.iterdir() if d.is_dir()]):
        algo_name = algo_dir.name
        target_root = out / f"{algo_name}_{func_name}"
        target_root.mkdir(parents=True, exist_ok=True)

        data_folder = target_root / f"data_f{func_id}_{func_name}"
        data_folder.mkdir(exist_ok=True)

        # dat file path used in JSON 'path'
        dat_filename = f"IOHprofiler_f{func_id}_DIM{dim}.dat"
        dat_path = data_folder / dat_filename

        runs_meta = []
        # create/overwrite dat file; append each run separated by a blank line
        with open(dat_path, 'w', newline='') as datf:
            # write a simple header line (not strictly required by uploader, but helpful)
            datf.write("# evaluations raw_y\n")
            for seed_fn in sorted(algo_dir.glob("seed_*.csv")):
                rows = read_run_csv(seed_fn)
                if not rows:
                    continue
                for ev, val in rows:
                    datf.write(f"{ev} {val}\n")
                datf.write("\n")  # blank line separates runs
                # metadata about this run for the JSON file
                last_eval, best_y = rows[-1]
                # we don't have x (solution vector) here; leave empty list
                runs_meta.append({
                    "instance": 1,
                    "evals": last_eval,
                    "best": {"evals": last_eval, "y": float(best_y), "x": []}
                })

        # create minimal IOHprofiler json manifest (mirrors your example)
        json_obj = {
            "version": "0.3.22",
            "suite": "unknown_suite",
            "function_id": func_id,
            "function_name": func_name,
            "maximization": bool(int(maximize)),
            "algorithm": {"name": algo_name, "info": "{}"},
            "attributes": ["evaluations", "raw_y"],
            "scenarios": [
                {
                    "dimension": dim,
                    "path": str(Path(data_folder.name) / dat_filename),
                    "runs": runs_meta
                }
            ]
        }
        json_path = target_root / f"IOHprofiler_f{func_id}_{func_name}.json"
        with open(json_path, 'w') as jf:
            json.dump(json_obj, jf, indent=4)

    return out

def zip_bundle(bundle_dir, out_zip):
    bundle_dir = Path(bundle_dir)
    with zipfile.ZipFile(out_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in bundle_dir.rglob('*'):
            z.write(p, p.relative_to(bundle_dir.parent))
    return out_zip

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="IOH_Data", help="Input dir with per-algo subdirs")
    p.add_argument("--output", default="IOH_Data_bundle", help="Output IOH_Data-like bundle dir")
    p.add_argument("--func-id", type=int, default=3)
    p.add_argument("--func-name", default="Rastrigin")
    p.add_argument("--dim", type=int, default=2)
    p.add_argument("--maximize", default=0, choices=[0,1], type=int)
    p.add_argument("--zip", action="store_true", help="Create ioh_all_bundle.zip next to output")
    args = p.parse_args()

    bundle_dir = make_ioh_bundle(args.input, args.output, args.func_id, args.func_name, args.dim, args.maximize)
    print(f"Created IOH-style bundle at: {bundle_dir}")

    if args.zip:
        out_zip = bundle_dir.parent / "ioh_all_bundle.zip"
        zip_bundle(bundle_dir, out_zip)
        print(f"Wrote zip: {out_zip}")

if __name__ == "__main__":
    main()