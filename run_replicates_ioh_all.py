#!/usr/bin/env python3
"""Run all algorithms with IOH logging enabled.
This script will:
 - For each algorithm, for each replicate seed, create a GymProblem
 - Wrap the GymProblem evaluation with an IOH problem and Analyzer logger
 - Patch the instance's play_episode to call the IOH-wrapped problem (so logging happens)
 - Call the algorithm via adapters.<algo>() so existing implementations are reused
 - Detach logger and continue

Produces an `IOH_Data_all` folder that can be zipped for IOHanalyzer.
"""
import os
import zipfile
import time
from types import MethodType
import shutil
import argparse

import ioh

from problem import GymProblem
import adapters
from algorithms import adaptive_differential_evolution

ALGOS = [
    "adaptive_de",
    "adaptive_mixed_de",
    "de",
    "rand2",
    "rand_to_best2",
    "best1",
    "best2",
    "current_to_best1",
    "combo",
    "plain",
    "random",
    "self",
]
REPS = 30
BUDGET = 20000
SEED_START = 1000
OUT_ROOT = "IOH_baseline_6x1k"


def run_one_algo(algo_name: str, reps=30, budget=1000, seed_start=1000):
    print(f"=== Running algorithm {algo_name} (reps={reps}, budget={budget}) ===")
    for i in range(reps):
        seed = seed_start + i
        print(f"  Run {i+1}/{reps}, seed={seed}")
        # create Gym problem instance
        gym_problem = GymProblem()
        # Ensure environments created by this GymProblem instance disable
        # the passive env checker so that numpy-array actions produced by
        # our linear policy are accepted. We wrap env_spec.make per-instance
        # to avoid editing `problem.py`.
        # No per-instance env wrapping needed for targeted runs.
        # Ensure environments created include disable_env_checker so Gym's
        # passive checker does not reject numpy action arrays produced by our policies.
        try:
            gym_problem.simulation_params['disable_env_checker'] = True
        except Exception:
            pass

        # save original bound method for evaluation (to be used by IOH wrapped function)
        orig_play = gym_problem.play_episode

        # wrapped function used by ioh to actually run the environment once and get scalar
        def wrapped_fn(x):
            # call the original play_episode of this instance to get (returns, rewards)
            val = orig_play(x)
            # orig_play may return (returns, rewards) or just returns
            if isinstance(val, tuple) or isinstance(val, list):
                returns = val[0]
            else:
                returns = val
            return returns

        # create an IOH problem (only used for logging/metadata)
        ioh_problem = ioh.wrap_problem(
            function=wrapped_fn,
            name="LunarLander",
            problem_class=ioh.ProblemClass.REAL,
            dimension=gym_problem.n_variables,
            optimization_type=ioh.OptimizationType.MAX,
            lb=-1.0,
            ub=1.0,
        )

        # create logger for this algorithm/problem
        # Use a consistent folder name per algorithm so runs are grouped
        logger = ioh.logger.Analyzer(
            root=OUT_ROOT,
            folder_name=f"{algo_name}_LunarLander",
            algorithm_name=algo_name,
            store_positions=True,
        )

        # Attach logger to ioh_problem so that calls to ioh_problem() will be logged
        ioh_problem.attach_logger(logger)

        # Attach play_episode to route through ioh_problem
        def patched_play(self, x, **env_kwargs):
            val = ioh_problem(x)
            return val, []

        gym_problem.play_episode = MethodType(patched_play, gym_problem)

        # Run the algorithm via adapter or direct function where needed
        try:
            if algo_name == 'adaptive_de':
                # call the adaptive DE implementation directly
                alg_func = adaptive_differential_evolution.adaptive_differential_evolution
            else:
                alg_func = getattr(adapters, algo_name)

            start = time.time()
            # call the algorithm; many implementations accept (problem, budget, seed,...)
            alg_func(gym_problem, budget=budget, seed=seed, print_every=0)
            dur = time.time() - start
            print(f"    seed={seed} done (took {dur:.1f}s)")
        except Exception as e:
            print(f"    seed={seed} failed: {e}")
            import traceback
            traceback.print_exc(limit=12)

        # detach and close logger
        try:
            ioh_problem.detach_logger()
            logger.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='Run all algos with IOH logging')
    parser.add_argument('--reps', type=int, default=REPS, help='Number of replicates per algorithm')
    parser.add_argument('--budget', type=int, default=BUDGET, help='Budget/iterations for each alg')
    parser.add_argument('--seed-start', type=int, default=SEED_START, help='Starting seed (increments per replicate)')
    parser.add_argument('--out-root', type=str, default=OUT_ROOT, help='Output folder for IOH logs')
    parser.add_argument('--clean', action='store_true', help='Remove existing out-root before running')
    parser.add_argument('--force-exact', action='store_true', help='Use the exact --budget and --reps for all algorithms (ignore budget-evals mapping)')
    parser.add_argument('--presets',    type=str, default='nominal', help='Comma-separated list of presets (e.g. nominal,wind_med,wind_high)')
    parser.add_argument('--algos', type=str, default=None, help='Comma-separated list of algos to run (default: all)')
    parser.add_argument('--budget-evals', type=int, default=None, help='Total environment evaluations target (overrides --budget by translating per-algo iterations)')
    parser.add_argument('--dry-run', action='store_true', help='Print planned presets/algos/budgets and exit without running')
    args = parser.parse_args()

    # echo configuration for reproducibility
    print(f"Configuration: presets={args.presets} algos={args.algos or 'ALL'} reps={args.reps} budget={args.budget} budget_evals={args.budget_evals} seed_start={args.seed_start} out_root={args.out_root} force_exact={args.force_exact}")

    out_root = args.out_root
    if os.path.exists(out_root):
        if args.clean:
            print(f"Removing existing output folder {out_root}")
            shutil.rmtree(out_root)
        else:
            ts = int(time.time())
            new_out = f"{out_root}_{ts}"
            print(f"Output folder {out_root} exists; using new folder {new_out} to avoid mixing previous runs")
            out_root = new_out

    os.makedirs(out_root, exist_ok=True)

    presets = [p.strip() for p in args.presets.split(',') if p.strip()]
    if args.algos:
        requested = [a.strip() for a in args.algos.split(',') if a.strip()]
        algos = [a for a in requested if a in ALGOS]
        unknown = [a for a in requested if a not in ALGOS]
        if unknown:
            print(f"Warning: unknown algos ignored: {unknown}")
    else:
        algos = ALGOS

        # Diagnostic: list available adapter callables so we can detect mismatches
        available_adapters = [n for n in dir(adapters) if not n.startswith('_')]
        print(f"Available adapters: {available_adapters}")
        print(f"Requested/selected algos to run: {algos}")

        # Ensure each requested algo is actually callable. Special-case 'adaptive_de'
        # which maps directly to `adaptive_differential_evolution` in this script.
        effective_algos = []
        for a in algos:
            if a == 'adaptive_de' or (a in available_adapters):
                effective_algos.append(a)
            else:
                print(f"Warning: algorithm '{a}' not available in adapters and will be skipped")
        algos = effective_algos
        print(f"Final algos list after availability check: {algos}")

        if args.dry_run:
            print("Dry run requested — no algorithms will be executed.")
            # Show per-preset/per-algo mapped budgets that would be used
            for preset in presets:
                env_kwargs = env_kwargs_for(preset)
                print(f"Preset '{preset}' env_kwargs={env_kwargs}")
                for algo in algos:
                    # Dry-run shows the exact `--budget` that will be used for
                    # every algorithm. We intentionally ignore `--budget-evals` to
                    # avoid implicit remapping that caused surprises.
                    effective_budget = args.budget
                    print(f"  Algo={algo} -> effective budget (iterations) = {effective_budget}")
            return

    def env_kwargs_for(preset: str):
        if preset == 'nominal':
            return {}
        if preset == 'nominal_continuous' or preset == 'continuous':
            return dict(continuous=True)
        if preset == 'wind_med':
            return dict(enable_wind=True, wind_power=8.0, turbulence_power=1.5)
        if preset == 'wind_high':
            return dict(enable_wind=True, wind_power=15.0, turbulence_power=2.5)
        if preset == 'gravity_low':
            return dict(gravity=-5.0)
        # Gym's LunarLander asserts gravity must be > -12.0 and < 0.0.
        # Use safe values inside that open interval (avoid -12.0 or lower).
        if preset == 'gravity_high':
            return dict(gravity=-11.5)
        if preset == 'turbulence_high':
            return dict(turbulence_power=3.0)
        if preset == 'wind_and_gravity':
            # keep gravity slightly above -12.0 to satisfy gym assertions
            return dict(enable_wind=True, wind_power=12.0, turbulence_power=2.0, gravity=-11.9)
        return {}

    for preset in presets:
        preset_out = os.path.join(out_root, preset)
        os.makedirs(preset_out, exist_ok=True)
        print(f"Running preset '{preset}' -> output: {preset_out}")
        env_kwargs = env_kwargs_for(preset)

        for algo in algos:
            for i in range(args.reps):
                seed = args.seed_start + i
                print(f"=== Preset={preset} Algo={algo} Run {i+1}/{args.reps} seed={seed} ===")
                gym_problem = GymProblem(**env_kwargs)

                orig_play = gym_problem.play_episode

                # For continuous action spaces, wrap play_episode to pass
                # disable_env_checker=True to avoid numpy ndarray rejection
                is_continuous = env_kwargs.get('continuous', False)
                
                def wrapped_fn(x):
                    if is_continuous:
                        # Call with disable_env_checker to prevent Gym's passive checker
                        # from rejecting numpy float64 arrays (expects float32)
                        val = orig_play(x, **{**env_kwargs, 'disable_env_checker': True})
                    else:
                        val = orig_play(x)
                    if isinstance(val, tuple) or isinstance(val, list):
                        returns = val[0]
                    else:
                        returns = val
                    return returns

                ioh_problem = ioh.wrap_problem(
                    function=wrapped_fn,
                    name="LunarLander",
                    problem_class=ioh.ProblemClass.REAL,
                    dimension=gym_problem.n_variables,
                    optimization_type=ioh.OptimizationType.MAX,
                    lb=-1.0,
                    ub=1.0,
                )

                logger = ioh.logger.Analyzer(
                    root=preset_out,
                    folder_name=f"{algo}_LunarLander",
                    algorithm_name=algo,
                    store_positions=True,
                )
                ioh_problem.attach_logger(logger)

                def patched_play(self, x, **env_kwargs_local):
                    val = ioh_problem(x)
                    return val, []

                gym_problem.play_episode = MethodType(patched_play, gym_problem)

                try:
                    if algo == 'adaptive_de':
                        alg_func = adaptive_differential_evolution.adaptive_differential_evolution
                    else:
                        alg_func = getattr(adapters, algo)

                    # map requested total environment-evaluations to per-algorithm
                    # iteration budgets so IOH 'evaluations' are comparable.
                    # Use the exact `--budget` value provided (or the module-level
                    # default `BUDGET`) for all algorithms. The previous behavior
                    # remapped a `--budget-evals` convenience flag into per-algo
                    # iteration counts which proved confusing. To keep runs
                    # deterministic and predictable we now always honor `--budget`.
                    effective_budget = args.budget

                    start = time.time()
                    print(f"    using budget for '{algo}': {effective_budget}")
                    alg_func(gym_problem, budget=effective_budget, seed=seed, print_every=0)
                    dur = time.time() - start
                    print(f"    seed={seed} done (took {dur:.1f}s)")
                except Exception as e:
                    print(f"    seed={seed} failed: {e}")
                    import traceback
                    traceback.print_exc(limit=12)

                try:
                    ioh_problem.detach_logger()
                    logger.close()
                except Exception:
                    pass

                # (no checkpoint extraction) -- run only produces IOH logs

    # Create a zip of the raw out_root (original behavior)
    zip_name = f"{out_root}.zip"
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_root):
            for f in files:
                path = os.path.join(root, f)
                arcname = os.path.relpath(path, start=os.path.dirname(out_root))
                zf.write(path, arcname=arcname)
    print(f"Created {zip_name}")


if __name__ == '__main__':
    main()
