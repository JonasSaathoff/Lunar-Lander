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

import numpy as np
import ioh

from problem import GymProblem
import adapters

ALGOS = [
    "adaptive_de",
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
OUT_ROOT = "IOH_Data_all"


def run_one_algo(algo_name: str, reps=30, budget=1000, seed_start=1000):
    print(f"=== Running algorithm {algo_name} (reps={reps}, budget={budget}) ===")
    for i in range(reps):
        seed = seed_start + i
        print(f"  Run {i+1}/{reps}, seed={seed}")
        # create Gym problem instance
        gym_problem = GymProblem()

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

        # Patch this instance's play_episode to route through ioh_problem (so existing algos log via play_episode)
        def patched_play(self, x, **env_kwargs):
            # calling ioh_problem(x) will evaluate using wrapped_fn which calls the saved orig_play
            val = ioh_problem(x)
            # return (returns, rewards) shape to satisfy callers; we don't re-run env so rewards not available
            return val, []

        gym_problem.play_episode = MethodType(patched_play, gym_problem)

        # Run the algorithm via adapter or direct function where needed
        try:
            if algo_name == 'adaptive_de':
                # call the adaptive DE implementation directly
                from adaptive_differential_evolution import adaptive_differential_evolution as alg_func
            else:
                alg_func = getattr(adapters, algo_name)

            start = time.time()
            # call the algorithm; many implementations accept (problem, budget, seed,...)
            alg_func(gym_problem, budget=budget, seed=seed, print_every=0)
            dur = time.time() - start
            print(f"    seed={seed} done (took {dur:.1f}s)")
        except Exception as e:
            print(f"    seed={seed} failed: {e}")

        # detach and close logger
        try:
            ioh_problem.detach_logger()
            logger.close()
        except Exception:
            pass


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    for algo in ALGOS:
        run_one_algo(algo, reps=REPS, budget=BUDGET, seed_start=SEED_START)

    # zip the result folder
    zip_name = "IOH_BUDGET_20000_REPS_30.zip"
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(OUT_ROOT):
            for f in files:
                path = os.path.join(root, f)
                arcname = os.path.relpath(path, start=os.path.dirname(OUT_ROOT))
                zf.write(path, arcname=arcname)
    print(f"Created {zip_name}")


if __name__ == '__main__':
    main()
