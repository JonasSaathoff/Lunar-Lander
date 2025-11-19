#!/usr/bin/env python3
"""Run a small smoke test that wraps the existing GymProblem with IOH logging.
Generates an IOH-style `IOH_Data` folder suitable for zipping and uploading to IOHanalyzer.
"""
import os
import zipfile
import numpy as np
from problem import GymProblem

import ioh


def problem_wrapper(gym_problem: GymProblem):
    def wrapped(x):
        # GymProblem returns (returns, rewards)
        val = gym_problem(x)
        if isinstance(val, tuple) or isinstance(val, list):
            returns = val[0]
        else:
            # fallback
            returns = val
        return returns
    return wrapped


def random_search_on_ioh(ioh_problem, budget=1000, seed=0):
    np.random.seed(seed)
    lb = ioh_problem.bounds.lb
    ub = ioh_problem.bounds.ub
    n = ioh_problem.meta_data.n_variables

    best_x = np.random.uniform(lb, ub, size=n)
    best_f = ioh_problem(best_x)

    for i in range(1, budget):
        x = np.random.uniform(lb, ub, size=n)
        f = ioh_problem(x)
        if f > best_f:
            best_f = f
            best_x = x.copy()
    return best_x, best_f


def main():
    out_root = "IOH_Data"
    os.makedirs(out_root, exist_ok=True)

    # create gym problem
    gym_problem = GymProblem()

    # create wrapped function and IOH problem
    wrapped = problem_wrapper(gym_problem)

    ioh_problem = ioh.wrap_problem(
        function=wrapped,
        name="LunarLander",
        problem_class=ioh.ProblemClass.REAL,
        dimension=gym_problem.n_variables,
        optimization_type=ioh.OptimizationType.MAX,
        lb=-1.0,
        ub=1.0,
    )

    # create a logger/analyzer
    logger = ioh.logger.Analyzer(
        root=out_root,
        folder_name="RandomSearch_LunarLander",
        algorithm_name="RandomSearch",
        store_positions=True,
    )

    # attach and run one short experiment
    ioh_problem.attach_logger(logger)
    print("Running a short random search with IOH logging (budget=200)...")
    best_x, best_f = random_search_on_ioh(ioh_problem, budget=200, seed=42)
    print("Done. best_f=", best_f)

    ioh_problem.detach_logger()
    logger.close()

    # zip the generated folder for upload
    zip_name = "IOH_Data_ioh_smoke.zip"
    with zipfile.ZipFile(zip_name, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(out_root):
            for f in files:
                path = os.path.join(root, f)
                arcname = os.path.relpath(path, start=os.path.dirname(out_root))
                zf.write(path, arcname=arcname)

    print(f"Created {zip_name}")


if __name__ == '__main__':
    main()
