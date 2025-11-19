import argparse
import ioh
import numpy as np

# import your algorithms (adjust names/modules if different)
# from adaptive_differential_evolution import adaptive_differential_evolution as adaptive_de
# from combo import combo
# from plain import plain
# from random_search import RandomSearch
# from self import self_algo
# If imports above fail, edit them to match your codebase.

def gym_to_ioh(gym_problem, name="GymProblem", maximize=True):
    # wrap GymProblem so it returns a single scalar for IOH
    def wrapped(x):
        # gym_problem.__call__ should return a single scalar (returns)
        val = gym_problem(x)
        # if gym_problem returns (returns, rewards), adapt:
        if isinstance(val, tuple):
            returns = val[0]
        else:
            returns = val
        return float(returns)
    ioh_problem = ioh.wrap_problem(
        function=wrapped,
        name=name,
        problem_class=ioh.ProblemClass.REAL,
        dimension=gym_problem.n_variables,
        instance=1,
        optimization_type=ioh.OptimizationType.MAX if maximize else ioh.OptimizationType.MIN,
        lb=-1.0,
        ub=1.0,
    )
    return ioh_problem

def run(algos, gym_problem, repeats=5, root="IOH_Data", store_positions=True):
    ioh_problem = gym_to_ioh(gym_problem, name="LunarPolicy", maximize=True)
    for alg_name, alg_callable in algos.items():
        folder = f"{alg_name}_LunarPolicy"
        logger = ioh.logger.Analyzer(root=root, folder_name=folder,
                                     algorithm_name=alg_name,
                                     algorithm_info=str(getattr(alg_callable, "__dict__", {})),
                                     store_positions=store_positions)
        ioh_problem.attach_logger(logger)
        for rep in range(1, repeats + 1):
            # ensure problem reset between runs
            print(f"Running {alg_name} run {rep}/{repeats}")
            # many of your algos are callables that accept a problem
            alg_callable(ioh_problem)
            ioh_problem.reset()
        ioh_problem.detach_logger()
        logger.close()
    print(f"IOH logs written to: {root}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--root", default="IOH_Data")
    args = p.parse_args()

    # TODO: replace these placeholders with your actual algorithm callables and GymProblem instance
    from problem import GymProblem
    gp = GymProblem()  # adjust kwargs if needed

    # Build the algorithm mapping; replace with your real functions/classes
    algos = {
        "adaptive_de": lambda prob: print("replace with adaptive_de(prob)"),
        "combo": lambda prob: print("replace with combo(prob)"),
        "plain": lambda prob: print("replace with plain(prob)"),
        "random": lambda prob: print("replace with random(prob)"),
        "self": lambda prob: print("replace with self_algo(prob)"),
    }

    # Edit algos mapping to call your real optimisers before running
    run(algos, gp, repeats=args.repeats, root=args.root)