"""
Simple 1+1 Evolution Strategy implementation for the LunarLander GymProblem.

Usage:
    python one_plus_one_es.py

This implements a minimal (1+1)-ES with isotropic Gaussian mutation and
an optional 1/5th rule step-size adaptation.

It follows the evaluation pattern used in `random_search.py` and uses
`GymProblem` from `problem.py`.
"""

import numpy as np
from problem import GymProblem
import matplotlib.pyplot as plt


def one_plus_one_es(problem: GymProblem, budget=5000, sigma=0.1, adapt=True,
                    seed: int | None = None, print_every: int = 0,
                    mirrored: bool = False, reevaluate_k: int = 1):
    """
    Basic (1+1)-ES with several small, optional improvements:

    - seed: optional RNG seed for reproducibility
    - clipping: ensures parameters remain in [-1,1] (project convention)
    - mirrored: if True, uses mirrored sampling (z and -z) to reduce variance
    - reevaluate_k: number of evaluations to average when assessing a candidate (k>=1)

    Returns: best_x, best_f, f_hist, best_rewards, sigma_end, accept_rate
    """
    n = problem.n_variables

    # set RNG seed if provided for reproducible experiments
    if seed is not None:
        np.random.seed(seed)

    x = problem.sample()

    # initial evaluation (optionally averaged)
    if reevaluate_k <= 1:
        f, rewards = problem(x)
    else:
        vals = [problem(x)[0] for _ in range(reevaluate_k)]
        f = float(np.mean(vals))
        _, rewards = problem(x)

    best_x = x.copy()
    best_f = f
    # keep the initial rewards as the current best rewards in case no improvement happens
    best_rewards = rewards
    f_hist = [f]

    success_count = 0
    total_accepted = 0

    for i in range(1, budget):
        # sample a direction
        z = np.random.randn(n)

        # if mirrored sampling is enabled, evaluate both +z and -z and pick the better
        if mirrored:
            y1 = np.clip(x + sigma * z, -1.0, 1.0)
            y2 = np.clip(x - sigma * z, -1.0, 1.0)

            # evaluate candidates (optionally averaged)
            if reevaluate_k <= 1:
                f1, r1 = problem(y1)
                f2, r2 = problem(y2)
            else:
                f1 = float(np.mean([problem(y1)[0] for _ in range(reevaluate_k)]))
                f2 = float(np.mean([problem(y2)[0] for _ in range(reevaluate_k)]))

            if f1 >= f2:
                fy, r = f1, r1
                y = y1
            else:
                fy, r = f2, r2
                y = y2

        else:
            # standard single-sided mutation with clipping
            y = np.clip(x + sigma * z, -1.0, 1.0)
            if reevaluate_k <= 1:
                fy, r = problem(y)
            else:
                vals = [problem(y)[0] for _ in range(reevaluate_k)]
                fy = float(np.mean(vals))

        # selection: replace parent if the child is better
        if fy > f:
            x = y
            f = fy
            success_count += 1
            total_accepted += 1
            if f > best_f:
                best_f = f
                best_x = x.copy()
                best_rewards = r

        f_hist.append(f)

        # print progress only every `print_every` iterations (0 = no printing)
        if print_every and i % print_every == 0:
            print(f"Iter {i}: current f = {f:.3f}, best = {best_f:.3f}, sigma = {sigma:.4f}")

        # Simple 1/5th rule every 20 evaluations when adapt=True
        if adapt and i % 20 == 0:
            rate = success_count / 20.0
            if rate > 0.2:
                sigma *= 1.2
            else:
                sigma *= 0.85
            success_count = 0

    sigma_end = sigma
    accept_rate = total_accepted / float(budget)

    return best_x, best_f, f_hist, best_rewards, sigma_end, accept_rate


def main():
    problem = GymProblem()
    # one_plus_one_es now returns additional metadata (sigma_end, accept_rate)
    best_x, best_f, f_hist, best_rewards, sigma_end, accept_rate = one_plus_one_es(problem, budget=5000)
    print(f"Best fitness: {best_f}")
    print(f"Final sigma: {sigma_end}, accept_rate: {accept_rate:.3f}")
    problem.show(best_x)

    plt.subplot(2, 1, 1)
    plt.plot(f_hist)
    plt.title("Fitness History")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")

    plt.subplot(2, 1, 2)
    plt.plot(best_rewards)
    plt.title("Best solution rewards")
    plt.xlabel("Simulation step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
