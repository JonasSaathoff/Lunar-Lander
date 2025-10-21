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


def one_plus_one_es(problem: GymProblem, budget=1000, sigma=0.1, adapt=True):
    n = problem.n_variables
    x = problem.sample()
    f, rewards = problem(x)
    best_x = x.copy()
    best_f = f
    # keep the initial rewards as the current best rewards in case no improvement happens
    best_rewards = rewards
    f_hist = [f]
    success_count = 0

    for i in range(1, budget):
        z = np.random.randn(n)
        y = x + sigma * z
        fy, r = problem(y)
        if fy > f:
            x = y
            f = fy
            success_count += 1
            if f > best_f:
                best_f = f
                best_x = x.copy()
                best_rewards = r
        f_hist.append(f)
        # print progress each iteration so the user sees the per-iteration fitness
        print(f"Iter {i}: current f = {f:.3f}, best = {best_f:.3f}, sigma = {sigma:.4f}")

        # Simple 1/5th rule every 20 evaluations
        if adapt and i % 20 == 0:
            rate = success_count / 20.0
            if rate > 0.2:
                sigma *= 1.2
            else:
                sigma *= 0.85
            success_count = 0

    return best_x, best_f, f_hist, best_rewards


def main():
    problem = GymProblem()
    best_x, best_f, f_hist, best_rewards = one_plus_one_es(problem, budget=200)
    print(f"Best fitness: {best_f}")
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
