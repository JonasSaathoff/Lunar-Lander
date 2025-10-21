"""
Self-adaptive (1+1)-Evolution Strategy for LunarLander.

The individual is [x, sigma]. Sigma is mutated with a log-normal update:
    sigma' = sigma * exp(tau * N(0,1))
Then x' = clip(x + sigma' * z, -1, 1) with z ~ N(0, I)

Selection: replace parent with child if child's fitness is better.

Usage:
    python one_plus_one_self_adaptive.py
"""

import numpy as np
from problem import GymProblem
import matplotlib.pyplot as plt


def one_plus_one_self_adaptive(problem: GymProblem, budget=1000, sigma0=0.1, seed=None, print_every=20):
    n = problem.n_variables
    if seed is not None:
        np.random.seed(seed)

    # initialize
    x = problem.sample()
    sigma = float(sigma0)
    f, rewards = problem(x)
    best_x = x.copy()
    best_f = f
    best_rewards = rewards

    f_hist = [f]
    sigma_hist = [sigma]

    tau = 1.0 / np.sqrt(n)

    for i in range(1, budget):
        # mutate sigma (log-normal)
        s_prime = sigma * np.exp(tau * np.random.randn())
        s_prime = max(s_prime, 1e-12)

        # mutate x using the new sigma
        z = np.random.randn(n)
        y = np.clip(x + s_prime * z, -1.0, 1.0)

        fy, r = problem(y)

        # selection
        if fy > f:
            x = y
            f = fy
            sigma = s_prime
            if f > best_f:
                best_f = f
                best_x = x.copy()
                best_rewards = r

        f_hist.append(f)
        sigma_hist.append(sigma)

        if i % print_every == 0:
            print(f"Iter {i}: current f = {f:.3f}, best = {best_f:.3f}, sigma = {sigma:.5f}")

    return best_x, best_f, f_hist, sigma_hist, best_rewards


def main():
    problem = GymProblem()
    best_x, best_f, f_hist, sigma_hist, best_rewards = one_plus_one_self_adaptive(problem, budget=200, sigma0=0.1, seed=42, print_every=20)
    print(f"Best fitness: {best_f}")
    problem.show(best_x)

    plt.subplot(3, 1, 1)
    plt.plot(f_hist)
    plt.title("Fitness History")
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")

    plt.subplot(3, 1, 2)
    plt.plot(sigma_hist)
    plt.title("Sigma History")
    plt.xlabel("Iteration")
    plt.ylabel("Sigma")

    plt.subplot(3, 1, 3)
    plt.plot(best_rewards)
    plt.title("Best solution rewards")
    plt.xlabel("Simulation step")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
