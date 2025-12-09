'''
Requirements: 
    pip install "gymnasium[box2d]" numpy matplotlib
'''

from problem import GymProblem
import numpy as np
import matplotlib.pyplot as plt


def random_search(problem: GymProblem, budget: int = 1000, seed: int | None = None, print_every: int = 0):
    """Random search that is callable from run_replicates.

    Returns (best_x, best_f, f_hist, best_rewards)
    """
    if seed is not None:
        np.random.seed(seed)

    f_hist = []
    f_prime = -np.inf
    x_prime = None
    best_rewards = None

    for i in range(budget):
        x = problem.sample()  # generate a random solution

        # use headless evaluation (problem(x)) which returns (fitness, rewards)
        try:
            f, rewards = problem(x)
        except Exception:
            # fallback to play_episode if callable API differs
            f, rewards = problem.play_episode(x)

        if f > f_prime:  # if improvement, save the best solution found
            f_prime = f
            x_prime = x.copy()
            best_rewards = rewards

        f_hist.append(f)
        if print_every and i % print_every == 0:
            print(f"Gen {i}: Best f = {f_prime:.3f}")

    return x_prime, f_prime, f_hist, best_rewards


def main():
    # generate a problem instance with default simulation parameters (no wind, no turbulences and default moon gravity)
    problem = GymProblem()

    budget = 1000

    # call the callable random_search for the interactive demo
    best_x, best_f, f_hist, best_rewards = random_search(problem, budget=budget, seed=None, print_every=1)

    print(f"Best fitness: {best_f}")
    # use shared visualizer for consistent output
    try:
        from viz import plot_and_show
        plot_and_show(problem, best_x, f_hist, best_rewards, title='Random search', out_plot=None, render=True)
    except Exception:
        # fallback to previous behaviour
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


if __name__ == "__main__":
    main()
