import matplotlib.pyplot as plt
from problem import GymProblem


def plot_and_show(problem: GymProblem, best_x, f_hist, best_rewards, title: str = 'Result', out_plot: str | None = None, render: bool = True):
    """Plot fitness history and best rewards, and show the best policy with problem.show().

    This provides a consistent output across different algorithm mains.
    """
    print(f"{title}: best fitness = {f_hist[-1] if f_hist else 'N/A'}")

    # show the policy visually (may open a window)
    if render:
        try:
            problem.show(best_x)
        except Exception as e:
            print('Rendering failed:', e)

    # plots
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    if f_hist:
        plt.plot(f_hist)
    plt.title(f'{title} — Fitness History')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')

    plt.subplot(2, 1, 2)
    if best_rewards is not None:
        plt.plot(best_rewards)
    plt.title(f'{title} — Best solution rewards')
    plt.xlabel('Simulation step')
    plt.ylabel('Reward')

    plt.tight_layout()
    if out_plot:
        plt.savefig(out_plot)
        print(f'Saved plot to {out_plot}')
    else:
        plt.show()
