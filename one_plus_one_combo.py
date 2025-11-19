"""
Combined (1+1)-ES variant: self-adaptive sigma + mirrored sampling + re-evaluation + occasional sigma boost.

Features:
- self-adaptation: sigma' = sigma * exp(tau * N(0,1)) with tau = 1/sqrt(n)
- mirrored sampling: evaluate +z and -z, pick the better child
- reevaluate_k: average k evaluations when comparing candidates to reduce noise
- restart_no_improve: if no improvement for this many iterations, boost sigma


Usage:
    python one_plus_one_combo.py --budget 1000 --reseed 42

"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from problem import GymProblem


def one_plus_one_combo(problem: GymProblem, budget: int = 1000, sigma0: float = 0.1,
                       seed: int | None = None, print_every: int = 0,
                       mirrored: bool = True, reevaluate_k: int = 3,
                       restart_no_improve: int = 200):
    """Run the combined ES and return results.

    Returns:
      best_x, best_f, f_hist, sigma_hist, best_rewards, sigma_end, accept_rate
    """
    n = problem.n_variables
    if seed is not None:
        np.random.seed(seed)

    # initialize
    x = problem.sample()

    # initial evaluation (average if requested)
    if reevaluate_k <= 1:
        f, rewards = problem(x)
    else:
        vals = [problem(x)[0] for _ in range(reevaluate_k)]
        f = float(np.mean(vals))
        _, rewards = problem(x)

    best_x = x.copy()
    best_f = f
    best_rewards = rewards

    sigma = float(sigma0)
    tau = 1.0 / np.sqrt(n)

    f_hist = [f]
    sigma_hist = [sigma]

    accept_count = 0
    no_improve = 0

    for i in range(1, budget):
        # self-adapt sigma
        s_prime = max(1e-12, sigma * np.exp(tau * np.random.randn()))

        # sample direction and evaluate mirrored or single child
        z = np.random.randn(n)

        if mirrored:
            y1 = np.clip(x + s_prime * z, -1.0, 1.0)
            y2 = np.clip(x - s_prime * z, -1.0, 1.0)

            if reevaluate_k <= 1:
                f1, _ = problem(y1)
                f2, _ = problem(y2)
            else:
                f1 = float(np.mean([problem(y1)[0] for _ in range(reevaluate_k)]))
                f2 = float(np.mean([problem(y2)[0] for _ in range(reevaluate_k)]))

            if f1 >= f2:
                fy = f1
                y = y1
            else:
                fy = f2
                y = y2

        else:
            y = np.clip(x + s_prime * z, -1.0, 1.0)
            if reevaluate_k <= 1:
                fy, _ = problem(y)
            else:
                fy = float(np.mean([problem(y)[0] for _ in range(reevaluate_k)]))

        # selection
        if fy > f:
            x = y
            f = fy
            sigma = s_prime
            accept_count += 1
            no_improve = 0
            # get rewards for the accepted candidate (single rollout)
            try:
                _, rewards = problem(x)
            except Exception:
                rewards = best_rewards
            if f > best_f:
                best_f = f
                best_x = x.copy()
                best_rewards = rewards
        else:
            no_improve += 1

        f_hist.append(f)
        sigma_hist.append(sigma)

        # occasional sigma boost to escape plateaus
        if restart_no_improve and no_improve >= restart_no_improve:
            sigma *= 3.0
            no_improve = 0
            # small random perturbation to x to escape local optimum
            x = np.clip(x + 0.1 * np.random.randn(n), -1.0, 1.0)

        if print_every and i % print_every == 0:
            print(f"Iter {i}: current f = {f:.3f}, best = {best_f:.3f}, sigma = {sigma:.5f}")

    sigma_end = sigma
    accept_rate = accept_count / float(budget)

    return best_x, best_f, f_hist, sigma_hist, best_rewards, sigma_end, accept_rate


def compute_landed(problem: GymProblem, x) -> bool:
    """Resimulate `x` on a fresh environment created from GymProblem and return landed boolean."""
    tmp = GymProblem()
    env = tmp.env_spec.make(**tmp.simulation_params)
    M = x.reshape(tmp.state_size, tmp.n_outputs)
    observation, *_ = env.reset()
    obs_history = [observation]
    for _ in range(tmp.env_spec.max_episode_steps):
        action = tmp.activation(M.T @ observation)
        observation, reward, terminated, truncated, info = env.step(action)
        obs_history.append(observation)
        if terminated or truncated:
            break
    env.close()
    final = obs_history[-1]
    left = bool(final[6])
    right = bool(final[7])
    vel_x = final[2]
    vel_y = final[3]
    angle = final[4]
    return left and right and abs(vel_y) < 0.5 and abs(vel_x) < 0.3 and abs(angle) < 0.2


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=int, default=1000)
    p.add_argument('--sigma0', type=float, default=0.1)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--print-every', type=int, default=50)
    p.add_argument('--reeval', type=int, default=3)
    p.add_argument('--restart-no-improve', type=int, default=200)
    p.add_argument('--out-plot', type=str, default=None)
    args = p.parse_args()

    problem = GymProblem()
    best_x, best_f, f_hist, sigma_hist, best_rewards, sigma_end, accept_rate = one_plus_one_combo(
        problem,
        budget=args.budget,
        sigma0=args.sigma0,
        seed=args.seed,
        print_every=args.print_every,
        mirrored=True,
        reevaluate_k=args.reeval,
        restart_no_improve=args.restart_no_improve,
    )

    print(f"Best fitness: {best_f}")
    print(f"Final sigma: {sigma_end}, accept_rate: {accept_rate:.3f}")
    landed = compute_landed(problem, best_x)
    print(f"Landed (resimulated): {landed}")
    # use shared visualizer for consistent output (it will also render if desired)
    try:
        from viz import plot_and_show
        # show both fitness and sigma as two figures: combine sigma into the rewards plot by plotting sigma_hist on top
        plot_and_show(problem, best_x, f_hist, best_rewards, title='Combo ES', out_plot=args.out_plot, render=True)
    except Exception:
        # fallback to previous behavior
        plt.subplot(3, 1, 1)
        plt.plot(f_hist)
        plt.title('Fitness History')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')

        plt.subplot(3, 1, 2)
        plt.plot(sigma_hist)
        plt.title('Sigma History')
        plt.xlabel('Iteration')
        plt.ylabel('Sigma')

        plt.subplot(3, 1, 3)
        plt.plot(best_rewards)
        plt.title('Best solution rewards')
        plt.xlabel('Simulation step')
        plt.ylabel('Reward')

        plt.tight_layout()
        if args.out_plot:
            plt.savefig(args.out_plot)
            print(f"Saved plot to {args.out_plot}")
        else:
            plt.show()


if __name__ == '__main__':
    main()
