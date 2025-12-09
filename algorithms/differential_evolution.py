"""
Simple Differential Evolution (DE/rand/1/bin) implementation for the GymProblem.

API:
  differential_evolution(problem, budget=1000, pop_size=20, F=0.8, CR=0.9, seed=None, print_every=0)

Return value (compatible with run_replicates.py normalization):
  (best_x, best_f, f_hist, sigma_hist_placeholder, best_rewards)

Notes:
- Solutions are numpy arrays in [-1,1] shaped (n_variables,)
- Budget counts fitness evaluations (one evaluation per trial candidate accepted or not)
"""

import numpy as np
from problem import GymProblem


def differential_evolution(problem: GymProblem, budget: int = 1000, pop_size: int = 20,
                           F: float = 0.8, CR: float = 0.9, seed: int | None = None,
                           print_every: int = 0):
    """Run DE/rand/1/bin and return best solution and history."""
    n = problem.n_variables
    if seed is not None:
        np.random.seed(seed)

    # initialize population uniformly in [-1,1]
    pop = np.stack([problem.sample() for _ in range(pop_size)])
    fitness = np.zeros(pop_size, dtype=float)
    rewards_list = [None] * pop_size

    # initial evaluation
    evals = 0
    for i in range(pop_size):
        f, rewards = problem(pop[i])
        fitness[i] = float(f)
        rewards_list[i] = rewards
        evals += 1
        if evals >= budget:
            break

    best_idx = int(np.argmax(fitness))
    best_x = pop[best_idx].copy()
    best_f = float(fitness[best_idx])
    best_rewards = rewards_list[best_idx]

    f_hist = [best_f]

    # evolution loop
    gen = 0
    while evals < budget:
        gen += 1
        for i in range(pop_size):
            # pick three distinct indices a,b,c != i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(idxs, size=3, replace=False)
            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation
            mutant = xa + F * (xb - xc)
            # crossover (binomial)
            cross = np.random.rand(n) < CR
            # ensure at least one component from mutant
            jrand = np.random.randint(n)
            cross[jrand] = True
            trial = np.where(cross, mutant, pop[i])
            trial = np.clip(trial, -1.0, 1.0)

            # evaluate trial
            f_trial, rewards_trial = problem(trial)
            evals += 1

            # selection
            if f_trial > fitness[i]:
                pop[i] = trial
                fitness[i] = float(f_trial)
                rewards_list[i] = rewards_trial
                # update global best
                if f_trial > best_f:
                    best_f = float(f_trial)
                    best_x = trial.copy()
                    best_rewards = rewards_trial

            f_hist.append(best_f)
            if evals >= budget:
                break

        if print_every and gen % print_every == 0:
            print(f"DE gen {gen}: best_f = {best_f:.3f}, evals={evals}/{budget}")

    # return 5-tuple compatible with run_replicates mapping
    sigma_hist_placeholder = None
    return best_x, best_f, f_hist, sigma_hist_placeholder, best_rewards


def main():
    import argparse
    import matplotlib.pyplot as plt

    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=int, default=1000)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--F', type=float, default=0.8)
    p.add_argument('--CR', type=float, default=0.9)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--print-every', type=int, default=0)
    p.add_argument('--out-plot', type=str, default=None)
    p.add_argument('--render', action='store_true', help='Resimulate and render the best policy in a window')
    args = p.parse_args()

    problem = GymProblem()
    best_x, best_f, f_hist, _, best_rewards = differential_evolution(
        problem,
        budget=args.budget,
        pop_size=args.pop_size,
        F=args.F,
        CR=args.CR,
        seed=args.seed,
        print_every=args.print_every,
    )

    print(f"Best fitness: {best_f}")
    # resimulate to check landing (same heuristic used elsewhere)
    try:
        tmp = GymProblem()
        env = tmp.env_spec.make(**tmp.simulation_params)
        M = best_x.reshape(tmp.state_size, tmp.n_outputs)
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
        landed = left and right and abs(vel_y) < 0.5 and abs(vel_x) < 0.3 and abs(angle) < 0.2
        print(f"Landed (resimulated): {landed}")
    except Exception:
        print("Landed (resimulated): unknown (resimulation failed)")
    
    # render the best policy interactively if requested
    if args.render:
        try:
            print('Rendering best policy (close the window to continue)...')
            problem.show(best_x)
        except Exception:
            print('Rendering failed: ensure you have a graphical environment available')
    
    # use shared visualizer for consistent plots
    try:
        from viz import plot_and_show
        plot_and_show(problem, best_x, f_hist, best_rewards, title='DE result', out_plot=args.out_plot, render=args.render)
    except Exception:
        pass

    # quick plot of best fitness history
    plt.plot(f_hist)
    plt.title('DE best fitness history')
    plt.xlabel('Evaluation index')
    plt.ylabel('Best fitness')
    if args.out_plot:
        plt.savefig(args.out_plot)
        print(f"Saved plot to {args.out_plot}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
