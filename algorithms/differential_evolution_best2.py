"""
Differential Evolution Best 2 (DE/best/2/bin) implementation for the GymProblem.

Mutation strategy:
  v = best + F*((xb - xc) + (xd - xe))

API mirrors other DE variants in this repo.
"""

import numpy as np
from problem import GymProblem


def differential_evolution_best2(problem: GymProblem, budget: int = 1000, pop_size: int = 20,
                           F: float = 0.8, CR: float = 0.9, seed: int | None = None,
                           print_every: int = 0):
    """Run DE/best/2/bin and return best solution and history."""
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
            # pick four distinct indices b,c,d,e != i
            idxs = [idx for idx in range(pop_size) if idx != i]
            b, c, d, e = np.random.choice(idxs, size=4, replace=False)
            xb, xc, xd, xe = pop[b], pop[c], pop[d], pop[e]

            # mutation: best/2
            mutant = best_x + F * ((xb - xc) + (xd - xe))
            # crossover (binomial)
            cross = np.random.rand(n) < CR
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
    args = p.parse_args()

    problem = GymProblem()
    best_x, best_f, f_hist, _, best_rewards = differential_evolution_best2(
        problem,
        budget=args.budget,
        pop_size=args.pop_size,
        F=args.F,
        CR=args.CR,
        seed=args.seed,
        print_every=args.print_every,
    )

    print(f"Best fitness: {best_f}")
    plt.plot(f_hist)
    plt.title('DE (best/2) best fitness history')
    plt.xlabel('Evaluation index')
    plt.ylabel('Best fitness')
    if args.out_plot:
        plt.savefig(args.out_plot)
        print(f"Saved plot to {args.out_plot}")
    else:
        plt.show()
