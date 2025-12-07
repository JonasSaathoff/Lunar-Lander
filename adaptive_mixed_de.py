"""
Adaptive mixed-strategy Differential Evolution for LunarLander.

This variant extends a jDE-like self-adaptive DE by selecting a mutation
strategy per trial from a small pool (rand/1, rand/2, rand-to-best/2).
Strategy selection probabilities are adapted online using an exponential
moving-average of recent success rates (a lightweight SaDE-style scheme).

Designed as a drop-in replacement for `adaptive_differential_evolution`.
"""
import numpy as np
from problem import GymProblem
import os, csv


def adaptive_mixed_de(problem: GymProblem, budget: int = 1000, pop_size: int = 30,
                      F: float = 0.8, CR: float = 0.9, seed: int | None = None,
                      print_every: int = 0,
                      strategy_pool=None, strategy_probs=None,
                      adapt_strategy: bool = True, alpha: float = 0.05,
                      algo_name: str = "adaptive_mixed_de", ioh_out: str | None = None):
    """Run an adaptive mixed-strategy DE.

    Parameters:
    - strategy_pool: list of strategy keys (supported: 'rand1','rand2','rand_to_best2')
    - strategy_probs: initial probabilities (if None -> uniform)
    - adapt_strategy: whether to adapt strategy probabilities online
    - alpha: EMA rate for updating strategy success estimates (0..1)

    Returns: (best_x, best_f, f_hist, sigma_hist_placeholder, best_rewards)
    """
    n = problem.n_variables
    if seed is not None:
        np.random.seed(seed)

    if strategy_pool is None:
        strategy_pool = ['rand1', 'rand2', 'rand_to_best2']
    K = len(strategy_pool)
    if strategy_probs is None:
        strategy_probs = [1.0 / K] * K

    # map strategy to index
    strat_to_idx = {s: i for i, s in enumerate(strategy_pool)}

    # initialize strategy success EMA (small epsilon to avoid zeros)
    strategy_score = np.array([1e-3] * K, dtype=float)

    # jDE-like self-adaptation arrays
    pop = np.stack([problem.sample() for _ in range(pop_size)])
    F_pop = np.full(pop_size, F)
    CR_pop = np.full(pop_size, CR)

    fitness = np.zeros(pop_size, dtype=float)
    rewards_list = [None] * pop_size

    evals = 0
    # initial evaluation
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

    gen = 0
    while evals < budget:
        gen += 1
        # per-generation temporary success counts (for this generation)
        gen_success = np.zeros(K, dtype=float)

        for i in range(pop_size):
            # adapt F and CR per individual (jDE style)
            if np.random.rand() < 0.1:
                F_trial = 0.1 + np.random.rand() * 0.9
            else:
                F_trial = F_pop[i]
            if np.random.rand() < 0.1:
                CR_trial = np.random.rand()
            else:
                CR_trial = CR_pop[i]

            # pick strategy according to current probabilities
            probs = np.array(strategy_probs, dtype=float)
            # if adapting, derive probs from strategy_score (soft-normalized)
            if adapt_strategy:
                p = strategy_score + 1e-6
                probs = p / p.sum()
            probs = probs / probs.sum()
            strat = np.random.choice(strategy_pool, p=probs)
            si = strat_to_idx[strat]

            # prepare indices excluding i
            idxs = [idx for idx in range(pop_size) if idx != i]

            # compute mutant according to strategy
            if strat == 'rand1':
                a, b, c = np.random.choice(idxs, size=3, replace=False)
                xa, xb, xc = pop[a], pop[b], pop[c]
                mutant = xa + F_trial * (xb - xc)
            elif strat == 'rand2':
                # rand/2: combine two difference vectors
                a, b, c, d, e = np.random.choice(idxs, size=5, replace=False)
                xa, xb, xc, xd, xe = pop[a], pop[b], pop[c], pop[d], pop[e]
                mutant = xa + F_trial * (xb - xc) + F_trial * (xd - xe)
            elif strat == 'rand_to_best2':
                # rand-to-best/2: base at best plus two difference vectors
                a, b, c, d = np.random.choice(idxs, size=4, replace=False)
                xb, xc, xd, xe = pop[a], pop[b], pop[c], pop[d]
                mutant = best_x + F_trial * (xb - xc) + F_trial * (xd - xe)
            else:
                # fallback to rand1
                a, b, c = np.random.choice(idxs, size=3, replace=False)
                xa, xb, xc = pop[a], pop[b], pop[c]
                mutant = xa + F_trial * (xb - xc)

            # crossover (binomial)
            cross = np.random.rand(n) < CR_trial
            jrand = np.random.randint(n)
            cross[jrand] = True
            trial = np.where(cross, mutant, pop[i])
            trial = np.clip(trial, -1.0, 1.0)

            # evaluate
            f_trial, rewards_trial = problem(trial)
            evals += 1

            # selection
            if f_trial > fitness[i]:
                pop[i] = trial
                fitness[i] = float(f_trial)
                rewards_list[i] = rewards_trial
                F_pop[i] = F_trial
                CR_pop[i] = CR_trial
                gen_success[si] += 1.0
                # update global best
                if f_trial > best_f:
                    best_f = float(f_trial)
                    best_x = trial.copy()
                    best_rewards = rewards_trial

            f_hist.append(best_f)
            if evals >= budget:
                break

        # update strategy_score via EMA of success rate in this generation
        if adapt_strategy:
            # normalized successes per strategy (divide by pop_size to get rate)
            gen_rate = gen_success / max(1.0, float(pop_size))
            strategy_score = (1.0 - alpha) * strategy_score + alpha * gen_rate

        if print_every and gen % print_every == 0:
            print(f"Mixed-DE gen {gen}: best_f = {best_f:.3f}, evals={evals}/{budget}")

    sigma_hist_placeholder = None

    # optional IOH export
    if ioh_out is not None:
        os.makedirs(ioh_out, exist_ok=True)
        algo_dir = os.path.join(ioh_out, algo_name)
        os.makedirs(algo_dir, exist_ok=True)
        seed_label = str(seed) if seed is not None else "na"
        fn = os.path.join(algo_dir, f'seed_{seed_label}.csv')
        with open(fn, 'w', newline='') as of:
            w = csv.writer(of)
            w.writerow(['evaluation', 'best_f'])
            for i, val in enumerate(f_hist, start=1):
                w.writerow([i, float(val)])
        if print_every:
            print(f"Wrote IOH CSV to {fn}")

    return best_x, best_f, f_hist, sigma_hist_placeholder, best_rewards


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--budget', type=int, default=1000)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--print-every', type=int, default=0)
    args = p.parse_args()

    problem = GymProblem()
    best_x, best_f, f_hist, _, best_rewards = adaptive_mixed_de(
        problem,
        budget=args.budget,
        pop_size=args.pop_size,
        seed=args.seed,
        print_every=args.print_every,
    )
    print(f"Best fitness: {best_f}")


if __name__ == '__main__':
    main()
