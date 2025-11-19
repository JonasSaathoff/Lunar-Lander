import numpy as np
from problem import GymProblem

# Define the adaptation rate constants (tau values)
# These control the probability of re-sampling F and CR
TAU1 = 0.1
TAU2 = 0.1

def adaptive_de_current_to_best(
        problem: GymProblem,
        budget: int = 1000,
        pop_size: int = 30,
        F: float = 0.8,
        CR: float = 0.9,
        seed: int | None = None,
        print_every: int = 0):
    """
    Adaptive DE/current-to-best/1/bin (jDE-inspired).

    Returns:
        best_x, best_f, f_hist, sigma_hist_placeholder, best_rewards
        (compatible with run_replicates.py)
    """
    n = problem.n_variables
    if seed is not None:
        np.random.seed(seed)

    # 1. Initialize population and individual control parameters
    pop = np.stack([problem.sample() for _ in range(pop_size)])
    F_pop = np.full(pop_size, F)
    CR_pop = np.full(pop_size, CR)

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
            x_i = pop[i]

            # 2. Adaptive Parameter Generation (F' and CR')
            F_trial = F_pop[i]
            if np.random.rand() < TAU1:
                F_trial = 0.1 + np.random.rand() * 0.9

            CR_trial = CR_pop[i]
            if np.random.rand() < TAU2:
                CR_trial = np.random.rand()

            # 3. current-to-best/1 mutation
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2 = np.random.choice(idxs, size=2, replace=False)
            xr1, xr2 = pop[r1], pop[r2]
            mutant = x_i + F_trial * (best_x - x_i) + F_trial * (xr1 - xr2)

            # 4. binomial crossover
            cross = np.random.rand(n) < CR_trial
            jrand = np.random.randint(n)
            cross[jrand] = True
            trial = np.where(cross, mutant, x_i)
            trial = np.clip(trial, -1.0, 1.0)

            # 5. evaluate trial
            f_trial, rewards_trial = problem(trial)
            evals += 1

            # 6. selection + self-adaptation
            if f_trial > fitness[i]:
                pop[i] = trial
                fitness[i] = float(f_trial)
                rewards_list[i] = rewards_trial
                F_pop[i] = F_trial
                CR_pop[i] = CR_trial

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
