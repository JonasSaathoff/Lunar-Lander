import numpy as np
from problem import GymProblem

# Define the adaptation rate constants (tau values)
# These control the probability of re-sampling F and CR
TAU1 = 0.1
TAU2 = 0.1

def adaptive_differential_evolution(problem: GymProblem, budget: int = 1000, pop_size: int = 30,
                           F: float = 0.8, CR: float = 0.9, seed: int | None = None,
                           print_every: int = 0):
    """Run Adaptive DE/rand/1/bin (jDE-inspired) and return best solution and history."""
    n = problem.n_variables
    if seed is not None:
        np.random.seed(seed)

    # 1. Initialize population and individual control parameters
    pop = np.stack([problem.sample() for _ in range(pop_size)])
    # F and CR arrays store the parameter values for each individual
    F_pop = np.full(pop_size, F)
    CR_pop = np.full(pop_size, CR)

    fitness = np.zeros(pop_size, dtype=float)
    rewards_list = [None] * pop_size

    # initial evaluation (unchanged)
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
            # 2. Adaptive Parameter Generation (F' and CR')
            # Generate temporary F' (F_trial) for the current mutation
            F_trial = F_pop[i]
            if np.random.rand() < TAU1:
                # Re-sample F from U[0.1, 1.0]
                F_trial = 0.1 + np.random.rand() * 0.9 
            
            # Generate temporary CR' (CR_trial) for the current crossover
            CR_trial = CR_pop[i]
            if np.random.rand() < TAU2:
                # Re-sample CR from U[0.0, 1.0]
                CR_trial = np.random.rand()

            # pick three distinct indices a,b,c != i
            idxs = [idx for idx in range(pop_size) if idx != i]
            a, b, c = np.random.choice(idxs, size=3, replace=False)
            xa, xb, xc = pop[a], pop[b], pop[c]

            # mutation: use F_trial
            mutant = xa + F_trial * (xb - xc)
            
            # crossover (binomial): use CR_trial
            cross = np.random.rand(n) < CR_trial
            jrand = np.random.randint(n)
            cross[jrand] = True
            trial = np.where(cross, mutant, pop[i])
            trial = np.clip(trial, -1.0, 1.0)

            # evaluate trial
            f_trial, rewards_trial = problem(trial)
            evals += 1

            # selection
            if f_trial > fitness[i]:
                # Successful trial: replace individual, its fitness, its rewards,
                # AND its control parameters (self-adaptation)
                pop[i] = trial
                fitness[i] = float(f_trial)
                rewards_list[i] = rewards_trial
                F_pop[i] = F_trial   # <--- Keep the successful F
                CR_pop[i] = CR_trial # <--- Keep the successful CR
                
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