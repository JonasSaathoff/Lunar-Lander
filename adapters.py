"""Non-invasive adapters that expose a canonical callable API for each algorithm.

Each function uses the original implementation (without modifying original files)
and returns the algorithm's result in a form compatible with run_replicates.

Functions provided: plain, self, combo, de, random
"""
from typing import Any


def plain(problem, budget=1000, sigma0=0.1, adapt=True, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for one_plus_one_es (plain ES)."""
    from one_plus_one_es import one_plus_one_es
    return one_plus_one_es(problem, budget=budget, sigma=sigma0, adapt=adapt)


def self(problem, budget=1000, sigma0=0.1, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for one_plus_one_self_adaptive."""
    from one_plus_one_self_adaptive import one_plus_one_self_adaptive
    return one_plus_one_self_adaptive(problem=problem, budget=budget, sigma0=sigma0, seed=seed, print_every=print_every)


def combo(problem, budget=1000, sigma0=0.1, seed=None, print_every=0, mirrored=True, reevaluate_k=3, restart_no_improve=200, **kwargs) -> Any:
    """Adapter for one_plus_one_combo."""
    from one_plus_one_combo import one_plus_one_combo
    return one_plus_one_combo(problem, budget=budget, sigma0=sigma0, seed=seed, print_every=print_every, mirrored=mirrored, reevaluate_k=reevaluate_k, restart_no_improve=restart_no_improve)


def de(problem, budget=1000, pop_size=30, F=0.8, CR=0.9, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for differential_evolution."""
    from differential_evolution import differential_evolution
    return differential_evolution(problem, budget=budget, pop_size=pop_size, F=F, CR=CR, seed=seed, print_every=print_every)


def random(problem, budget=1000, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for random_search demo (non-invasive)."""
    from random_search import random_search
    return random_search(problem, budget=budget, seed=seed, print_every=print_every)


def rand2(problem, budget=1000, pop_size=30, F=0.8, CR=0.9, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for differential_evolution_rand2 (DE/rand/2/bin)."""
    from differential_evolution_rand2 import differential_evolution_rand2
    return differential_evolution_rand2(problem, budget=budget, pop_size=pop_size, F=F, CR=CR, seed=seed, print_every=print_every)


def rand_to_best2(problem, budget=1000, pop_size=30, F=0.8, CR=0.9, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for differential_evolution_rand_to_best2 (DE/rand-to-best/2/bin)."""
    from differential_evolution_rand_to_best2 import differential_evolution_rand_to_best2
    return differential_evolution_rand_to_best2(problem, budget=budget, pop_size=pop_size, F=F, CR=CR, seed=seed, print_every=print_every)


def best1(problem, budget=1000, pop_size=30, F=0.8, CR=0.9, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for differential_evolution_best1 (DE/best/1/bin)."""
    from differential_evolution_best1 import differential_evolution_best1
    return differential_evolution_best1(problem, budget=budget, pop_size=pop_size, F=F, CR=CR, seed=seed, print_every=print_every)


def best2(problem, budget=1000, pop_size=30, F=0.8, CR=0.9, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for differential_evolution_best2 (DE/best/2/bin)."""
    from differential_evolution_best2 import differential_evolution_best2
    return differential_evolution_best2(problem, budget=budget, pop_size=pop_size, F=F, CR=CR, seed=seed, print_every=print_every)


def current_to_best1(problem, budget=1000, pop_size=30, F=0.8, CR=0.9, seed=None, print_every=0, **kwargs) -> Any:
    """Adapter for differential_evolution_current_to_best1 (DE/current-to-best/1/bin)."""
    from differential_evolution_current_to_best1 import differential_evolution_current_to_best1
    return differential_evolution_current_to_best1(problem, budget=budget, pop_size=pop_size, F=F, CR=CR, seed=seed, print_every=print_every)
