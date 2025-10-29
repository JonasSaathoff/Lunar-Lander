"""
Run multiple replicates of a chosen ES algorithm and report mean/std of best fitness.

Usage examples:
    python run_replicates.py --algo self --reps 10 --budget 200

Algorithms supported:
 - self : one_plus_one_self_adaptive.one_plus_one_self_adaptive
 - plain: one_plus_one_es.one_plus_one_es

Outputs a CSV `replicates_results.csv` with columns: seed,best_f
"""

import argparse
import csv
import numpy as np
import time


def run(args):
    # Prefer adapters if available (non-invasive wrappers that standardize call signature)
    alg = None
    try:
        import adapters
        if hasattr(adapters, args.algo):
            alg = getattr(adapters, args.algo)
    except Exception:
        alg = None

    # fallback to importing implementations directly if adapter is not present
    if alg is None:
        if args.algo == 'self':
            from one_plus_one_self_adaptive import one_plus_one_self_adaptive as alg
        elif args.algo == 'plain':
            from one_plus_one_es import one_plus_one_es as alg
        elif args.algo == 'combo':
            from one_plus_one_combo import one_plus_one_combo as alg
        elif args.algo == 'de':
            from differential_evolution import differential_evolution as alg
        elif args.algo == 'adaptive_de':
            from adaptive_differential_evolution import adaptive_differential_evolution as alg
        elif args.algo == 'de2':
            from differential_evolution_rand2 import differential_evolution_rand2 as alg
        elif args.algo == 'random':
            from random_search import random_search as alg
        else:
            raise ValueError('Unknown algorithm: ' + args.algo)

    results = []

    for i in range(args.reps):
        seed = args.seed_start + i
        print(f"=== Run {i+1}/{args.reps}, seed={seed} ===")
        start = time.time()
        # call the selected algorithm and normalize its return signature
        Problem = __import__('problem').GymProblem
        problem = Problem()

        if args.algo == 'self':
            res = alg(
                problem=problem,
                budget=args.budget,
                sigma0=args.sigma0,
                seed=seed,
                print_every=args.print_every,
            )
        elif args.algo == 'plain':
            res = alg(
                problem,
                budget=args.budget,
                sigma=args.sigma0,
                adapt=args.adapt if hasattr(args, 'adapt') else True,
            )
        elif args.algo == 'combo':  # combo
            res = alg(
                problem,
                budget=args.budget,
                sigma0=args.sigma0,
                seed=seed,
                print_every=args.print_every,
                mirrored=True,
                reevaluate_k=args.reeval if hasattr(args, 'reeval') else 3,
                restart_no_improve=args.restart_no_improve if hasattr(args, 'restart_no_improve') else 200,
            )
        elif args.algo == 'de':
            res = alg(
                problem,
                budget=args.budget,
                pop_size=args.pop_size if hasattr(args, 'pop_size') else 30,
                F=args.F if hasattr(args, 'F') else 0.8,
                CR=args.CR if hasattr(args, 'CR') else 0.9,
                seed=seed,
                print_every=args.print_every,
            )
        elif args.algo == 'adaptive_de':   
            res = alg(
                problem=problem,
                budget=args.budget,
                pop_size=args.pop_size if hasattr(args, 'pop_size') else 30,
                # Note: ADE typically ignores the fixed F and CR, but you can pass them as initial M_F/M_CR
                seed=seed,
                print_every=args.print_every,
            )
        elif args.algo == 'de2':
            res = alg(
                problem,
                budget=args.budget,
                pop_size=args.pop_size if hasattr(args, 'pop_size') else 30,
                F=args.F if hasattr(args, 'F') else 0.8,
                CR=args.CR if hasattr(args, 'CR') else 0.9,
                seed=seed,
                print_every=args.print_every,
            )
        elif args.algo == 'random':
            # random_search returns (best_x, best_f, f_hist, best_rewards)
            res = alg(problem, budget=args.budget, seed=seed, print_every=args.print_every)
        else:
            raise RuntimeError('Unknown algorithm selection at run-time')

        # normalize returned tuple formats between algorithms
        sigma_end = None
        accept_rate = None
        sigma_hist = None
        best_rewards = None

        if isinstance(res, tuple):
            if len(res) == 7:
                best_x, best_f, f_hist, sigma_hist, best_rewards, sigma_end, accept_rate = res
            elif len(res) == 6:
                best_x, best_f, f_hist, sigma_hist, best_rewards, sigma_end = res
            elif len(res) == 5:
                best_x, best_f, f_hist, sigma_hist, best_rewards = res
            elif len(res) == 4:
                best_x, best_f, f_hist, best_rewards = res
            else:
                # fallback: try to unpack first four
                best_x, best_f, f_hist, best_rewards = res[:4]
        else:
            raise RuntimeError('Algorithm did not return expected tuple')

        elapsed = time.time() - start
        print(f"seed={seed} best_f={best_f:.6f} (took {elapsed:.1f}s)")

        # determine landed boolean by re-simulating the best_x on a fresh environment
        # We avoid changing problem.py by creating the env from problem.env_spec and simulation_params
        landed = None
        try:
            Problem = __import__('problem').GymProblem
            tmp_problem = Problem()
            env = tmp_problem.env_spec.make(**tmp_problem.simulation_params)
            M = best_x.reshape(tmp_problem.state_size, tmp_problem.n_outputs)
            observation, *_ = env.reset()
            obs_history = [observation]
            returns_sim = 0.0
            for _ in range(tmp_problem.env_spec.max_episode_steps):
                action = tmp_problem.activation(M.T @ observation)
                observation, reward, terminated, truncated, info = env.step(action)
                returns_sim += reward
                obs_history.append(observation)
                if terminated or truncated:
                    break
            env.close()
            final_obs = obs_history[-1]
            # observation layout: [pos_x, pos_y, vel_x, vel_y, angle, ang_vel, left_contact, right_contact]
            left_contact = bool(final_obs[6])
            right_contact = bool(final_obs[7])
            vel_x = final_obs[2]
            vel_y = final_obs[3]
            angle = final_obs[4]
            landed = left_contact and right_contact and abs(vel_y) < 0.5 and abs(vel_x) < 0.3 and abs(angle) < 0.2
        except Exception:
            landed = None

        results.append({'seed': seed, 'best_f': float(best_f), 'sigma_end': sigma_end, 'accept_rate': accept_rate, 'landed': landed})

    # write CSV
    csv_path = args.out or 'replicates_results.csv'
    # include metadata fields if available
    fieldnames = ['seed', 'best_f', 'sigma_end', 'accept_rate', 'landed']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            # ensure all expected keys exist to avoid writer errors
            row = {k: r.get(k, '') for k in fieldnames}
            writer.writerow(row)

    bests = np.array([r['best_f'] for r in results])
    print('\nSummary:')
    print(f'  mean(best_f) = {bests.mean():.6f}')
    print(f'  std(best_f)  = {bests.std(ddof=1):.6f}')
    print(f'  min = {bests.min():.6f}, max = {bests.max():.6f}')
    print(f'  results saved to {csv_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--algo', choices=['self', 'plain', 'combo', 'de', 'adaptive_de', 'de2', 'random'], default='self', help='which ES to run')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--budget', type=int, default=200)
    p.add_argument('--sigma0', type=float, default=0.1)
    p.add_argument('--seed-start', type=int, default=0)
    p.add_argument('--print-every', type=int, default=0)
    p.add_argument('--reeval', type=int, default=3)
    p.add_argument('--restart-no-improve', type=int, default=200)
    p.add_argument('--pop-size', type=int, default=30)
    p.add_argument('--F', type=float, default=0.8)
    p.add_argument('--CR', type=float, default=0.9)
    p.add_argument('--out', type=str, default='replicates_results.csv')
    p.add_argument('--adapt', type=bool, default=True)
    args = p.parse_args()
    run(args)
