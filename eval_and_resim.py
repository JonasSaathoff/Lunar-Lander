#!/usr/bin/env python3
"""Run algorithms, collect best_f and f_hist, resimulate best solutions.

Usage:
  python eval_and_resim.py --algos plain combo rand2 --reps 3 --budget 200 --resim 5 --out results/test_eval

The script saves per-replicate CSVs and a summary with statistics and simple tests.
"""
import argparse
import os
import time
import csv
import json
from statistics import mean, median

import numpy as np
import matplotlib.pyplot as plt

import adapters
from problem import GymProblem


def run_one_rep(algo_name, seed, budget):
    problem = GymProblem()
    # special-case adaptive_de which lives in its own module
    if algo_name == 'adaptive_de':
        from adaptive_differential_evolution import adaptive_differential_evolution as func
        res = func(problem, budget=budget, seed=seed, print_every=0)
    else:
        # use adapter to run algorithm; adapters return algorithm-specific tuple
        func = getattr(adapters, algo_name)
        res = func(problem, budget=budget, seed=seed, print_every=0)
    # many algos return (best_x, best_f, f_hist, ...)
    if isinstance(res, tuple) or isinstance(res, list):
        best_x = res[0]
        best_f = float(res[1])
        f_hist = list(res[2]) if len(res) > 2 and res[2] is not None else []
    else:
        # fallback if adapter returns single numeric
        best_x = None
        best_f = float(res)
        f_hist = []
    return best_x, best_f, f_hist


def resimulate_best(best_x, M):
    # resimulate best_x M times and return mean reward
    if best_x is None:
        return float('nan'), []
    rewards = []
    for _ in range(M):
        p = GymProblem()
        val = p.play_episode(best_x)
        if isinstance(val, (list, tuple)):
            rewards.append(float(val[0]))
        else:
            rewards.append(float(val))
    return float(np.mean(rewards)), rewards


def summarize_and_plot(results, out_dir):
    # results: list of dicts {algo, seed, best_f, resim_mean, resim_rewards}
    algos = sorted(set(r['algo'] for r in results))
    summary = {}
    for algo in algos:
        vals = [r['best_f'] for r in results if r['algo'] == algo]
        resim_means = [r['resim_mean'] for r in results if r['algo'] == algo]
        summary[algo] = {
            'n': len(vals),
            'mean': float(np.mean(vals)) if vals else float('nan'),
            'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'median': float(median(vals)) if vals else float('nan'),
            'iqr': float(np.percentile(vals, 75) - np.percentile(vals, 25)) if vals else float('nan'),
            'resim_mean': float(np.mean(resim_means)) if resim_means else float('nan'),
            'success_rate': float(np.mean([1 if rm >= 200 else 0 for rm in resim_means])) if resim_means else 0.0,
        }
    # write JSON summary
    with open(os.path.join(out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)

    # boxplot of best_f
    plt.figure(figsize=(8, 5))
    data = [ [r['best_f'] for r in results if r['algo'] == algo] for algo in algos ]
    plt.boxplot(data, labels=algos, showmeans=True)
    plt.title('Best fitness per algorithm')
    plt.ylabel('best_f')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'best_f_boxplot.png'))

    # ECDF plot
    plt.figure(figsize=(8, 5))
    for algo in algos:
        arr = np.sort([r['best_f'] for r in results if r['algo'] == algo])
        if len(arr) == 0:
            continue
        y = np.arange(1, len(arr)+1) / float(len(arr))
        plt.step(arr, y, where='post', label=algo)
    plt.xlabel('best_f')
    plt.ylabel('ECDF')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'best_f_ecdf.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algos', nargs='+', default=['plain', 'combo', 'rand2'])
    parser.add_argument('--reps', type=int, default=3)
    parser.add_argument('--budget', type=int, default=200)
    parser.add_argument('--resim', type=int, default=5)
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--out', type=str, default='results/eval_' + time.strftime('%Y%m%d_%H%M%S'))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, 'results.csv')
    results = []

    for algo in args.algos:
        for i in range(args.reps):
            seed = args.seed_start + i
            print(f'Running {algo} rep {i+1}/{args.reps} seed={seed} budget={args.budget}')
            try:
                best_x, best_f, f_hist = run_one_rep(algo, seed, args.budget)
            except Exception as e:
                print('  run failed:', e)
                best_x, best_f, f_hist = None, float('nan'), []

            resim_mean, resim_rewards = resimulate_best(best_x, args.resim)

            rec = {
                'algo': algo,
                'seed': seed,
                'best_f': float(best_f),
                'f_hist': f_hist,
                'resim_mean': float(resim_mean),
                'resim_rewards': resim_rewards,
            }
            results.append(rec)

            # append to CSV
            write_header = not os.path.exists(csv_path)
            with open(csv_path, 'a', newline='') as cf:
                writer = csv.writer(cf)
                if write_header:
                    writer.writerow(['algo', 'seed', 'best_f', 'resim_mean'])
                writer.writerow([algo, seed, rec['best_f'], rec['resim_mean']])

    summarize_and_plot(results, args.out)
    print('Done. Results in', args.out)


if __name__ == '__main__':
    main()
