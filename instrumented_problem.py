import csv
import pathlib
import time
from typing import Optional

from problem import GymProblem


class InstrumentedProblem:
    """Proxy wrapper around GymProblem that logs per-evaluation best-so-far to CSV.

    Intended usage:
      p = InstrumentedProblem(GymProblem(), ioh_out='ioh_runs', algorithm='de', seed=123)
      algo(p, ...)

    The wrapper exposes the same public methods as GymProblem: sample(), play_episode(),
    __call__(), show(). Each call to the problem (evaluation) appends a row to
    ioh_out/<algorithm>/seed_<seed>.csv with columns: evaluation,best_f

    The wrapper assumes the objective is to maximize (higher reward is better).
    """

    def __init__(self, inner: GymProblem, ioh_out: str = "ioh_runs", algorithm: str = "alg", seed: Optional[int] = None, problem_name: Optional[str] = None):
        self.inner = inner
        self.ioh_out = pathlib.Path(ioh_out)
        self.algorithm = algorithm
        self.seed = seed if seed is not None else int(time.time())
        self.problem_name = problem_name or getattr(inner, "env_spec", None) and getattr(inner.env_spec, "id", None) or "problem"

        # tracking
        self.evaluation = 0
        self.best_f = float("-inf")

        # output file
        algo_dir = self.ioh_out / self.algorithm
        algo_dir.mkdir(parents=True, exist_ok=True)
        self.outfile = algo_dir / f"seed_{self.seed}.csv"

        # write header if file doesn't exist
        if not self.outfile.exists():
            with self.outfile.open("w", newline="") as fh:
                writer = csv.writer(fh)
                writer.writerow(["evaluation", "best_f"])

    # pass-through helpers
    def sample(self):
        return self.inner.sample()

    def play_episode(self, x, **env_kwargs):
        # delegate to inner
        returns, rewards = self.inner.play_episode(x, **env_kwargs)
        # record evaluation
        self._record(returns)
        return returns, rewards

    def __call__(self, x):
        returns, rewards = self.inner(x)
        self._record(returns)
        return returns, rewards

    def show(self, x):
        returns, rewards = self.inner.show(x)
        self._record(returns)
        return returns, rewards

    def _record(self, value: float):
        # maximize
        self.evaluation += 1
        if value > self.best_f:
            self.best_f = value
        # append to CSV
        with self.outfile.open("a", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow([self.evaluation, self.best_f])

    # expose inner attributes if needed
    def __getattr__(self, name):
        return getattr(self.inner, name)
