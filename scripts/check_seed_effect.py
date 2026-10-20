"""Confirm that the training seed now changes the fitted model.

train_xgb accepted a random_state that never reached XGBoost, so every "seed"
refit the identical model. This trains a few short boosters directly and reports
whether the predictions differ, both with and without subsampling.

Short boosters (num_boost_round=60) are enough: if the seed has any effect it
shows up in the first rounds.
"""

from __future__ import annotations

import sys

import numpy as np
import xgboost as xgb

sys.path.insert(0, "notebooks")

RNG = np.random.default_rng(0)
N, D = 4000, 60


def fake_data() -> tuple[xgb.DMatrix, xgb.DMatrix]:
    X = RNG.normal(size=(N, D))
    w = RNG.normal(size=D)
    p = 1 / (1 + np.exp(-(X @ w) / np.sqrt(D)))
    y = (RNG.uniform(size=N) < p).astype(int)
    cut = N // 2
    return (xgb.DMatrix(X[:cut], label=y[:cut]),
            xgb.DMatrix(X[cut:], label=y[cut:]))


def train(dtr: xgb.DMatrix, dte: xgb.DMatrix, seed: int, subsample: float) -> np.ndarray:
    params = {
        "tree_method": "hist", "max_bin": 512,
        "objective": "binary:logistic", "eval_metric": "aucpr",
        "verbosity": 0, "eta": 0.1, "max_depth": 5,
        "subsample": subsample, "colsample_bytree": subsample,
        "seed": seed,
    }
    booster = xgb.train(params, dtr, num_boost_round=60)
    return booster.predict(dte)


def main() -> None:
    dtr, dte = fake_data()
    for subsample in (1.0, 0.8, 0.5):
        a = train(dtr, dte, seed=0, subsample=subsample)
        b = train(dtr, dte, seed=1, subsample=subsample)
        same = np.allclose(a, b)
        delta = float(np.abs(a - b).max())
        verdict = "IDENTICAL (seed has no effect)" if same else "differs"
        print(f"subsample=colsample={subsample}: {verdict}, max|delta| = {delta:.3e}")

    print("\nExpected: subsample=1.0 identical (hist is deterministic without "
          "subsampling); 0.8 and 0.5 differ.")


if __name__ == "__main__":
    main()
