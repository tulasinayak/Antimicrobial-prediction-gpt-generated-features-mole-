"""Task 8a: compare rows under MolE's full original protocol, repeated over trials.

MolE's protocol (rolayoalarcon/mole_antimicrobial_potential, workflow/02 and 03):
  - random search: 5 configurations drawn with replacement from
      n_estimators {30,100,300,500,1000} x max_depth {5,10,50,100}
      x eta {0.3,0.1,0.05,1} x subsample {0.3,0.5,0.8,1.0}
  - each configuration fitted 5 times; selection on the mean validation PR-AUC
  - raw features (NaN -> 0), no scaling, no early stopping, no class weighting
  - pooled PR-AUC = sklearn auc(recall, precision)

Here each trial k draws its 5 configurations with random.Random(k), shared by every row
in that trial (so rows are paired), and reports the selected configuration's mean
validation and test PR-AUC over its fits. Configurations with subsample = 1.0 are
deterministic, so they are fitted once and the result counted for all 5 fits.

Every fit is appended to <out>/fits.csv, so an interrupted run resumes where it stopped.

Usage:
    python -I scripts/run_mole_protocol_search.py --device cuda                   # Kaggle GPU
    python -I scripts/run_mole_protocol_search.py --rows mole_only --trials 0 --smoke   # quick CPU test
    python -I scripts/run_mole_protocol_search.py --aggregate_only
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reproduce_mole_original import (SF3, SPLIT_TSV, get_split, load_block,  # noqa: E402
                                     load_labels, metrics)

SPACE = {"n_estimators": [30, 100, 300, 500, 1000], "max_depth": [5, 10, 50, 100],
         "eta": [0.3, 0.1, 0.05, 1], "subsample": [0.3, 0.5, 0.8, 1.0]}
ROWS = {
    "mole_only": "mole",
    "mole_plus_gpt-5.2": "mole,gpt:gpt-5.2",
    "gpt_only_gpt-5.2": "gpt:gpt-5.2",
    "rdkit_only": "chemdesc",
    "mole_plus_rdkit": "mole,chemdesc",
}
DEFAULT_ROWS = "mole_only,mole_plus_gpt-5.2,gpt_only_gpt-5.2"
N_CONFIGS, N_FITS = 5, 5


def configs_for(trial: int) -> list[dict]:
    """MolE's select_params: independent random.choice per key, drawn with replacement."""
    rng = random.Random(trial)
    return [{k: rng.choice(v) for k, v in SPACE.items()} for _ in range(N_CONFIGS)]


def fit_one(data, cfg: dict, seed: int, device: str, nthread: int) -> dict:
    X_tr, y_tr, _ = data["train"]
    model = XGBClassifier(n_estimators=cfg["n_estimators"], max_depth=cfg["max_depth"],
                          learning_rate=cfg["eta"], subsample=cfg["subsample"],
                          objective="binary:logistic", tree_method="hist", device=device,
                          base_score=0.5, random_state=seed, n_jobs=nthread)
    model.fit(X_tr, y_tr)
    out = {}
    for name in ("valid", "test"):
        X, y, _ = data[name]
        m = metrics(y, model.predict_proba(X)[:, 1])
        out[f"{name}_prauc"] = m["prauc_trapz"]
        out[f"{name}_ap"] = m["average_precision"]
    return out


def run(args) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    fits_path = os.path.join(args.output_dir, "fits.csv")
    done = pd.read_csv(fits_path) if os.path.exists(fits_path) else pd.DataFrame()
    have = set(map(tuple, done[["row", "trial", "config", "fit"]].values)) if len(done) else set()

    labels = load_labels(SF3)
    split = pd.read_csv(SPLIT_TSV, sep="\t", index_col="prestwick_ID")["split"]
    trials = [int(t) for t in args.trials.split(",")]

    for row in args.rows.split(","):
        blocks = ROWS[row].split(",")
        features = pd.concat([load_block(b) for b in blocks], axis=1, join="outer")
        data = get_split(features, labels, split)
        cache: dict[tuple, dict] = {}
        for k in trials:
            for i, cfg in enumerate(configs_for(k)):
                if args.smoke:
                    cfg = {**cfg, "n_estimators": 10, "max_depth": min(cfg["max_depth"], 5)}
                deterministic = cfg["subsample"] == 1.0
                for f in range(N_FITS):
                    if (row, k, i, f) in have:
                        continue
                    key = (tuple(sorted(cfg.items())),)
                    if deterministic and key in cache:
                        res, minutes = cache[key], 0.0
                    else:
                        started = time.time()
                        res = fit_one(data, cfg, seed=1000 * k + 10 * i + f, device=args.device,
                                      nthread=args.nthread)
                        minutes = (time.time() - started) / 60
                        if deterministic:
                            cache[key] = res
                    rec = {"row": row, "trial": k, "config": i, "fit": f, **cfg,
                           "deterministic": deterministic, **res, "fit_minutes": round(minutes, 2)}
                    pd.DataFrame([rec]).to_csv(fits_path, mode="a", index=False,
                                               header=not os.path.exists(fits_path))
                    have.add((row, k, i, f))
                    print(f"{row} trial {k} config {i} fit {f}: val {res['valid_prauc']:.4f} "
                          f"test {res['test_prauc']:.4f} ({minutes:.1f} min) {cfg}", flush=True)


def aggregate(args) -> None:
    fits = pd.read_csv(os.path.join(args.output_dir, "fits.csv"))
    per_cfg = fits.groupby(["row", "trial", "config"]).agg(
        val=("valid_prauc", "mean"), test=("test_prauc", "mean"), n=("fit", "size"),
        n_estimators=("n_estimators", "first"), max_depth=("max_depth", "first"),
        eta=("eta", "first"), subsample=("subsample", "first")).reset_index()
    complete = per_cfg[per_cfg.n == N_FITS]
    # Selection: best mean validation PR-AUC among the trial's 5 configurations.
    best = (complete.sort_values(["row", "trial", "val"], ascending=[True, True, False])
            .groupby(["row", "trial"]).head(1)
            .merge(complete.groupby(["row", "trial"]).size().rename("n_configs").reset_index()))
    best = best[best.n_configs == N_CONFIGS]
    best.to_csv(os.path.join(args.output_dir, "selected_per_trial.csv"), index=False)

    rows = []
    for row, g in best.groupby("row", sort=False):
        rows.append({"row": row, "n_trials": len(g),
                     "val_mean": g.val.mean(), "val_sd": g.val.std(ddof=1),
                     "test_mean": g.test.mean(), "test_sd": g.test.std(ddof=1),
                     "test_min": g.test.min(), "test_max": g.test.max()})
    table = pd.DataFrame(rows)
    wide = best.pivot(index="trial", columns="row", values="test")
    if "mole_only" in wide:
        for r in wide.columns.drop("mole_only"):
            pair = wide[["mole_only", r]].dropna()
            if len(pair) >= 2:
                d = pair[r] - pair["mole_only"]
                t = stats.ttest_rel(pair[r], pair["mole_only"])
                table.loc[table.row == r, "diff_vs_mole"] = d.mean()
                table.loc[table.row == r, "diff_sd"] = d.std(ddof=1)
                table.loc[table.row == r, "wins_vs_mole"] = f"{int((d > 0).sum())}/{len(d)}"
                table.loc[table.row == r, "p_paired_t"] = t.pvalue
    table.to_csv(os.path.join(args.output_dir, "table_mole_protocol.csv"), index=False)
    pd.set_option("display.width", 220)
    print(table.round(4).to_string(index=False))
    print("\nSelected configuration per trial:")
    print(best[["row", "trial", "config", "n_estimators", "max_depth", "eta", "subsample", "val", "test"]]
          .round(4).to_string(index=False))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", default=DEFAULT_ROWS, help=f"Comma-separated, from: {', '.join(ROWS)}")
    ap.add_argument("--trials", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--nthread", type=int, default=os.cpu_count())
    ap.add_argument("--output_dir", default="results/mole_protocol_trials")
    ap.add_argument("--smoke", action="store_true",
                    help="Tiny models (10 trees, depth <= 5) to test the pipeline; writes real files.")
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        run(args)
    aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
