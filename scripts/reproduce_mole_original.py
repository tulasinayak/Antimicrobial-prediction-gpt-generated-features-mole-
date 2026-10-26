"""Reproduce MolE's published test PR-AUC (0.7797) with MolE's own training protocol.

MolE (Olayo-Alarcon et al., Nat. Commun. 2025) selected its antimicrobial model by an
unseeded random search, 5 configurations x 5 fits, over

    n_estimators {30,100,300,500,1000}, max_depth {5,10,50,100},
    eta {0.3,0.1,0.05,1}, subsample {0.3,0.5,0.8,1.0}

(rolayoalarcon/mole_antimicrobial_potential, workflow/02.model_training.ipynb). Their
committed data/02.model_training/strain_performance.tsv.gz shows the winner was

    n_estimators=1000, max_depth=100, eta=0.1, subsample=1.0
    -> validation PR-AUC 0.70097, test PR-AUC 0.77976

This script trains exactly that configuration the way MolE did:
  - raw 1000-d MolE representation, NaN -> 0, no scaling, no feature dropping
  - one-hot strain encoding appended, long-format compound x strain rows in MolE's order
  - XGBClassifier with fixed n_estimators: no early stopping, no scale_pos_weight
  - PR-AUC as sklearn auc(recall, precision) on pooled pairs (average precision also shown)

MolE ran xgboost 1.6.2, whose default tree_method for a dataset this size is "exact" and
whose default base_score is 0.5. xgboost >= 2 defaults to "hist" and estimates base_score
from the labels, so both are set explicitly here; --tree_method hist shows the effect of
the newer default.

Inputs are this repository's copies, verified identical to MolE's published files
(features max |diff| 0, labels 0 mismatches, split 1197/1197).

Usage:
    python -I scripts/reproduce_mole_original.py
    python -I scripts/reproduce_mole_original.py --tree_method hist
    python -I scripts/reproduce_mole_original.py --compare_predictions <complete_test_predictions.tsv.gz>
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import xgboost
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier

RAW = "data/raw_data"
MOLE_TSV = f"{RAW}/maier_mole_representation_with_id.tsv.gz"
CHEMDESC_TSV = f"{RAW}/maier_chemdesc_representation.tsv"
GPT_DIR = "data/gpt_generation_without_guidance"
SPLIT_TSV = f"{RAW}/maier_scaffold_split.tsv"
SF3 = f"{RAW}/screen_results_info_SF3.xlsx"

# MolE's selected configuration (model_3 in their strain_performance.tsv.gz).
MOLE_BEST = {"n_estimators": 1000, "max_depth": 100, "eta": 0.1, "subsample": 1.0,
             "objective": "binary:logistic"}
PUBLISHED = {"validation_prauc": 0.7009655554019908, "test_prauc": 0.7797609717204966}


def load_labels(path: str, cutoff: float = 0.05) -> pd.DataFrame:
    """Compound x strain binary matrix, built as MolE's 01.prepare_training_data does."""
    screen = pd.read_excel(path, sheet_name="S3a. Adjusted p-values")
    screen = screen.drop(columns=["chemical_name", "drug_class", "n_hit"]).set_index("prestwick_ID")
    # MolE uses <=, and NaN p-values compare False, so they become negatives.
    return (screen <= cutoff).astype(int)


def get_split(features: pd.DataFrame, labels: pd.DataFrame, split: pd.Series):
    """Port of MolE's get_split: same row order (strain-major) and column order."""
    strains = list(labels.columns)
    # OneHotEncoder orders categories alphabetically; reproduce that column order.
    taxa_ohe = pd.DataFrame(
        (np.array(strains)[:, None] == np.array(sorted(strains))[None, :]).astype(float),
        index=strains, columns=[f"strain_{s}" for s in sorted(strains)])

    melt = labels.unstack().reset_index().rename(columns={0: "label", "level_0": "taxa_name"})
    features = features.copy()
    features.columns = [str(c) for c in features.columns]
    features = features.fillna(0)

    joined = melt.join(features, on="prestwick_ID").join(taxa_ohe, on="taxa_name")
    assert joined.shape[0] == melt.shape[0]

    out = {}
    for name in ("train", "valid", "test"):
        ids = split.index[split == name]
        part = joined.loc[joined["prestwick_ID"].isin(ids)]
        out[name] = (part.drop(columns=["prestwick_ID", "label", "taxa_name"]),
                     part["label"].to_numpy(), part[["prestwick_ID", "taxa_name"]])
    return out


def load_block(name: str) -> pd.DataFrame:
    """One feature block indexed by prestwick_ID: mole, chemdesc, or gpt:<model>."""
    if name == "mole":
        df = pd.read_csv(MOLE_TSV, sep="\t").set_index("prestwick_ID")
        return df[[str(i) for i in range(1000)]].add_prefix("mole_")
    if name == "chemdesc":
        # The same 98 descriptors as MolE's maier_chemdesc_representation.tsv.gz.
        df = pd.read_csv(CHEMDESC_TSV, sep="\t").set_index("prestwick_ID")
        return df[~df.index.duplicated()]
    if name.startswith("gpt:"):
        model = name.split(":", 1)[1]
        hits = [f for f in os.listdir(GPT_DIR) if f.endswith(f"_{model}_freeform.csv")]
        if len(hits) != 1:
            raise ValueError(f"expected one free-form file for {model} in {GPT_DIR}, found {hits}")
        df = pd.read_csv(os.path.join(GPT_DIR, hits[0])).set_index("prestwick_ID")
        # Numeric descriptors only; metadata such as ProductName or input is text.
        return df.select_dtypes(include="number").add_prefix("gpt_")
    raise ValueError(f"unknown feature block {name!r}; use mole, chemdesc or gpt:<model>")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    precision, recall, _ = precision_recall_curve(y, p)
    return {"prauc_trapz": float(auc(recall, precision)),
            "average_precision": float(average_precision_score(y, p)),
            "roc_auc": float(roc_auc_score(y, p))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="mole",
                    help="Comma-separated feature blocks joined on prestwick_ID: mole, chemdesc, "
                         "gpt:<model> (e.g. gpt:gpt-5.2). Default mole reproduces the paper.")
    ap.add_argument("--tree_method", default="exact", choices=["exact", "hist", "approx"],
                    help="exact matches xgboost 1.6.2's default for this data size.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="cuda trains on the GPU; it supports hist only, not exact.")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Do nothing if this run's summary JSON already exists.")
    ap.add_argument("--seed", type=int, default=1234,
                    help="MolE's final refit used random_state=1234 (irrelevant at subsample=1).")
    ap.add_argument("--nthread", type=int, default=os.cpu_count())
    ap.add_argument("--compare_predictions", default=None,
                    help="MolE's complete_test_predictions.tsv.gz, to correlate per-pair probabilities.")
    ap.add_argument("--output_dir", default="results/mole_original_protocol")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    blocks = [b.strip() for b in args.features.split(",") if b.strip()]
    if args.device == "cuda" and args.tree_method != "hist":
        ap.error("--device cuda needs --tree_method hist (xgboost has no GPU exact method)")
    device = "" if args.device == "cpu" else f"_{args.device}"
    tag = (f"{'+'.join(b.replace(':', '-') for b in blocks)}_{args.tree_method}{device}"
           f"_xgb{xgboost.__version__}")
    summary_path = os.path.join(args.output_dir, f"summary_{tag}.json")
    if args.skip_existing and os.path.exists(summary_path):
        print(f"{summary_path} exists, skipping")
        return 0

    # Left-join on the MolE/screen compound order; missing values become 0 in get_split,
    # as in MolE's pipeline.
    features = pd.concat([load_block(b) for b in blocks], axis=1, join="outer")
    labels = load_labels(SF3)
    split = pd.read_csv(SPLIT_TSV, sep="\t", index_col="prestwick_ID")["split"]
    data = get_split(features, labels, split)
    X_tr, y_tr, _ = data["train"]
    print(f"train {X_tr.shape}, valid {data['valid'][0].shape}, test {data['test'][0].shape}; "
          f"xgboost {xgboost.__version__}, tree_method={args.tree_method}", flush=True)

    model = XGBClassifier(**MOLE_BEST, tree_method=args.tree_method, base_score=0.5,
                          device=args.device, random_state=args.seed, n_jobs=args.nthread)
    started = time.time()
    model.fit(X_tr, y_tr)
    minutes = (time.time() - started) / 60
    print(f"fit in {minutes:.1f} min", flush=True)

    result = {"features": blocks, "n_features": int(X_tr.shape[1]),
              "xgboost": xgboost.__version__, "tree_method": args.tree_method,
              "device": args.device,
              "params": MOLE_BEST, "fit_minutes": minutes, "published_mole_only": PUBLISHED}
    for name in ("valid", "test"):
        X, y, keys = data[name]
        p = model.predict_proba(X)[:, 1]
        result[name] = metrics(y, p)
        print(f"{name:5s} PR-AUC (auc) {result[name]['prauc_trapz']:.4f}  "
              f"AP {result[name]['average_precision']:.4f}  "
              f"ROC-AUC {result[name]['roc_auc']:.4f}", flush=True)
        if name == "test":
            preds = keys.assign(y_true=y, proba=p)
            preds.to_csv(os.path.join(args.output_dir, f"test_predictions_{tag}.csv"), index=False)

    if blocks == ["mole"]:
        print(f"published: valid {PUBLISHED['validation_prauc']:.4f}  test {PUBLISHED['test_prauc']:.4f}")

    if args.compare_predictions:
        theirs = pd.read_csv(args.compare_predictions, sep="\t")
        theirs = theirs.loc[theirs["representation"] == "MolE",
                            ["prestwick_ID", "strain", "1"]].rename(columns={"1": "proba_mole"})
        merged = preds.merge(theirs, left_on=["prestwick_ID", "taxa_name"],
                             right_on=["prestwick_ID", "strain"])
        result["vs_published_predictions"] = {
            "n_pairs": int(len(merged)),
            "pearson": float(np.corrcoef(merged["proba"], merged["proba_mole"])[0, 1]),
            "max_abs_diff": float((merged["proba"] - merged["proba_mole"]).abs().max()),
        }
        print("vs published predictions:", result["vs_published_predictions"])

    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
