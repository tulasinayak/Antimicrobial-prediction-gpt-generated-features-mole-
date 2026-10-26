"""Train every Table 1 row with MolE's own recipe, then summarise with bootstrap CIs.

Each row is fitted by scripts/reproduce_mole_original.py with MolE's selected configuration
(n_estimators=1000, max_depth=100, eta=0.1, subsample=1.0; raw features, no early stopping,
no class weighting). That configuration is deterministic, so seeds give no spread; the
uncertainty here comes from resampling the 120 test compounds instead (all 40 strains of
a compound move together). Every row is scored on the same resamples, so differences
against MolE alone are paired.

Usage:
    python -I scripts/run_mole_method.py --device cuda --tree_method hist     # Kaggle GPU
    python -I scripts/run_mole_method.py --tree_method exact                  # CPU, MolE-exact
    python -I scripts/run_mole_method.py --tree_method hist --device cuda --aggregate_only
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROWS = [
    ("mole_only", "mole"),
    ("rdkit_only", "chemdesc"),
    ("mole_plus_rdkit", "mole,chemdesc"),
    ("mole_plus_gpt-4o", "mole,gpt:gpt-4o"),
    ("mole_plus_gpt-4o-mini", "mole,gpt:gpt-4o-mini"),
    ("mole_plus_gpt-4.1", "mole,gpt:gpt-4.1"),
    ("mole_plus_gpt-4.1-mini", "mole,gpt:gpt-4.1-mini"),
    ("mole_plus_gpt-5.1", "mole,gpt:gpt-5.1"),
    ("mole_plus_gpt-5.2", "mole,gpt:gpt-5.2"),
]


def find(out_dir: str, prefix: str, features: str, tree_method: str, device: str) -> str | None:
    tag = f"{features.replace(',', '+').replace(':', '-')}_{tree_method}"
    tag += "" if device == "cpu" else f"_{device}"
    hits = glob.glob(os.path.join(out_dir, f"{prefix}_{tag}_xgb*"))
    return hits[0] if hits else None


def bootstrap(preds: dict[str, pd.DataFrame], n_boot: int, seed: int) -> pd.DataFrame:
    """Compound-level bootstrap of pooled test AP for every row, on shared resamples."""
    base = next(iter(preds.values()))
    compounds = base["prestwick_ID"].unique()
    rng = np.random.default_rng(seed)
    draws = [rng.choice(compounds, size=len(compounds), replace=True) for _ in range(n_boot)]

    scores = {}
    for row, df in preds.items():
        y, p = df["y_true"].to_numpy(), df["proba"].to_numpy()
        positions = df.groupby("prestwick_ID").indices
        scores[row] = np.array([
            average_precision_score(y[idx], p[idx])
            for idx in (np.concatenate([positions[c] for c in draw]) for draw in draws)
        ])
    return pd.DataFrame(scores)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree_method", default="hist", choices=["exact", "hist"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--only", default=None, help="Comma-separated subset of row names.")
    ap.add_argument("--output_dir", default="results/mole_method_all_rows")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--aggregate_only", action="store_true")
    args = ap.parse_args()

    rows = ROWS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        rows = [(r, f) for r, f in ROWS if r in wanted]

    if not args.aggregate_only:
        for i, (row, features) in enumerate(rows, 1):
            print(f"[{i}/{len(rows)}] {row} ({features})", flush=True)
            started = time.time()
            proc = subprocess.run([
                sys.executable, "-I", "-u", "scripts/reproduce_mole_original.py",
                "--features", features, "--tree_method", args.tree_method,
                "--device", args.device, "--output_dir", args.output_dir, "--skip_existing",
            ])
            print(f"      exit {proc.returncode} in {(time.time() - started) / 60:.1f} min", flush=True)
            if proc.returncode != 0:
                return proc.returncode

    summaries, preds = {}, {}
    for row, features in rows:
        s = find(args.output_dir, "summary", features, args.tree_method, args.device)
        p = find(args.output_dir, "test_predictions", features, args.tree_method, args.device)
        if s and p:
            summaries[row] = json.load(open(s))
            preds[row] = pd.read_csv(p)
    if not summaries:
        print("No finished rows found.")
        return 1

    boot = bootstrap(preds, args.n_boot, seed=0)
    has_ref = "mole_only" in boot
    table = []
    for row in summaries:
        s, b = summaries[row], boot[row]
        entry = {
            "row": row,
            "n_features": s["n_features"],
            "val_pr_auc": round(s["valid"]["prauc_trapz"], 4),
            "test_pr_auc": round(s["test"]["prauc_trapz"], 4),
            "test_ap": round(s["test"]["average_precision"], 4),
            "test_ap_ci95": f"[{np.percentile(b, 2.5):.4f}, {np.percentile(b, 97.5):.4f}]",
            "fit_minutes": round(s["fit_minutes"], 1),
        }
        if has_ref and row != "mole_only":
            d = b - boot["mole_only"]
            entry["diff_vs_mole"] = round(float(np.mean(d)), 4)
            entry["diff_ci95"] = f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
            entry["p_better_than_mole"] = round(float((d > 0).mean()), 3)
        table.append(entry)

    table = pd.DataFrame(table)
    tag = f"{args.tree_method}" + ("" if args.device == "cpu" else f"_{args.device}")
    out = os.path.join(args.output_dir, f"table_mole_method_{tag}.csv")
    table.to_csv(out, index=False)
    pd.set_option("display.width", 240)
    print(table.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
