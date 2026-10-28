"""Aggregate the per-seed metrics from run_table1.py into one table.

Writes results/table1/table1.csv (mean +/- sd across seeds, per row) and
results/table1/RESULTS_MANIFEST.csv (every reported number tied to the run
directory, metric and commit that produced it).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess

import pandas as pd

OUT_ROOT = "results/table1"

# Display order and labels for the paper table.
ROW_ORDER = [
    ("mole_only", "MolE embeddings only"),
    ("rdkit_only", "Chemical descriptors only"),
    ("mole_plus_rdkit", "MolE + chemical descriptors"),
    ("mole_plus_gpt-4o", "MolE + GPT-4o"),
    ("mole_plus_gpt-4o-mini", "MolE + GPT-4o-mini"),
    ("mole_plus_gpt-4.1", "MolE + GPT-4.1"),
    ("mole_plus_gpt-4.1-mini", "MolE + GPT-4.1-mini"),
    ("mole_plus_gpt-5.1", "MolE + GPT-5.1"),
    ("mole_plus_gpt-5.2", "MolE + GPT-5.2"),
    ("gpt_only_gpt-4o", "GPT-4o only"),
    ("gpt_only_gpt-4o-mini", "GPT-4o-mini only"),
    ("gpt_only_gpt-4.1", "GPT-4.1 only"),
    ("gpt_only_gpt-4.1-mini", "GPT-4.1-mini only"),
    ("gpt_only_gpt-5.1", "GPT-5.1 only"),
    ("gpt_only_gpt-5.2", "GPT-5.2 only"),
    *[(f"mole_plus_guided_{m}", f"MolE + {lab} (guided)") for m, lab in (
        ("gpt-4o-mini", "GPT-4o-mini"), ("gpt-4.1-mini", "GPT-4.1-mini"), ("gpt-4o", "GPT-4o"),
        ("gpt-4.1", "GPT-4.1"), ("gpt-5.1", "GPT-5.1"), ("gpt-5.2", "GPT-5.2"))],
    *[(f"guided_only_{m}", f"{lab} only (guided)") for m, lab in (
        ("gpt-4o-mini", "GPT-4o-mini"), ("gpt-4.1-mini", "GPT-4.1-mini"), ("gpt-4o", "GPT-4o"),
        ("gpt-4.1", "GPT-4.1"), ("gpt-5.1", "GPT-5.1"), ("gpt-5.2", "GPT-5.2"))],
]

# The primary metric. Mean per-strain PR-AUC, matching how MolE report per-strain
# performance; pooled PR-AUC is carried alongside as a secondary column.
PRIMARY = "test_macro_pr_auc_by_strain"
SECONDARY = "pr_auc"


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def best_params_of(row_dir: str) -> str:
    meta_path = glob.glob(os.path.join(row_dir, "summary_metrics_*[!0-9].json"))
    if not meta_path:
        return ""
    with open(meta_path[0]) as fh:
        return json.dumps(json.load(fh).get("best_params", {}))


def load_row(out_root: str, row: str) -> tuple[pd.DataFrame, list[str]] | None:
    """Per-run metrics for one row, plus the run directories they came from.

    Seed runs (run_table1.py --seeds) keep every seed in one seed_metrics CSV.
    Trial runs (run_table1.py --trials) keep one run per <out_root>/trial_<k>/<row>;
    those are stacked, one line per trial.
    """
    trial_dirs = sorted(glob.glob(os.path.join(out_root, "trial_*", row)))
    dirs = trial_dirs or [os.path.join(out_root, row)]

    frames, used = [], []
    for d in dirs:
        hits = glob.glob(os.path.join(d, "seed_metrics_*.csv"))
        if not hits:
            continue
        df = pd.read_csv(hits[0])
        if trial_dirs:
            df.insert(0, "trial", os.path.basename(os.path.dirname(d)))
        frames.append(df)
        used.append(d)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True), used


def mean_sd(s: pd.Series) -> tuple[float, float | str]:
    return round(s.mean(), 4), (round(s.std(ddof=1), 4) if len(s) > 1 else "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=OUT_ROOT,
                    help=f"Run directory written by run_table1.py (default: {OUT_ROOT}).")
    out_root = ap.parse_args().root

    commit = git_commit()
    summary_rows, manifest_rows = [], []

    for row, label in ROW_ORDER:
        loaded = load_row(out_root, row)
        if loaded is None:
            continue
        runs, dirs = loaded

        primary, secondary = runs[PRIMARY], runs[SECONDARY]
        val = runs["val_macro_pr_auc_by_strain"]

        summary_rows.append({
            "row": label,
            "n_runs": len(runs),
            "val_pr_auc_mean": mean_sd(val)[0],
            "val_pr_auc_sd": mean_sd(val)[1],
            "test_pr_auc_mean": mean_sd(primary)[0],
            "test_pr_auc_sd": mean_sd(primary)[1],
            "test_pr_auc_min": round(primary.min(), 4),
            "test_pr_auc_max": round(primary.max(), 4),
            "test_pooled_pr_auc_mean": mean_sd(secondary)[0],
            "test_pooled_pr_auc_sd": mean_sd(secondary)[1],
            "test_f1_mean": round(runs["f1"].mean(), 4),
            "accuracy_mean": round(100 * runs["accuracy"].mean(), 2),
            "n_test_pairs": int(runs["n_test"].iloc[0]),
            "n_features": int(runs["n_features"].iloc[0]),
        })

        for d, (_, r) in zip(dirs if "trial" in runs else dirs * len(runs), runs.iterrows()):
            manifest_rows.append({
                "row": label,
                "run_dir": d.replace(os.sep, "/"),
                "trial": r.get("trial", ""),
                "seed": int(r["seed"]),
                "val_pr_auc": r["val_macro_pr_auc_by_strain"],
                "test_pr_auc": r[PRIMARY],
                "test_pooled_pr_auc": r[SECONDARY],
                "commit": commit,
                "best_params": best_params_of(d),
            })

    if not summary_rows:
        print(f"No completed rows found under {out_root}/. Run scripts/run_table1.py first.")
        return

    table = pd.DataFrame(summary_rows)
    manifest = pd.DataFrame(manifest_rows)

    table.to_csv(os.path.join(out_root, "table1.csv"), index=False)
    manifest.to_csv(os.path.join(out_root, "RESULTS_MANIFEST.csv"), index=False)

    pd.set_option("display.width", 240)
    print(table.to_string(index=False))
    print(f"\nWrote {out_root}/table1.csv and {out_root}/RESULTS_MANIFEST.csv")


if __name__ == "__main__":
    main()
