"""Aggregate the per-seed metrics from run_table1.py into one table.

Writes results/table1/table1.csv (mean +/- sd across seeds, per row) and
results/table1/RESULTS_MANIFEST.csv (every reported number tied to the run
directory, metric and commit that produced it).
"""

from __future__ import annotations

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


def load_row(row_dir: str) -> pd.DataFrame | None:
    hits = glob.glob(os.path.join(row_dir, "seed_metrics_*.csv"))
    if not hits:
        return None
    return pd.read_csv(hits[0])


def main() -> None:
    commit = git_commit()
    summary_rows, manifest_rows = [], []

    for row, label in ROW_ORDER:
        row_dir = os.path.join(OUT_ROOT, row)
        seeds = load_row(row_dir)
        if seeds is None:
            continue

        primary, secondary = seeds[PRIMARY], seeds[SECONDARY]
        sd = primary.std(ddof=1) if len(primary) > 1 else float("nan")

        summary_rows.append({
            "row": label,
            "n_seeds": len(seeds),
            "test_pr_auc_mean": round(primary.mean(), 4),
            "test_pr_auc_sd": round(sd, 4) if len(primary) > 1 else "",
            "test_pr_auc_min": round(primary.min(), 4),
            "test_pr_auc_max": round(primary.max(), 4),
            "test_pooled_pr_auc_mean": round(secondary.mean(), 4),
            "test_f1_mean": round(seeds["f1"].mean(), 4),
            "accuracy_mean": round(100 * seeds["accuracy"].mean(), 2),
            "n_test_pairs": int(seeds["n_test"].iloc[0]),
            "n_features": int(seeds["n_features"].iloc[0]),
        })

        meta_path = glob.glob(os.path.join(row_dir, "summary_metrics_*[!0-9].json"))
        best_params = ""
        if meta_path:
            with open(meta_path[0]) as fh:
                best_params = json.dumps(json.load(fh).get("best_params", {}))

        manifest_rows.append({
            "row": label,
            "run_dir": row_dir.replace(os.sep, "/"),
            "primary_metric": "mean per-strain test PR-AUC",
            "seeds": ",".join(str(s) for s in seeds["seed"].tolist()),
            "commit": commit,
            "best_params": best_params,
        })

    if not summary_rows:
        print(f"No completed rows found under {OUT_ROOT}/. Run scripts/run_table1.py first.")
        return

    table = pd.DataFrame(summary_rows)
    manifest = pd.DataFrame(manifest_rows)

    table.to_csv(os.path.join(OUT_ROOT, "table1.csv"), index=False)
    manifest.to_csv(os.path.join(OUT_ROOT, "RESULTS_MANIFEST.csv"), index=False)

    pd.set_option("display.width", 220)
    print(table.to_string(index=False))
    print(f"\nWrote {OUT_ROOT}/table1.csv and {OUT_ROOT}/RESULTS_MANIFEST.csv")


if __name__ == "__main__":
    main()
