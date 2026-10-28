"""Task 1: per-trial values of the main table and paired tests between rows."""

from __future__ import annotations

import glob
import hashlib
import json

import pandas as pd
from scipy import stats

from common import (DATA, LABEL, N_TRIALS, ROWS, TRIALS_FULL, ensure_dirs, fmt_pm, load_run,
                    write_section)

PAIRS = [
    ("mole_plus_gpt-5.2", "rdkit_only"),
    ("mole_plus_gpt-5.2", "mole_only"),
    ("mole_plus_gpt-4.1", "rdkit_only"),
    ("mole_plus_gpt-5.1", "rdkit_only"),
    ("mole_plus_rdkit", "rdkit_only"),
]
TUNED = ["eta", "max_depth", "subsample", "colsample_bytree", "min_child_weight", "gamma"]


def split_fingerprints() -> pd.Series:
    """Hash of the (compound, split) assignment each of the 45 runs actually used."""
    out = {}
    for path in sorted(glob.glob(f"{TRIALS_FULL}/trial_*/*/exact_split_used_*.csv")):
        df = pd.read_csv(path)
        id_col = "prestwick_ID" if "prestwick_ID" in df else df.columns[0]
        split_col = "split" if "split" in df else df.columns[1]
        key = df[[id_col, split_col]].sort_values(id_col).to_csv(index=False)
        out[path] = hashlib.sha256(key.encode()).hexdigest()[:16]
    return pd.Series(out)


def main() -> None:
    ensure_dirs()
    records = []
    for row, label in ROWS:
        for k in range(N_TRIALS):
            m, s = load_run(k, row)
            records.append({
                "representation": label,
                "trial": k,
                "val_prauc_perstrain": m["val_macro_pr_auc_by_strain"],
                "test_prauc_perstrain": m["test_macro_pr_auc_by_strain"],
                "test_prauc_pooled": m["pr_auc"],
                "selected_hyperparams": json.dumps({p: s["best_params"][p] for p in TUNED}),
                # Mean val PR-AUC over the three search fits (seeds 0,1,2) that selected
                # the configuration; val_prauc_perstrain is the final seed-k model's value.
                "val_prauc_selection_mean": s["best_val_macro_pr_auc"],
            })
    df = pd.DataFrame(records)
    df.to_csv(f"{DATA}/per_trial_results.csv", index=False)

    fp = split_fingerprints()
    identical = fp.nunique() == 1
    published = pd.read_csv("data/raw_data/maier_scaffold_split.tsv", sep="\t")
    counts = published["split"].value_counts().to_dict()

    wide = df.pivot(index="trial", columns="representation", values="test_prauc_perstrain")
    test_rows = []
    for a, b in PAIRS:
        x, y = wide[LABEL[a]], wide[LABEL[b]]
        d = x - y
        t = stats.ttest_rel(x, y)
        w = stats.wilcoxon(x, y)
        test_rows.append({
            "comparison": f"{LABEL[a]} vs {LABEL[b]}",
            "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
            "wins": f"{int((d > 0).sum())}/{len(d)}",
            "t": t.statistic, "p_t": t.pvalue, "p_wilcoxon": w.pvalue,
        })
    tests = pd.DataFrame(test_rows)
    tests.to_csv(f"{DATA}/paired_tests.csv", index=False)

    # ---- markdown ----
    lines = ["## Task 1. Per-trial values and paired tests", ""]
    lines += ["Per-trial values for all 9 rows x 5 trials: `data/per_trial_results.csv`. "
              "`val_prauc_perstrain` is the validation PR-AUC of the final trial-k model (the value "
              "behind the main table's validation column); the extra column "
              "`val_prauc_selection_mean` is the mean over the three search fits (seeds 0,1,2) "
              "that selected the configuration.", ""]
    lines += ["**Is the train/val/test scaffold split identical across all 5 trials?** "
              + ("**Yes.**" if identical else "**No.**")
              + f" All {len(fp)} runs (9 rows x 5 trials) wrote the split they used; "
              f"{fp.nunique()} distinct assignment(s). It is MolE's published scaffold split "
              f"(`maier_scaffold_split.tsv`, {counts.get('train')} train / {counts.get('valid')} "
              f"valid / {counts.get('test')} test compounds), fixed for every trial. Trials differ "
              "only in the hyperparameter configurations sampled and the training seed.", ""]

    lines += ["Per-trial test PR-AUC (mean per-strain):", "",
              "| Representation | " + " | ".join(f"Trial {k}" for k in range(N_TRIALS)) + " | Mean ± SD |",
              "|---|" + "---|" * (N_TRIALS + 1)]
    for _, label in ROWS:
        v = wide[label]
        lines.append(f"| {label} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {fmt_pm(v.mean(), v.std(ddof=1))} |")
    lines.append("")

    lines += ["Paired comparisons on test PR-AUC (per-strain), pairs = same trial index, n = 5:", "",
              "| Comparison | Mean diff | SD of diff | Wins | t | p (paired t) | p (Wilcoxon) |",
              "|---|---|---|---|---|---|---|"]
    for r in tests.itertuples():
        lines.append(f"| {r.comparison} | {r.mean_diff:+.4f} | {r.sd_diff:.4f} | {r.wins} | "
                     f"{r.t:.3f} | {r.p_t:.3f} | {r.p_wilcoxon:.3f} |")
    lines += ["",
              "With n = 5 the two-sided exact Wilcoxon test cannot go below p = 0.0625 (all five "
              "differences with the same sign), so it can never reach 0.05. No correction for "
              "multiple comparisons is applied above."]
    write_section("task1", "\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
