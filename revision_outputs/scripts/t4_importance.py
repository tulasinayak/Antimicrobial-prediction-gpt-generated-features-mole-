"""Task 4: gain importance for MolE + GPT-5.2, averaged over the 5 trial models.

Importance type 'gain' (average gain per split) as in the submission's table; a feature a
trial model never split on contributes 0 to that trial. Group shares use 'total_gain'
(summed over splits), which is what "fraction of total gain" means.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from common import (DATA, GPT_FILES, N_TRIALS, ensure_dirs, chemdesc_names, load_model,
                    write_section)

ROW = "mole_plus_gpt-5.2"


def importances(kind: str) -> pd.DataFrame:
    cols = {}
    for k in range(N_TRIALS):
        m = load_model(k, ROW)
        names = list(m["preproc"].get_feature_names_out())
        raw = m["booster"].get_score(importance_type=kind)
        cols[k] = pd.Series({names[int(f[1:])]: v for f, v in raw.items()}).reindex(names).fillna(0.0)
    return pd.DataFrame(cols)


def block(name: str) -> str:
    return name.split("__", 1)[0]


def main() -> None:
    ensure_dirs()
    gain = importances("gain")
    total = importances("total_gain")

    summary = pd.DataFrame({
        "feature": [n.split("__", 1)[1] for n in gain.index],
        "block": [block(n) for n in gain.index],
        "mean_gain": gain.mean(1), "sd_gain": gain.std(1, ddof=1),
        "n_trials_used": (gain > 0).sum(1),
        "mean_total_gain": total.mean(1),
    }).sort_values("mean_gain", ascending=False)

    gpt = summary[summary.block == "gpt"]
    mole = summary[(summary.block == "mole") & ~summary.feature.str.startswith("missingindicator")]
    top_gpt, top_mole = gpt.head(15).copy(), mole.head(3).copy()

    # Max |Spearman| of each top GPT feature against the 98 RDKit descriptors (1,197 compounds).
    g = pd.read_csv(GPT_FILES["gpt-5.2"]).set_index("prestwick_ID")
    rd = pd.read_csv("data/raw_data/maier_chemdesc_representation.tsv", sep="\t").set_index("prestwick_ID")
    rd = rd[~rd.index.duplicated()].loc[g.index, chemdesc_names()]
    best, best_rho = [], []
    for f in top_gpt.feature:
        rhos = rd.apply(lambda c: stats.spearmanr(g[f], c, nan_policy="omit").statistic)
        rhos = rhos.abs().dropna()
        best.append(rhos.idxmax())
        best_rho.append(rhos.max())
    top_gpt["max_abs_spearman_rdkit"] = best_rho
    top_gpt["best_rdkit_descriptor"] = best

    table = pd.concat([top_gpt, top_mole]).sort_values("mean_gain", ascending=False)
    table.insert(0, "rank", range(1, len(table) + 1))
    table.to_csv(f"{DATA}/importance_gpt52.csv", index=False)

    tot = summary.mean_total_gain.sum()
    shares = summary.groupby("block").mean_total_gain.sum() / tot
    top10_share = gpt.head(10).mean_total_gain.sum() / tot
    top10_gpt_vs_mole = gpt.head(10).mean_total_gain.sum() / (gpt.head(10).mean_total_gain.sum()
                                                             + mole.mean_total_gain.sum())

    md = ["## Task 4. Feature importance, MolE + GPT-5.2 (5 trial models)", "",
          "XGBoost gain importance (`importance_type='gain'`, average gain per split, as in the submitted "
          "table), mean ± SD over the 5 trial models; a trial that never splits on a feature counts as 0. "
          "Top 15 GPT-5.2 features and the 3 highest-gain MolE dimensions, ranked together. "
          "`max |rho|` is the largest absolute Spearman correlation with any of the 98 RDKit descriptors "
          "over the 1,197 Prestwick compounds. Full list: `data/importance_gpt52.csv`.", "",
          "| Rank | Feature | Block | Mean gain ± SD | Trials used | Max abs(rho) with RDKit | Best RDKit match |",
          "|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        rho = f"{r.max_abs_spearman_rdkit:.3f}" if pd.notna(getattr(r, "max_abs_spearman_rdkit", np.nan)) else "-"
        desc = r.best_rdkit_descriptor if isinstance(getattr(r, "best_rdkit_descriptor", None), str) else "-"
        md.append(f"| {r.rank} | `{r.feature}` | {'GPT-5.2' if r.block == 'gpt' else 'MolE'} | "
                  f"{r.mean_gain:.1f} ± {r.sd_gain:.1f} | {r.n_trials_used}/5 | {rho} | {desc} |")
    md += ["", "Share of total gain (`total_gain`, summed over all splits, mean over trials):", "",
           "| Group | Share of total gain |", "|---|---|",
           f"| All GPT-5.2 features ({(summary.block == 'gpt').sum()}) | {shares.get('gpt', 0):.1%} |",
           f"| Top-10 GPT-5.2 features | {top10_share:.1%} |",
           f"| All MolE dimensions incl. missingness indicators ({(summary.block == 'mole').sum()}) | {shares.get('mole', 0):.1%} |",
           f"| Strain one-hot ({(summary.block == 'cat').sum()}) | {shares.get('cat', 0):.1%} |",
           "", f"Top-10 GPT features vs all MolE dimensions: the top-10 GPT features hold "
           f"{top10_gpt_vs_mole:.1%} of their combined total gain, MolE dimensions {1 - top10_gpt_vs_mole:.1%}. "
           "Per-split `gain` favours features used in few, high-gain splits; `total_gain` reflects overall "
           "reliance, which is dominated by the 935 MolE dimensions and the strain identity."]
    write_section("task4", "\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
