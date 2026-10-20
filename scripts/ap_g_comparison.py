"""Attribute the external-MIC correlation: MolE-only vs GPT-only vs MolE+GPT.

Section 3.5 of the submission reports a single Spearman correlation for the
MolE+GPT-5.2 model and states that it cannot separate the GPT contribution.
Supplementary Table 10 of the MolE paper already carries the published MolE and
ECFP4 antimicrobial-potential scores for the same compounds, so the matched
comparison the reviewers asked for needs no retraining.

Also reports the exact compound / observation / plotted-point counts, which the
submission gives inconsistently (29 compounds, 38 observations, ~34 points).

Outputs results/mechchem/ap_g_comparison.csv and prints the summary table.
"""

from __future__ import annotations

import math
import os

import pandas as pd
from scipy.stats import spearmanr

ST10 = "results/mechchem/SupplementaryTable10.xlsx"
GPT_MOLE = "results/mechchem/ap_scores_gpt52.csv"      # MolE + GPT-5.2
GPT_ONLY = "results/mechchem/ap_scores_gpt52_gptonly.csv"
OUT = "results/mechchem/ap_g_comparison.csv"

LN2 = math.log(2.0)


def load_st10() -> pd.DataFrame:
    """ST10 opens with a multi-line caption, so find the real header row."""
    probe = pd.read_excel(ST10, header=None)
    header_row = next(
        i for i, row in probe.iterrows()
        if row.astype(str).str.strip().eq("Catalog Number").any()
    )
    df = pd.read_excel(ST10, header=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    return df.dropna(how="all").dropna(subset=["ProductName"])


def to_observations(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (compound, gram class) MIC observation."""
    records = []
    for _, r in df.iterrows():
        for col, gram in (("MIC_gpositive", "positive"), ("MIC_gnegative", "negative")):
            mic = pd.to_numeric(r.get(col), errors="coerce")
            if pd.isna(mic) or mic <= 0:
                continue
            records.append({
                "ProductName": r["ProductName"],
                "Catalog Number": r.get("Catalog Number"),
                "gram": gram,
                "MIC": float(mic),
                "log2_MIC": math.log2(float(mic)),
                "ap_mole_total": pd.to_numeric(r.get("apscore_total.MolE"), errors="coerce"),
                "ap_mole_gpos": pd.to_numeric(r.get("apscore_gpositive.MolE"), errors="coerce"),
                "ap_mole_gneg": pd.to_numeric(r.get("apscore_gnegative.MolE"), errors="coerce"),
                "ap_ecfp4_total": pd.to_numeric(r.get("apscore_total.ECFP4"), errors="coerce"),
            })
    return pd.DataFrame(records)


def attach_ours(obs: pd.DataFrame) -> pd.DataFrame:
    """Add our AP-G scores, converted from natural log to log2 to match MolE."""
    for path, col in ((GPT_MOLE, "ap_mole_plus_gpt52"), (GPT_ONLY, "ap_gpt52_only")):
        if not os.path.exists(path):
            print(f"  (missing {path}; skipping {col})")
            continue
        src = pd.read_csv(path)
        # build_train.py computes AP-G with natural log; MolE define it as log2.
        src[col] = src["ap_score_gpt"] / LN2
        obs = obs.merge(src[["ProductName", col]], on="ProductName", how="left")
    return obs


def correlate(obs: pd.DataFrame, score_col: str, subset: pd.Series | None = None) -> dict:
    d = obs if subset is None else obs[subset]
    d = d.dropna(subset=[score_col, "log2_MIC"])
    if len(d) < 3:
        return {"n_obs": len(d), "n_compounds": d.ProductName.nunique(),
                "rho": float("nan"), "p": float("nan")}
    rho, p = spearmanr(d[score_col], d["log2_MIC"])
    return {"n_obs": len(d), "n_compounds": d.ProductName.nunique(),
            "rho": round(float(rho), 4), "p": round(float(p), 5)}


def main() -> None:
    st10 = load_st10()
    obs = attach_ours(to_observations(st10))

    print(f"ST10 rows with a ProductName : {len(st10)}")
    print(f"distinct compounds           : {st10.ProductName.nunique()}")
    print(f"MIC observations             : {len(obs)}"
          f"  (gram+ {(obs.gram == 'positive').sum()}, gram- {(obs.gram == 'negative').sum()})")
    print(f"compounds with >=1 MIC       : {obs.ProductName.nunique()}")
    dup = obs.duplicated(subset=["ap_mole_total", "log2_MIC"]).sum()
    print(f"exactly overplotted points   : {dup}  "
          f"-> {len(obs) - dup} visually distinct markers\n")

    # The matched subset: the observations our models actually score. Comparing
    # our rho on 38 observations against MolE's on 47 would not be like for like.
    scored = "ap_mole_plus_gpt52"
    matched = obs[scored].notna() & obs["log2_MIC"].notna() if scored in obs.columns else None
    if matched is not None:
        m = obs[matched]
        dup_m = m.duplicated(subset=[scored, "log2_MIC"]).sum()
        print(f"matched subset               : {len(m)} observations, "
              f"{m.ProductName.nunique()} compounds, {dup_m} exact overlaps "
              f"-> {len(m) - dup_m} visible markers\n")

    rows = []
    for label, col in [
        ("MolE (published, total)", "ap_mole_total"),
        ("ECFP4 (published, total)", "ap_ecfp4_total"),
        ("MolE + GPT-5.2 (ours)", "ap_mole_plus_gpt52"),
        ("GPT-5.2 only (ours)", "ap_gpt52_only"),
    ]:
        if col not in obs.columns:
            continue
        rows.append({"score": label, "subset": "all available", **correlate(obs, col)})
        if matched is not None:
            rows.append({"score": label, "subset": "matched subset",
                         **correlate(obs, col, matched)})

    # Gram-specific scores against gram-matched MIC values (Ma8V question 3).
    for label, col, gram in [
        ("MolE G+ (published)", "ap_mole_gpos", "positive"),
        ("MolE G- (published)", "ap_mole_gneg", "negative"),
    ]:
        rows.append({"score": label, "subset": f"gram-{gram} MIC only",
                     **correlate(obs, col, obs.gram == gram)})

    table = pd.DataFrame(rows)
    print(table.to_string(index=False))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    obs.to_csv(OUT, index=False)
    print(f"\nWrote per-observation data to {OUT}")


if __name__ == "__main__":
    main()
