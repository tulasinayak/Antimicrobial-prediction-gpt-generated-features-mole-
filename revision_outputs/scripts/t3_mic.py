"""Task 3: external MIC check for MolE-only, RDKit-only and MolE+GPT-5.2, with gram split.

Observations are built exactly as analysis/AP_VS_MIC_MCE.ipynb did for the paper: ST10 rows
predicted by MolE, one observation per reported gram-positive / gram-negative MIC, MIC <= 128,
inner-joined to the 29 compounds that have GPT-5.2 features (-> 38 observations).

Scores use the task's definition G = (1/40) sum_j log2 p_ij, and G+ / G- over the 22
gram-positive / 18 gram-negative strains recovered from MolE's published scores
(data/strain_gram.csv). The paper's original score used the natural log (same Spearman rho,
OLS slope larger by 1/ln 2 in log2 units).
"""

from __future__ import annotations

import math
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import (DATA, MCE_GPT52, MCE_LIBRARY, N_TRIALS, ST10, ensure_dirs, fmt_pm,
                    load_model, load_model_file, mce_features, predict_pairs, write_section,
                    chemdesc_names, chemdesc_model_names)

sys.path.insert(0, "notebooks")
from build_train import _STRAINS_40  # noqa: E402

REPS = [("mole_plus_gpt-5.2", "MolE + GPT-5.2", "mole_gpt52"),
        ("mole_only", "MolE", "mole"),
        ("rdkit_only", "RDKit descriptors", "rdkit")]
ORIGINAL_MODEL = "D:/Hiwi/GPT+Mole/GPT+Mole/results/freeform/gpt-5.2/xgb_model_mole_plus_gpt.pkl"
ORIGINAL_SCORES = "results/mechchem/ap_scores_gpt52.csv"
GRAM_NEG_GENERA = ("Escherichia", "Helicobacter", "Pseudomonas", "Klebsiella", "Acinetobacter",
                   "Salmonella", "Bacteroides", "Fusobacterium", "Neisseria", "Haemophilus",
                   "Vibrio", "Proteus", "Enterobacter", "Porphyromonas", "Shigella", "Serratia",
                   "Burkholderia", "Campylobacter", "Legionella", "Moraxella", "Prevotella")


def observations() -> pd.DataFrame:
    st = pd.read_excel(ST10, sheet_name="ST 10", header=6)
    st = st[st["Predicted by"].str.contains("MolE", na=False)]
    rows = []
    for _, r in st.iterrows():
        orgs = [o.strip() for o in str(r["Target_bacteria"]).split(",")] if pd.notna(r["Target_bacteria"]) else []
        for col, gram in (("MIC_gpositive", "positive"), ("MIC_gnegative", "negative")):
            if pd.isna(r[col]):
                continue
            if len(orgs) > 1:
                neg = [o for o in orgs if o.startswith(GRAM_NEG_GENERA)]
                pos = [o for o in orgs if not o.startswith(GRAM_NEG_GENERA)]
                org = ", ".join(neg if gram == "negative" else pos) or r["Target_bacteria"]
            else:
                org = r["Target_bacteria"]
            rows.append({"compound": r["ProductName"], "catalog": r["Catalog Number"],
                         "organism": org, "gram": gram, "MIC": float(r[col]),
                         "MIC_note": r["Notes"] if pd.notna(r["Notes"]) else ""})
    obs = pd.DataFrame(rows).query("MIC <= 128")
    scored = pd.read_csv(MCE_GPT52)["ProductName"]
    obs = obs[obs["compound"].isin(scored)].reset_index(drop=True)
    obs["log2MIC"] = np.log2(obs["MIC"])
    lib = pd.read_csv(MCE_LIBRARY, sep="\t").drop_duplicates("Catalog Number").set_index("Catalog Number")
    obs["smiles"] = obs["catalog"].map(lib["rdkit_no_salt"])
    return obs


def fill_missing_chemdesc(feats: pd.DataFrame, obs: pd.DataFrame) -> list[str]:
    """MolE's MCE descriptor file lacks some compounds; compute the 98 descriptors with RDKit."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    filled = []
    smiles = obs.drop_duplicates("compound").set_index("compound")["smiles"]
    for cpd in feats.index[feats["chemdesc_0"].isna()]:
        mol = Chem.MolFromSmiles(smiles.get(cpd, "")) if isinstance(smiles.get(cpd), str) else None
        if mol is None:
            continue
        vals = [getattr(Descriptors, n)(mol) if hasattr(Descriptors, n) else np.nan for n in chemdesc_names()]
        feats.loc[cpd, chemdesc_model_names()] = vals
        filled.append(cpd)
    return filled


def scores(model: dict, feats: pd.DataFrame, gram: pd.Series) -> pd.DataFrame:
    p = predict_pairs(model, feats, _STRAINS_40)
    p["l2"] = np.log2(p["proba"])
    p["gram"] = p["bacterium"].map(gram)
    g = p.groupby("compound")["l2"].mean().rename("G")
    gp = p[p.gram == "positive"].groupby("compound")["l2"].mean().rename("Gplus")
    gn = p[p.gram == "negative"].groupby("compound")["l2"].mean().rename("Gminus")
    return pd.concat([g, gp, gn], axis=1)


def correlate(obs: pd.DataFrame, s: pd.DataFrame) -> dict:
    d = obs.join(s, on="compound")
    matched = np.where(d.gram == "positive", d.Gplus, d.Gminus)
    r_all = stats.spearmanr(d.G, d.log2MIC)
    r_m = stats.spearmanr(matched, d.log2MIC)
    ols = stats.linregress(d.log2MIC, d.G)
    return {"rho_G": r_all.statistic, "p_G": r_all.pvalue, "rho_matched": r_m.statistic,
            "p_matched": r_m.pvalue, "slope_G": ols.slope, "p_slope": ols.pvalue}


def main() -> None:
    ensure_dirs()
    gram = pd.read_csv(f"{DATA}/strain_gram.csv").set_index("strain")["gram"]
    obs = observations()
    feats = mce_features()
    filled = fill_missing_chemdesc(feats, obs)

    points = obs[["compound", "smiles", "organism", "gram", "MIC", "log2MIC", "MIC_note"]].copy()
    per_trial, trial0 = [], {}
    for row, label, short in REPS:
        for k in range(N_TRIALS):
            s = scores(load_model(k, row), feats, gram)
            per_trial.append({"representation": label, "trial": k, **correlate(obs, s)})
            if k == 0:
                trial0[label] = correlate(obs, s)
                points[f"G_{short}"] = points.compound.map(s.G)
                points[f"Gplus_{short}"] = points.compound.map(s.Gplus)
                points[f"Gminus_{short}"] = points.compound.map(s.Gminus)

    # The model behind the submitted figure (older single run), log2 scale.
    orig = scores(load_model_file(ORIGINAL_MODEL), feats, gram)
    orig_corr = correlate(obs, orig)
    stored = pd.read_csv(ORIGINAL_SCORES).set_index("ProductName")["ap_score_gpt"]
    orig_reproduced = float(np.abs(orig.G * math.log(2) - stored.reindex(orig.index)).max())
    points["G_mole_gpt52_submitted_model"] = points.compound.map(orig.G)
    points["Gplus_mole_gpt52_submitted_model"] = points.compound.map(orig.Gplus)
    points["Gminus_mole_gpt52_submitted_model"] = points.compound.map(orig.Gminus)
    points.drop(columns="MIC_note").to_csv(f"{DATA}/mic_points.csv", index=False)
    pt = pd.DataFrame(per_trial)
    pt.to_csv(f"{DATA}/mic_correlations_per_trial.csv", index=False)

    # ---- point-count check (submitted figure: x = log2 MIC, y = submitted-model G) ----
    xy = points[["compound", "gram", "log2MIC"]].assign(G=points["G_mole_gpt52_submitted_model"])
    exact = xy.groupby(["log2MIC", "G"]).filter(lambda g: len(g) > 1)
    xr, yr = np.ptp(xy.log2MIC), np.ptp(xy.G)
    near = []
    for i, j in [(i, j) for i in range(len(xy)) for j in range(i + 1, len(xy))]:
        a, b = xy.iloc[i], xy.iloc[j]
        if abs(a.log2MIC - b.log2MIC) <= 0.02 * xr and abs(a.G - b.G) <= 0.02 * yr \
                and not (a.log2MIC == b.log2MIC and a.G == b.G):
            near.append((a.compound, a.gram, b.compound, b.gram, a.log2MIC, b.log2MIC, a.G, b.G))
    n_unique = xy[["log2MIC", "G"]].drop_duplicates().shape[0]

    # ---- markdown ----
    md = ["## Task 3. External MIC check", "",
          f"Observations rebuilt exactly as in the submitted analysis: ST10 rows predicted by MolE, one "
          f"observation per reported gram-positive / gram-negative MIC, MIC <= 128, inner-joined to the "
          f"29 compounds with GPT-5.2 features -> **{len(obs)} observations from {obs.compound.nunique()} "
          f"compounds** ({(obs.gram == 'positive').sum()} gram-positive, {(obs.gram == 'negative').sum()} "
          "gram-negative). `data/mic_points.csv` holds every observation with all scores.", "",
          "**Which model produced the submitted rho = -0.480?** An older one: the single-run MolE+GPT-5.2 "
          "model in `results/freeform/gpt-5.2/` (the pre-revision protocol), applied with MolE's own "
          "MedChemExpress embeddings. Re-scoring with that model reproduces the stored scores exactly "
          f"(max |diff| = {orig_reproduced:.1e} on the natural-log scale it used). It is **not** one of the "
          "5-trial models.", "",
          "Inputs for the 29 compounds: MolE embeddings and RDKit descriptors from MolE's repository "
          "(`data/04.new_predictions/*_medchemexpress.tsv.gz`), GPT-5.2 features from this project. "
          + (f"MolE's descriptor file lacks {', '.join(filled)}; its 98 descriptors were computed here "
             "with RDKit from the same SMILES. " if filled else "")
          + "Scores: G = (1/40) sum log2 p; G+ / G- over the gram-positive / gram-negative strains below.", "",
          "### 3a/3b. Spearman correlation with log2 MIC (n = 38)", "",
          "Trial-0 models of the 5-trial protocol, plus the submitted model for reference. "
          "'Matched' uses G+ for observations on gram-positive organisms and G- for gram-negative.", "",
          "| Model | rho (G) | p (G) | rho (G+/G- matched) | p (matched) | OLS slope of G on log2 MIC (p) |",
          "|---|---|---|---|---|---|"]
    rows_md = [("MolE + GPT-5.2 (submitted model)", orig_corr)] + [(f"{l} (trial 0)", trial0[l]) for _, l, _ in REPS]
    for name, c in rows_md:
        md.append(f"| {name} | {c['rho_G']:.3f} | {c['p_G']:.4f} | {c['rho_matched']:.3f} | "
                  f"{c['p_matched']:.4f} | {c['slope_G']:.3f} ({c['p_slope']:.4f}) |")
    md += ["", "Across the 5 trial models (mean ± SD of rho; range):", "",
           "| Representation | rho (G) | range | rho (G+/G- matched) | range | trials with p < 0.05 (G) |",
           "|---|---|---|---|---|---|"]
    for _, label, _ in REPS:
        t = pt[pt.representation == label]
        md.append(f"| {label} | {fmt_pm(t.rho_G.mean(), t.rho_G.std(ddof=1), 3)} | "
                  f"[{t.rho_G.min():.3f}, {t.rho_G.max():.3f}] | "
                  f"{fmt_pm(t.rho_matched.mean(), t.rho_matched.std(ddof=1), 3)} | "
                  f"[{t.rho_matched.min():.3f}, {t.rho_matched.max():.3f}] | {(t.p_G < 0.05).sum()}/5 |")
    md += ["", f"The submitted rho = -0.480 is reproduced by the submitted model ({orig_corr['rho_G']:.3f}). "
           f"Its OLS slope in log2 units is {orig_corr['slope_G']:.3f}; the paper's -0.200 was in natural-log "
           f"units (-0.200 / ln 2 = {-0.200 / math.log(2):.3f}).", "",
           "Strains counted as gram-positive / gram-negative (recovered from MolE's published G+/G- scores, "
           "which are exact linear functions of these subsets; `data/strain_gram.csv`):", ""]
    for g in ("positive", "negative"):
        names = sorted(gram[gram == g].index)
        md.append(f"- **Gram-{g} ({len(names)}):** " + "; ".join(names))
    notes = obs[obs.MIC_note != ""]
    md += ["", f"Data caveat: {len(notes)} of the 38 observations carry an ST10 note that the MIC is not in "
           "ug/mL (" + "; ".join(f"{r.compound} [{r.gram}]: '{r.MIC_note}'" for r in notes.itertuples())
           + "). They are kept unconverted, as in the submitted analysis and in MolE's ST10 filtering.", "",
           "### 3c. Point count", "",
           f"The figure plots {len(xy)} observations. Unique (log2 MIC, G) positions: **{n_unique}**.", ""]
    if len(exact):
        md += ["Exactly coincident observations (same compound, identical MIC for both gram classes):", ""]
        for (x, y), g in exact.groupby(["log2MIC", "G"]):
            md.append(f"- {', '.join(sorted(set(g.compound)))}: log2 MIC = {x:.3f}, G = {y:.3f}, "
                      f"grams {', '.join(g.gram)}")
    md += ["", f"Visually overlapping pairs (within 2% of the axis range in both x and y; markers in the "
           f"submitted plot are larger than that): {len(near)}", ""]
    md += [f"- {a} ({ga}) and {b} ({gb}): log2 MIC {x1:.2f} / {x2:.2f}, G {y1:.3f} / {y2:.3f}"
           for a, ga, b, gb, x1, x2, y1, y2 in near]
    md += ["", f"So {len(xy)} observations collapse to {n_unique} distinct positions, and "
           f"{len(near)} further pairs overlap at marker size, which accounts for the 34-35 points a reader "
           "can count. No observations were dropped or duplicated."]
    write_section("task3", "\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
