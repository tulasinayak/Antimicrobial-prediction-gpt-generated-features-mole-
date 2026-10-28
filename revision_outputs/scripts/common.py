"""Shared paths, labels and helpers for the revision analyses.

Run every script from the repository root, e.g.
    python -I revision_outputs/scripts/t1_per_trial.py
"""

from __future__ import annotations

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd

OUT = "revision_outputs"
DATA = f"{OUT}/data"
FIGS = f"{OUT}/figures"
SECTIONS = f"{OUT}/sections"

# Small per-trial metrics (committed) and the full Kaggle outputs incl. models (not committed).
TRIALS = "results/table1_trials"
TRIALS_FULL = "results/table1_trials_full"
N_TRIALS = 5

RAW = "data/raw_data"
CHEMDESC = f"{RAW}/maier_chemdesc_representation.tsv"
PRESTWICK = f"{RAW}/prestwick_library_screened.tsv.gz"
GPT_DIR = "data/gpt_generation_without_guidance"
GPT_FILES = {
    "gpt-4o-mini": f"{GPT_DIR}/gpt_features4omini_gpt-4o-mini_freeform.csv",
    "gpt-4.1-mini": f"{GPT_DIR}/gpt_features_4_gpt-4.1-mini_freeform.csv",
    "gpt-4o": f"{GPT_DIR}/gpt_features4o_gpt-4o_freeform.csv",
    "gpt-4.1": f"{GPT_DIR}/gpt_features_4_gpt-4.1_freeform.csv",
    "gpt-5.1": f"{GPT_DIR}/gpt_features5_gpt-5.1_freeform.csv",
    "gpt-5.2": f"{GPT_DIR}/gpt_features5_gpt-5.2_freeform.csv",
}
GPT_MODELS = list(GPT_FILES)  # paper order
GPT_LABEL = {"gpt-4o-mini": "GPT-4o-mini", "gpt-4.1-mini": "GPT-4.1-mini", "gpt-4o": "GPT-4o",
             "gpt-4.1": "GPT-4.1", "gpt-5.1": "GPT-5.1", "gpt-5.2": "GPT-5.2"}
METADATA_COLS = ["prestwick_ID", "ProductName", "smiles_resolution", "input"]

# Main-table rows in paper order: (run directory name, display label).
ROWS = [
    ("mole_only", "MolE"),
    ("rdkit_only", "RDKit descriptors"),
    ("mole_plus_rdkit", "MolE + RDKit"),
    ("mole_plus_gpt-4o-mini", "MolE + GPT-4o-mini"),
    ("mole_plus_gpt-4.1-mini", "MolE + GPT-4.1-mini"),
    ("mole_plus_gpt-4o", "MolE + GPT-4o"),
    ("mole_plus_gpt-4.1", "MolE + GPT-4.1"),
    ("mole_plus_gpt-5.1", "MolE + GPT-5.1"),
    ("mole_plus_gpt-5.2", "MolE + GPT-5.2"),
]
LABEL = dict(ROWS)

# MolE's MedChemExpress inputs, from rolayoalarcon/mole_antimicrobial_potential
# data/04.new_predictions/ (downloaded unchanged into data/mce/).
MCE_MOLE = "data/mce/MolE_representation_medchemexpress.tsv.gz"
MCE_CHEMDESC = "data/mce/chemDesc_representation_medchemexpress.tsv.gz"
MCE_LIBRARY = "data/mce/medchemexpress_filtered.tsv.gz"
MCE_GPT52 = "results/mechchem/gpt_features_chemical_library_fig5_compounds_gpt-5.2_freeform.csv"
MCE_CATALOG_MAP = "results/mechchem/fig5_compound_catalog_map.csv"
ST10 = "results/mechchem/SupplementaryTable10.xlsx"
MOLE_MCE_PREDICTIONS = "results/mechchem/mole_mce_predictions.xlsx"


def ensure_dirs() -> None:
    for d in (DATA, FIGS, SECTIONS):
        os.makedirs(d, exist_ok=True)


def write_section(name: str, text: str) -> None:
    ensure_dirs()
    with open(f"{SECTIONS}/{name}.md", "w", encoding="utf-8") as fh:
        fh.write(text.rstrip() + "\n")


def run_dir(trial: int, row: str, full: bool = False) -> str:
    return os.path.join(TRIALS_FULL if full else TRIALS, f"trial_{trial}", row)


def load_run(trial: int, row: str) -> tuple[pd.Series, dict]:
    """The trial's test metrics row and its summary JSON (selected params etc.)."""
    d = run_dir(trial, row)
    metrics = pd.read_csv(glob.glob(f"{d}/seed_metrics_*.csv")[0]).iloc[0]
    summary_path = [p for p in glob.glob(f"{d}/summary_metrics_*.json") if "_seed" not in p][0]
    with open(summary_path) as fh:
        summary = json.load(fh)
    return metrics, summary


def load_model_file(path: str) -> dict:
    """{'preproc', 'booster', 'threshold'} from a build_train.py pickle."""
    import joblib
    from sklearn.impute import SimpleImputer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(path)
    model = bundle["model"]
    # Pickles from scikit-learn < 1.8 carry SimpleImputer._fit_dtype, renamed _fill_dtype.
    for _, tr, _ in model["preproc"].transformers_:
        for _, step in getattr(tr, "steps", []):
            if isinstance(step, SimpleImputer) and hasattr(step, "_fit_dtype") \
                    and not hasattr(step, "_fill_dtype"):
                step._fill_dtype = step._fit_dtype
    model["booster"].set_param({"device": "cpu"})
    return model


def load_model(trial: int, row: str) -> dict:
    """Trial model of the 5-trial protocol (trained on the Kaggle GPU)."""
    return load_model_file(glob.glob(f"{run_dir(trial, row, full=True)}/xgb_model_*.pkl")[0])


def expected_columns(preproc) -> dict[str, list[str]]:
    return {name: list(cols) for name, _, cols in preproc.transformers_ if name != "remainder"}


def predict_pairs(model: dict, compound_features: pd.DataFrame, strains: list[str]) -> pd.DataFrame:
    """Probability for every compound x strain, built the way build_train.py scores test pairs.

    compound_features: one row per compound (index = compound id), columns named as the
    model's raw inputs (mole_<i>, RDKit descriptor names, GPT feature names). Columns the
    model expects but the frame lacks become NaN for MolE/RDKit (median-imputed by the
    pipeline) and 0 for GPT features (the training-time GPT fill value).
    """
    import xgboost as xgb
    pre, booster = model["preproc"], model["booster"]
    cols = expected_columns(pre)
    feat = compound_features.copy()
    for block, names in cols.items():
        if block == "cat":
            continue
        missing = [c for c in names if c not in feat.columns]
        if missing:
            fill = 0.0 if block == "gpt" else np.nan
            feat = pd.concat([feat, pd.DataFrame(fill, index=feat.index, columns=missing)], axis=1)
    pairs = (feat.reset_index(names="compound")
             .merge(pd.DataFrame({"bacterium": strains}), how="cross"))
    X = pre.transform(pairs)
    proba = booster.predict(xgb.DMatrix(X))
    return pairs[["compound", "bacterium"]].assign(proba=proba)


def chemdesc_names() -> list[str]:
    """The 98 RDKit descriptor names, in file order."""
    return [c for c in pd.read_csv(CHEMDESC, sep="\t", nrows=0).columns if c != "prestwick_ID"]


def chemdesc_model_names() -> list[str]:
    """build_train.py renames the descriptors positionally to chemdesc_<i>."""
    return [f"chemdesc_{i}" for i in range(len(chemdesc_names()))]


def prestwick_features() -> pd.DataFrame:
    """Raw MolE + RDKit + GPT-5.2 inputs for the 1,197 Prestwick compounds (index prestwick_ID)."""
    mole = pd.read_csv(f"{RAW}/maier_mole_representation_with_id.tsv.gz", sep="\t").set_index("prestwick_ID")
    mole = mole[[str(i) for i in range(1000)]].set_axis([f"mole_{i}" for i in range(1000)], axis=1)
    chem = pd.read_csv(CHEMDESC, sep="\t").set_index("prestwick_ID")
    chem = chem[~chem.index.duplicated()].set_axis(chemdesc_model_names(), axis=1)
    gpt = pd.read_csv(GPT_FILES["gpt-5.2"]).set_index("prestwick_ID").select_dtypes("number")
    return pd.concat([mole, chem, gpt], axis=1).loc[mole.index]


def mce_features() -> pd.DataFrame:
    """Raw MolE + RDKit + GPT-5.2 inputs for the 29 scored MCE compounds (index ProductName).

    MolE embeddings and RDKit descriptors are MolE's own MedChemExpress files; the
    GPT-5.2 features are this project's free-form generation for the same compounds.
    """
    gpt = pd.read_csv(MCE_GPT52).set_index("ProductName").select_dtypes("number")
    catalog = pd.read_csv(MCE_CATALOG_MAP).set_index("ProductName")["Catalog Number"].loc[gpt.index]
    mole = pd.read_csv(MCE_MOLE, sep="\t", index_col=0)
    mole = mole.set_axis([f"mole_{i}" for i in range(mole.shape[1])], axis=1)
    chem = pd.read_csv(MCE_CHEMDESC, sep="\t", index_col=0).drop(columns="smiles", errors="ignore")
    mole = mole.reindex(catalog.values).set_axis(gpt.index)
    chem = chem[~chem.index.duplicated()][chemdesc_names()].set_axis(chemdesc_model_names(), axis=1)
    chem = chem.reindex(catalog.values).set_axis(gpt.index)
    return pd.concat([mole, chem, gpt], axis=1)


def fmt_pm(mean: float, sd: float, nd: int = 4) -> str:
    return f"{mean:.{nd}f} ± {sd:.{nd}f}"
