"""
build_train.py
--------------
Full ML training pipeline for antimicrobial activity prediction.

Loads the SF3 binary activity labels and combines three feature blocks
(MolE embeddings, GPT molecular features, RDKit/ChemDesc descriptors)
into a single design matrix.  Performs an exact Bemis-Murcko scaffold
split (no data leakage), then trains XGBoost with a random-search
hyperparameter sweep.  Outputs per-strain metrics (AUC-ROC, AUC-PR,
MCC, F1, …), a predictions CSV, and feature-importance tables.

Usage:
    python build_train.py \\
        --labels        data/prestwick_library_screened.tsv.gz \\
        --mole          data/mole_embeddings.tsv \\
        --gpt           data/gpt_feature_generation/gpt_features_4o_gpt-4.1_unified.csv \\
        --chemdesc      data/maier_chemdesc_representation.tsv \\
        --output_dir    data/results/mole_plus_gpt_4.1

Dependencies: xgboost, scikit-learn, rdkit, pandas, numpy
"""

import os
import re
import gzip
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import random

from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score,
    balanced_accuracy_score, matthews_corrcoef
)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# CHANGE: no longer using GroupShuffleSplit for final paper-matching split
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold


# ----------------------------- I/O HELPERS -----------------------------

def read_csv_robust(path: str, sep: str = ",", compression="infer", force_str: bool = True) -> pd.DataFrame:
    """Robust CSV/TSV reader with fallbacks to avoid dtype conversion bugs."""
    dtype = str if force_str else None
    try:
        if sep == "\t":
            try:
                return pd.read_csv(path, sep=sep, engine="pyarrow")
            except Exception:
                pass
        return pd.read_csv(path, sep=sep, compression=compression, dtype=dtype, low_memory=False)
    except Exception:
        try:
            return pd.read_csv(
                path, sep=sep, compression=compression, dtype=dtype,
                low_memory=False, engine="python", on_bad_lines="skip", keep_default_na=False
            )
        except Exception:
            if str(path).endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
                    return pd.read_csv(
                        f, sep=sep, dtype=dtype, low_memory=False,
                        engine="python", on_bad_lines="skip", keep_default_na=False
                    )
            raise


def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower()


def find_prestwick_id_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.lower().strip() == "prestwick_id":
            return c
    for c in df.columns:
        if "prestw" in c.lower():
            return c
    raise KeyError("Could not find prestwick_ID column.")


# ----------------------- SF3 LABELS: BUILD BINARY -----------------------

def load_sf3_binary_matrix(sf3_path: str, pval_cutoff: float = 0.05) -> pd.DataFrame:
    """
    Load Supplementary File 3 and return a binary matrix (rows=prestwick_ID, cols=strains).
    Prefers Excel sheet 'S3a. Adjusted p-values' if xlsx, else will read TSV if provided.
    """
    if not os.path.exists(sf3_path):
        raise FileNotFoundError(f"SF3 path does not exist: {sf3_path}")

    _, ext = os.path.splitext(sf3_path)
    if ext.lower() in (".tsv", ".gz"):
        df = read_csv_robust(sf3_path, sep="\t")
        id_col = find_prestwick_id_col(df)
        df = df.set_index(id_col)
        vals = df.apply(pd.to_numeric, errors="coerce")
        if np.nanmax(vals.values.astype(float)) <= 1.0:
            bin_mat = (vals <= pval_cutoff).astype(int)
        else:
            bin_mat = vals.fillna(0).astype(int)
        bin_mat.index.name = "prestwick_ID"
        return bin_mat

    try:
        import openpyxl
    except Exception as e:
        raise RuntimeError("openpyxl is required for reading SF3 Excel files. Please install it.") from e

    wb = openpyxl.load_workbook(sf3_path, data_only=True, read_only=True)
    sheet_name = "S3a. Adjusted p-values"
    if sheet_name not in wb.sheetnames:
        raise KeyError(f"Sheet '{sheet_name}' not found in {sf3_path}. Sheets: {wb.sheetnames}")

    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header = [h if h is not None else "" for h in rows[0]]
    data = [list(r) for r in rows[1:]]
    df = pd.DataFrame(data, columns=header)

    for c in ("chemical_name", "drug_class", "n_hit"):
        if c in df.columns:
            df.drop(columns=c, inplace=True)

    if "prestwick_ID" not in df.columns:
        raise KeyError("Column 'prestwick_ID' not found in SF3 sheet.")

    df = df.set_index("prestwick_ID")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    bin_mat = (df <= pval_cutoff).astype(int)
    bin_mat.index.name = "prestwick_ID"
    return bin_mat


def labels_to_pairs(bin_mat: pd.DataFrame) -> pd.DataFrame:
    """Convert wide binary matrix to long (prestwick_ID, bacterium, label)."""
    pairs = bin_mat.reset_index().melt(
        id_vars=["prestwick_ID"], var_name="bacterium", value_name="label"
    )
    return pairs


# -------------------- MOLE FEATURES --------------------

def compute_or_load_mole(prestwick_screened_tsv: str,
                         mole_mode: str,
                         mole_tsv_out: str = None,
                         mole_model_dir: str = None) -> pd.DataFrame | None:
    """
    mole_mode: 'skip' | 'compute' | 'load'
      - skip: return None
      - compute: compute reps from SMILES
      - load: read precomputed TSV
    Returns DataFrame indexed by prestwick_ID with columns mole_*
    """
    if mole_mode == "skip":
        print("[MolE] Skipping MolE features.")
        return None

    if mole_mode == "load":
        if not mole_tsv_out or not os.path.exists(mole_tsv_out):
            raise FileNotFoundError("mole_mode=load requires --mole_tsv pointing to an existing file.")
        mole_df = read_csv_robust(mole_tsv_out, sep="\t")
        id_col = find_prestwick_id_col(mole_df)
        mole_df = mole_df.set_index(id_col)
        mole_df.columns = [c if str(c).startswith("mole_") else f"mole_{i}" for i, c in enumerate(mole_df.columns)]
        print(f"[MolE] Loaded precomputed MolE: {mole_tsv_out}  shape={mole_df.shape}")
        return mole_df

    if mole_mode == "compute":
        if not os.path.exists(prestwick_screened_tsv):
            raise FileNotFoundError(f"prestwick_screened TSV not found: {prestwick_screened_tsv}")
        try:
            try:
                from mole_representation import process_dataset as _proc
                mode = "dataset"
            except Exception:
                from mole_representation import process_representation as _proc
                mode = "representation"

            print(f"[MolE] Computing MolE using process_{mode}(...). This may take a while.")
            kwargs = dict(
                dataset_path=prestwick_screened_tsv,
                smile_column_str="rdkit_no_salt",
                id_column_str="prestwick_ID",
            )
            if mode == "dataset":
                kwargs.update(
                    pretrain_architecture="gin_concat",
                    pretrained_model="model_ginconcat_btwin_100k_d8000_l0.0001",
                    split_approach="scaffold",
                    validation_proportion=0.1,
                    test_proportion=0.1,
                )
            else:
                if not mole_model_dir:
                    raise ValueError("mole_mode=compute requires --mole_model_dir for process_representation.")
                kwargs.update(pretrained_dir=mole_model_dir)

            res = _proc(**kwargs)
            if isinstance(res, tuple) and len(res) >= 2:
                _, mole_df = res[0], res[1]
            else:
                mole_df = res

            if "prestwick_ID" in mole_df.columns:
                mole_df = mole_df.set_index("prestwick_ID")

            mole_df = mole_df.copy()
            mole_df.columns = [f"mole_{i}" for i in range(mole_df.shape[1])]

            if mole_tsv_out:
                mole_df.to_csv(mole_tsv_out, sep="\t", index=True, compression="gzip")
                print(f"[MolE] Saved MolE TSV: {mole_tsv_out}  shape={mole_df.shape}")

            return mole_df

        except Exception as e:
            warnings.warn(f"[MolE] Could not compute MolE: {e}. Proceeding without MolE.")
            return None

    raise ValueError("mole_mode must be one of: skip | compute | load")


# -------------------- GPT FEATURES --------------------

def load_gpt_features(gpt_csv: str) -> pd.DataFrame:
    if not os.path.exists(gpt_csv):
        raise FileNotFoundError(f"GPT features CSV not found: {gpt_csv}")
    sep = "\t" if gpt_csv.endswith(".tsv") or gpt_csv.endswith(".tsv.gz") else ","
    gpt = read_csv_robust(gpt_csv, sep=sep)
    id_col = find_prestwick_id_col(gpt)
    gpt = gpt.rename(columns={id_col: "prestwick_ID"})
    gpt = gpt.drop_duplicates(subset=["prestwick_ID"], keep="first").set_index("prestwick_ID")
    return gpt


# -------------------- CHEMDESC FEATURES --------------------
# CHANGE: added ChemDesc support

def load_chemdesc_features(chemdesc_csv: str) -> pd.DataFrame:
    if not os.path.exists(chemdesc_csv):
        raise FileNotFoundError(f"ChemDesc features file not found: {chemdesc_csv}")

    sep = "\t" if chemdesc_csv.endswith(".tsv") or chemdesc_csv.endswith(".tsv.gz") else ","
    df = read_csv_robust(chemdesc_csv, sep=sep)

    id_col = find_prestwick_id_col(df)
    df = df.rename(columns={id_col: "prestwick_ID"})
    df = df.drop_duplicates(subset=["prestwick_ID"], keep="first").set_index("prestwick_ID")

    df.columns = [c if str(c).startswith("chemdesc_") else f"chemdesc_{i}" for i, c in enumerate(df.columns)]
    return df


# -------------------- FEATURE COMBINATION --------------------
# CHANGE: generalized to support GPT / MolE / ChemDesc

def combine_feature_blocks(feature_blocks: list[pd.DataFrame]) -> pd.DataFrame:
    blocks = [df for df in feature_blocks if df is not None]
    if not blocks:
        raise ValueError("No feature blocks provided.")

    common = blocks[0].index
    for df in blocks[1:]:
        common = common.intersection(df.index)

    combined = pd.concat([df.loc[common] for df in blocks], axis=1)
    return combined


# -------------------- BUILD TRAINING TABLE --------------------

def build_pairs_with_features(pairs: pd.DataFrame, feat_df: pd.DataFrame) -> pd.DataFrame:
    pairs = pairs.copy()
    pairs["_join_id"] = normalize_id_series(pairs["prestwick_ID"])

    idx = feat_df.reset_index()[["prestwick_ID"]].copy()
    idx["_join_id"] = normalize_id_series(idx["prestwick_ID"])

    pairs_feat = (pairs.merge(idx, on="_join_id", how="inner")
                       .drop(columns=["prestwick_ID_y"])
                       .rename(columns={"prestwick_ID_x": "prestwick_ID"}))

    feat_df2 = feat_df.reset_index().copy()
    feat_df2["_join_id"] = normalize_id_series(feat_df2["prestwick_ID"])
    feat_df2 = feat_df2.drop_duplicates(subset=["_join_id"], keep="first")
    feat_df2 = feat_df2.drop(columns=["prestwick_ID"])

    joined = pairs_feat.merge(feat_df2, on="_join_id", how="inner").drop(columns=["_join_id"])
    return joined


# -------------------- EXACT SPLIT SUPPORT --------------------
# CHANGE: added support for their exact split

def load_exact_split(split_path: str) -> pd.DataFrame:
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    sep = "\t" if split_path.endswith(".tsv") or split_path.endswith(".tsv.gz") else ","
    df = read_csv_robust(split_path, sep=sep)

    id_col = find_prestwick_id_col(df)
    df = df.rename(columns={id_col: "prestwick_ID"})

    split_col = None
    for c in df.columns:
        if c.lower().strip() in {"split", "set", "partition"}:
            split_col = c
            break
    if split_col is None:
        raise KeyError(f"No split column found in {split_path}. Columns={df.columns.tolist()}")

    df = df[["prestwick_ID", split_col]].copy()
    df = df.rename(columns={split_col: "split"})
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    df["split"] = df["split"].replace({
        "validation": "val",
        "valid": "val",
        "dev": "val"
    })

    return df.drop_duplicates(subset=["prestwick_ID"])


def attach_split_to_pairs(joined: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    joined = joined.copy()
    joined["_join_id"] = normalize_id_series(joined["prestwick_ID"])

    split_df = split_df.copy()
    split_df["_join_id"] = normalize_id_series(split_df["prestwick_ID"])
    split_df = split_df.drop_duplicates(subset=["_join_id"], keep="first")

    joined = joined.merge(
        split_df[["_join_id", "split"]],
        on="_join_id",
        how="inner"
    ).drop(columns=["_join_id"])

    return joined


# -------------------- METRICS --------------------

def best_f1_threshold(y_true, scores):
    ps, rs, ths = precision_recall_curve(y_true, scores)
    f1s = 2 * ps[:-1] * rs[:-1] / np.clip(ps[:-1] + rs[:-1], 1e-12, None)
    i = int(np.nanargmax(f1s))
    return float(ths[i]), float(f1s[i])


def compute_metrics(y_true, scores, thr):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = (scores >= thr).astype(int)

    out = {}
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, scores))
    except Exception:
        out["pr_auc"] = float("nan")

    out["threshold"] = float(thr)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = float(
        (recall_score(y_true, y_pred, pos_label=1, zero_division=0) +
         recall_score(y_true, y_pred, pos_label=0, zero_division=0)) / 2.0
    )
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred)) if (y_pred.std() > 0 and y_true.std() > 0) else 0.0
    return out


def per_strain_metrics(df_split, y_true, scores, thr):
    rows = []
    df_ = df_split.reset_index(drop=True).copy()

    for bug, sub in df_.groupby("bacterium"):
        idx = sub.index.values
        y = y_true[idx]
        s = scores[idx]

        try:
            auc = roc_auc_score(y, s)
            aupr = average_precision_score(y, s)
        except Exception:
            auc, aupr = np.nan, np.nan

        f1b = f1_score(y, (s >= thr).astype(int), zero_division=0)

        rows.append({
            "bacterium": bug,
            "n": int(len(y)),
            "pos_rate": float(y.mean()),
            "roc_auc": float(auc) if not np.isnan(auc) else np.nan,
            "pr_auc": float(aupr) if not np.isnan(aupr) else np.nan,
            "f1_at_global_thr": float(f1b),
        })

    out = pd.DataFrame(rows).sort_values("bacterium")
    macro_pr_auc = float(out["pr_auc"].mean(skipna=True))
    return out, macro_pr_auc


# -------------------- XGBOOST PARAMS --------------------

PARAM_SPACE = {
    "eta": [0.1, 0.05, 0.03],
    "max_depth": [5, 8, 10],
    "subsample": [0.5, 0.8, 1.0],
    "colsample_bytree": [0.5, 0.8, 1.0],
    "min_child_weight": [1, 5],
    "gamma": [0.0, 0.1]
}

BASE_PARAMS = {
    "tree_method": "hist", "max_bin": 512,
    "lambda": 1.0, "alpha": 0.0,
    "objective": "binary:logistic", "eval_metric": "aucpr", "verbosity": 0,
}


def sample_params():
    p = BASE_PARAMS.copy()
    for k, v in PARAM_SPACE.items():
        p[k] = random.choice(v)
    return p


# -------------------- TRAIN --------------------
# CHANGE: now uses exact split instead of generating its own scaffold split

def train_xgb(joined: pd.DataFrame, params: dict, random_state: int = 42):

    joined = joined.copy()

    # CHANGE: use exact split already attached
    if "split" not in joined.columns:
        raise KeyError("Expected column 'split' in joined dataframe.")

    df_tr = joined[joined["split"] == "train"].copy()
    df_val = joined[joined["split"] == "val"].copy()
    df_te = joined[joined["split"] == "test"].copy()

    if len(df_tr) == 0 or len(df_val) == 0 or len(df_te) == 0:
        raise ValueError("Train/val/test split is empty. Check split file.")

    def _rate(y):
        y = np.asarray(y)
        return float(y.mean())

    print(f"[Split] pos_rate train={_rate(df_tr['label']):.3f}  "
          f"val={_rate(df_val['label']):.3f}  test={_rate(df_te['label']):.3f}")

    cat_col = ["bacterium"]

    # CHANGE: split is now a protected non-feature column
    non_feat = {"prestwick_ID", "bacterium", "label", "split"}

    def _num_cols(df):
        return [c for c in df.columns if c not in non_feat and pd.api.types.is_numeric_dtype(df[c])]

    num_tr = _num_cols(df_tr)

    # CHANGE: now detect chemdesc too
    mole_cols = [c for c in num_tr if str(c).startswith("mole_")]
    chemdesc_cols = [c for c in num_tr if str(c).startswith("chemdesc_")]
    gpt_cols = [
        c for c in num_tr
        if c not in mole_cols + chemdesc_cols
        and c != "bug_prior_train"
        and not str(c).startswith(("identifiers.", "input."))
    ]

    MIN_NONZERO_FRAC = 0.01
    gpt_cols = [c for c in gpt_cols if (df_tr[c] != 0).mean() >= MIN_NONZERO_FRAC]

    for frame in (df_tr, df_val, df_te):
        for c in mole_cols + chemdesc_cols:
            if frame[c].dtype == bool:
                frame[c] = frame[c].astype(int)

    y_tr = df_tr["label"].astype(int).values
    y_val = df_val["label"].astype(int).values
    y_te = df_te["label"].astype(int).values

    # CHANGE: add chemdesc block to preprocessor
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_col),

        ("mole", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())
        ]), mole_cols),

        ("chemdesc", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler())
        ]), chemdesc_cols),

        ("gpt", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value=0.0)),
        ]), gpt_cols)
    ])

    # CHANGE: include chemdesc columns
    FEATURE_COLS = cat_col + mole_cols + chemdesc_cols + gpt_cols
    preproc.fit(df_tr[FEATURE_COLS], y_tr)
        # CHANGE: save exact raw feature groups used before preprocessing
    raw_feature_summary = {
        "categorical_features": cat_col,
        "mole_features": mole_cols,
        "chemdesc_features": chemdesc_cols,
        "gpt_features": gpt_cols,
        "all_raw_features": FEATURE_COLS
    }

    print("\n[FEATURE SUMMARY BEFORE PREPROCESSING]")
    print(f" - categorical: {len(cat_col)}")
    print(f" - mole: {len(mole_cols)}")
    print(f" - chemdesc: {len(chemdesc_cols)}")
    print(f" - gpt: {len(gpt_cols)}")
    print(f" - total raw features: {len(FEATURE_COLS)}")

    X_tr = preproc.transform(df_tr[FEATURE_COLS]).astype(np.float32)
    X_val = preproc.transform(df_val[FEATURE_COLS]).astype(np.float32)
    X_te = preproc.transform(df_te[FEATURE_COLS]).astype(np.float32)
        # CHANGE: capture exact transformed feature names that XGBoost sees
    transformed_feature_names = list(preproc.get_feature_names_out())

    print("\n[FEATURE SUMMARY AFTER PREPROCESSING]")
    print(f" - transformed feature count sent to XGBoost: {len(transformed_feature_names)}")
    print(" - first 20 transformed features:")
    print(transformed_feature_names[:20])

    pos_rate = float(y_tr.mean())
    params = params.copy()
    params["scale_pos_weight"] = (1 - pos_rate) / pos_rate
    params["max_delta_step"] = 1

    dtr  = xgb.DMatrix(X_tr,  label=y_tr)
    dval = xgb.DMatrix(X_val, label=y_val)
    dte  = xgb.DMatrix(X_te,  label=y_te)

    booster = xgb.train(
        params,
        dtr,
        num_boost_round=3000,
        evals=[(dval, "val")],
        early_stopping_rounds=300,
        verbose_eval=False,
    )

    val_pred = booster.predict(dval)

    val_ap = average_precision_score(y_val, val_pred)
    val_base = float(y_val.mean())
    print(f"[Val] base_pos_rate={val_base:.3f}, norm_AP={val_ap/max(val_base,1e-9):.2f}x")
    print(f"[Val] ensemble AUPRC={val_ap:.5f}")

    thr, f1v = best_f1_threshold(y_val, val_pred)
    print(f"[Val] best F1={f1v:.4f} @ thr={thr:.4f}")

    te_score = booster.predict(dte)
    te_ap = average_precision_score(y_te, te_score)
    te_base = float(y_te.mean())
    print(f"[Test] AUPRC={te_ap:.5f}  base_pos_rate={te_base:.3f}, norm_AP={te_ap/max(te_base,1e-9):.2f}x")

    overall = compute_metrics(y_te, te_score, thr)
    print(f"[Test] F1={overall['f1']:.4f} @ thr={thr:.4f}")
    print(f"[Test] Precision={overall['precision']:.4f}")
    print(f"[Test] Recall={overall['recall']:.4f}")

    per_strain_val_df, macro_pr_auc_val = per_strain_metrics(df_val, y_val, val_pred, thr)
    per_strain_df, macro_pr_auc_test = per_strain_metrics(df_te, y_te, te_score, thr)

    df_te_ = df_te.reset_index(drop=True).copy()

    summary = {
        **overall,
        "val_best_f1": f1v,
        "val_best_threshold": thr,
        "val_macro_pr_auc_by_strain": macro_pr_auc_val,
        "test_macro_pr_auc_by_strain": macro_pr_auc_test,
        "n_train": int(len(df_tr)),
        "n_val": int(len(df_val)),
        "n_test": int(len(df_te)),
        "n_features": int(X_tr.shape[1])
    }

    print("\n================ PAPER METRIC ================")
    print(f"Mean per-strain AUPRC (paper-style): {macro_pr_auc_test:.4f} and accuracy: {overall['accuracy']}")
    print(f"Test F1: {overall['f1']:.4f}")
    print("================================================\n")

    preds = df_te_[["prestwick_ID", "bacterium", "label"]].copy()
    preds["proba"] = te_score
    preds["pred_f1"] = (preds["proba"] >= thr).astype(int)
    preds["pred_0044"] = (preds["proba"] >= 0.044).astype(int)

    model = {"preproc": preproc, "booster": booster, "threshold": thr}

    feature_debug = {
        "raw_feature_summary": raw_feature_summary,
        "transformed_feature_names": transformed_feature_names
    }

    return model, summary, preds, per_strain_df, dte, y_te, feature_debug


# ----------------------------- AP-G SCORING -----------------------------

_STRAINS_40 = [
    "Akkermansia muciniphila (NT5021)", "Bacteroides caccae (NT5050)",
    "Bacteroides fragilis (ET) (NT5033)", "Bacteroides fragilis (NT) (NT5003)",
    "Bacteroides ovatus (NT5054)", "Bacteroides thetaiotaomicron (NT5004)",
    "Bacteroides uniformis (NT5002)", "Bacteroides vulgatus (NT5001)",
    "Bacteroides xylanisolvens (NT5064)", "Bifidobacterium adolescentis (NT5022)",
    "Bifidobacterium longum (NT5028)", "Bilophila wadsworthia (NT5036)",
    "Blautia obeum (NT5069)", "Clostridium bolteae (NT5026)",
    "Clostridium difficile (NT5083)", "Clostridium perfringens (NT5032)",
    "Clostridium ramosum (NT5006)", "Clostridium saccharolyticum (NT5037)",
    "Collinsella aerofaciens (NT5073)", "Coprococcus comes (NT5048)",
    "Dorea formicigenerans (NT5076)", "Eggerthella lenta (NT5024)",
    "Escherichia coli ED1a (NT5078)", "Escherichia coli IAI1 (NT5077)",
    "Eubacterium eligens (NT5075)", "Eubacterium rectale (NT5009)",
    "Fusobacterium nucleatum (NT5025)", "Lactobacillus paracasei (NT5042)",
    "Odoribacter splanchnicus (NT5081)", "Parabacteroides distasonis (NT5074)",
    "Parabacteroides merdae (NT5071)", "Prevotella copri (NT5019)",
    "Roseburia hominis (NT5079)", "Roseburia intestinalis (NT5011)",
    "Ruminococcus bromii (NT5045)", "Ruminococcus gnavus (NT5046)",
    "Ruminococcus torques (NT5047)", "Streptococcus parasanguinis (NT5072)",
    "Streptococcus salivarius (NT5038)", "Veillonella parvula (NT5017)",
]


def compute_ap_scores(model_pkl_path: str,
                      gpt_features_csv: str,
                      id_col: str = "ProductName",
                      strains: list = None,
                      output_csv: str = None,
                      mole_embedding_tsv: str = None,
                      mole_id_map_tsv: str = None,
                      mole_map_left_col: str = "ProductName",
                      mole_map_right_col: str = "Catalog Number"):
    """
    Load a trained XGBoost pkl and compute AP-G scores for new compounds.

    AP-G = ln(mean predicted probability across all 40 strains).
    Saves a CSV with columns: <id_col>, ap_score_gpt.

    Optional MolE args (for mole_plus_gpt models):
      mole_embedding_tsv  : TSV with raw MolE embeddings (index = catalog/ID, cols = 0..999).
      mole_id_map_tsv     : TSV/CSV mapping id_col values to embedding index (e.g. ProductName → Catalog Number).
      mole_map_left_col   : column in id_map that matches id_col values (default "ProductName").
      mole_map_right_col  : column in id_map that matches embedding index (default "Catalog Number").
    """
    import joblib

    if strains is None:
        strains = _STRAINS_40

    print(f"\n[AP-G] Loading model from {model_pkl_path}")
    bundle  = joblib.load(model_pkl_path)
    model   = bundle["model"]
    preproc = model["preproc"]
    booster = model["booster"]

    # Patch sklearn 1.7→1.8 rename: _fit_dtype → _fill_dtype in SimpleImputer
    from sklearn.impute import SimpleImputer as _SI
    from sklearn.pipeline import Pipeline as _Pipe
    def _patch_si(obj):
        if isinstance(obj, _SI) and hasattr(obj, "_fit_dtype") and not hasattr(obj, "_fill_dtype"):
            obj._fill_dtype = obj._fit_dtype
        if isinstance(obj, _Pipe):
            for _, step in obj.steps:
                _patch_si(step)
    for _, tr, _ in preproc.transformers_:
        _patch_si(tr)

    print(f"[AP-G] Loading compound features from {gpt_features_csv}")
    feat_df = pd.read_csv(gpt_features_csv)
    if id_col not in feat_df.columns:
        raise ValueError(f"Column '{id_col}' not found in {gpt_features_csv}")

    compound_ids = feat_df[id_col].tolist()

    # Load MolE embeddings if provided
    mole_lookup = {}  # compound_id -> dict of {mole_0: val, mole_1: val, ...}
    if mole_embedding_tsv:
        print(f"[AP-G] Loading MolE embeddings from {mole_embedding_tsv}")
        mole_emb = pd.read_csv(mole_embedding_tsv, sep="\t", index_col=0)
        # rename integer columns to mole_<i>
        mole_emb.columns = [f"mole_{i}" for i in range(len(mole_emb.columns))]

        # build id → embedding index mapping
        if mole_id_map_tsv:
            sep = "\t" if mole_id_map_tsv.endswith((".tsv", ".tsv.gz", ".gz")) else ","
            id_map = pd.read_csv(mole_id_map_tsv, sep=sep, usecols=[mole_map_left_col, mole_map_right_col])
            name_to_key = dict(zip(id_map[mole_map_left_col], id_map[mole_map_right_col]))
        else:
            name_to_key = {cid: cid for cid in compound_ids}

        for cid in compound_ids:
            key = name_to_key.get(cid)
            if key is not None and key in mole_emb.index:
                mole_lookup[cid] = mole_emb.loc[key].to_dict()
            else:
                print(f"[AP-G][WARN] No MolE embedding for '{cid}', using zeros.")

        n_found = sum(1 for c in compound_ids if c in mole_lookup)
        print(f"[AP-G] MolE embeddings found for {n_found}/{len(compound_ids)} compounds.")

    # Infer which feature columns the preproc expects from its transformers
    ct = preproc
    all_expected = []
    for name, _, cols in ct.transformers_:
        if name != "remainder":
            all_expected.extend(list(cols))

    # Build pairs: each compound × each strain
    rows = []
    for cid in compound_ids:
        comp_row = feat_df[feat_df[id_col] == cid].iloc[0]
        mole_vals = mole_lookup.get(cid, {})
        for strain in strains:
            r = {id_col: cid, "bacterium": strain}
            for c in all_expected:
                if c == "bacterium":
                    continue
                # MolE column: use real embedding if available
                if str(c).startswith("mole_") and c in mole_vals:
                    r[c] = mole_vals[c]
                elif c in comp_row.index:
                    try:
                        r[c] = float(comp_row[c])
                    except (ValueError, TypeError):
                        r[c] = 0.0
                else:
                    r[c] = 0.0
            rows.append(r)

    pairs = pd.DataFrame(rows)
    feature_cols = [c for c in all_expected]
    for c in feature_cols:
        if c not in pairs.columns:
            pairs[c] = 0.0
    num_cols = [c for c in feature_cols if c != "bacterium"]
    pairs[num_cols] = pairs[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0)

    print(f"[AP-G] Running inference on {len(compound_ids)} compounds × {len(strains)} strains …")
    X     = preproc.transform(pairs[feature_cols]).astype(np.float32)
    proba = booster.predict(xgb.DMatrix(X))

    proba_mat  = proba.reshape(len(compound_ids), len(strains))
    # AP-G = log of geometric mean = mean of log(p) across strains
    ap_g       = np.log(proba_mat + 1e-12).mean(axis=1)

    result_df = pd.DataFrame({id_col: compound_ids, "ap_score_gpt": ap_g})
    result_df.to_csv(output_csv, index=False)
    print(f"[AP-G] Saved {len(result_df)} AP-G scores to {output_csv}")
    print(f"[AP-G] AP-G range: [{ap_g.min():.3f}, {ap_g.max():.3f}]")
    return result_df


# ----------------------------- CLI -----------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Build SF3 labels, combine GPT / MolE / ChemDesc features, and train XGBoost.")

    ap.add_argument("--sf3", required=True, help="Path to SF3 file (xlsx with S3a sheet, or TSV/TSV.GZ).")
    ap.add_argument("--exact_split", required=True, help="Path to exact split file with prestwick_ID + split.")
    ap.add_argument("--gpt_features", default=None, help="Path to flattened GPT features CSV.")
    ap.add_argument("--chemdesc_features", default=None, help="Path to ChemDesc features file.")

    ap.add_argument("--prestwick_screened", default=None, help="Path to prestwick_library_screened.tsv.gz (for MolE compute).")
    ap.add_argument("--mole_mode", choices=["skip", "compute", "load"], default="skip",
                    help="MolE features: 'skip', 'compute', or 'load' from --mole_tsv.")
    ap.add_argument("--mole_tsv", default=None, help="Path to precomputed MolE TSV (when --mole_mode load).")
    ap.add_argument("--mole_model_dir", default=None, help="Pretrained MolE model dir (when --mole_mode compute).")

    ap.add_argument("--pval_cutoff", type=float, default=0.05, help="P-value cutoff for labels.")
    ap.add_argument("--output_dir", required=True, help="Directory to write outputs.")
    ap.add_argument("--random_state", type=int, default=42, help="Random state.")

    # CHANGE: expanded feature modes
    ap.add_argument(
        "--feature_mode",
        choices=[
            "gpt_only",
            "mole_only",
            "chemdesc_only",
            "mole_plus_gpt",
            "chemdesc_plus_gpt",
            "mole_plus_chemdesc",
            "mole_plus_chemdesc_plus_gpt"
        ],
        default="mole_plus_gpt",
        help="Which feature set to use."
    )

    ap.add_argument("--ap_score_compounds", default=None,
                    help="GPT features CSV for new compounds (e.g. MCE library). "
                         "When provided, compute AP-G scores after training and save to --ap_score_output.")
    ap.add_argument("--ap_score_id_col", default="ProductName",
                    help="ID column name in --ap_score_compounds CSV (default: ProductName).")
    ap.add_argument("--ap_score_output", default=None,
                    help="Output path for AP-G scores CSV (default: <output_dir>/ap_scores.csv).")
    ap.add_argument("--ap_score_mole_tsv", default=None,
                    help="TSV with raw MolE embeddings for new compounds (cols = 0..999, index = compound key).")
    ap.add_argument("--ap_score_mole_id_map", default=None,
                    help="TSV/CSV mapping ap_score_id_col values to the MolE embedding index (e.g. ProductName→Catalog Number).")
    ap.add_argument("--ap_score_mole_map_left", default="ProductName",
                    help="Column in --ap_score_mole_id_map that matches ap_score_id_col (default: ProductName).")
    ap.add_argument("--ap_score_mole_map_right", default="Catalog Number",
                    help="Column in --ap_score_mole_id_map that matches the MolE embedding index (default: Catalog Number).")

    return ap.parse_args()


# ----------------------------- MAIN -----------------------------

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1] Loading SF3 and building binary labels.")
    labels_wide = load_sf3_binary_matrix(args.sf3, pval_cutoff=args.pval_cutoff)
    pairs = labels_to_pairs(labels_wide)
    print(f" - labels matrix: {labels_wide.shape}, pairs: {pairs.shape}")

    gpt_df = None
    chemdesc_df = None
    mole_df = None

    # -------------------- GPT --------------------
    if args.feature_mode in {
        "gpt_only", "mole_plus_gpt", "chemdesc_plus_gpt", "mole_plus_chemdesc_plus_gpt"
    }:
        if not args.gpt_features:
            raise ValueError("--gpt_features must be provided for this feature_mode.")
        print("[2] Loading GPT features.")
        gpt_df = load_gpt_features(args.gpt_features)
        print(f" - GPT features: {gpt_df.shape}")

        # CHANGE: remove ID-like columns from GPT
        _BLACKLIST_PREFIXES = ("identifiers.", "input.")
        _BLACKLIST_EXACT = {"identifiers.inchi", "identifiers.inchikey", "input.parsed"}

        drop_cols = [
            c for c in gpt_df.columns
            if c in _BLACKLIST_EXACT or any(str(c).startswith(p) for p in _BLACKLIST_PREFIXES)
        ]
        if drop_cols:
            print(f"[GPT] Dropping ID-like columns: {drop_cols[:20]}")
            gpt_df = gpt_df.drop(columns=drop_cols, errors="ignore")

    # -------------------- CHEMDESC --------------------
    if args.feature_mode in {
        "chemdesc_only", "chemdesc_plus_gpt", "mole_plus_chemdesc", "mole_plus_chemdesc_plus_gpt"
    }:
        if not args.chemdesc_features:
            raise ValueError("--chemdesc_features must be provided for this feature_mode.")
        print("[2b] Loading ChemDesc features.")
        chemdesc_df = load_chemdesc_features(args.chemdesc_features)
        print(f" - ChemDesc features: {chemdesc_df.shape}")

    # -------------------- MOLE --------------------
    if args.feature_mode in {
        "mole_only", "mole_plus_gpt", "mole_plus_chemdesc", "mole_plus_chemdesc_plus_gpt"
    }:
        print("[2c] Loading / computing MolE features.")
        mole_df = compute_or_load_mole(
            prestwick_screened_tsv=args.prestwick_screened,
            mole_mode=args.mole_mode,
            mole_tsv_out=args.mole_tsv,
            mole_model_dir=args.mole_model_dir,
        )
        if mole_df is not None:
            print(f" - MolE features: {mole_df.shape}")
        else:
            # mole_mode=skip was used but feature_mode requires MolE → fail loudly
            mole_required_modes = {"mole_only", "mole_plus_gpt", "mole_plus_chemdesc", "mole_plus_chemdesc_plus_gpt"}
            if args.feature_mode in mole_required_modes:
                raise ValueError(
                    f"feature_mode='{args.feature_mode}' requires MolE features, but --mole_mode is 'skip'.\n"
                    "Add:  --mole_mode load --mole_tsv <path/to/mole.tsv>"
                )

    print("[3] Feature mode:", args.feature_mode)

    # CHANGE: generalized feature block combination
    if args.feature_mode == "gpt_only":
        combined = combine_feature_blocks([gpt_df])

    elif args.feature_mode == "mole_only":
        combined = combine_feature_blocks([mole_df])

    elif args.feature_mode == "chemdesc_only":
        combined = combine_feature_blocks([chemdesc_df])

    elif args.feature_mode == "mole_plus_gpt":
        combined = combine_feature_blocks([mole_df, gpt_df])

    elif args.feature_mode == "chemdesc_plus_gpt":
        combined = combine_feature_blocks([chemdesc_df, gpt_df])

    elif args.feature_mode == "mole_plus_chemdesc":
        combined = combine_feature_blocks([mole_df, chemdesc_df])

    elif args.feature_mode == "mole_plus_chemdesc_plus_gpt":
        combined = combine_feature_blocks([mole_df, chemdesc_df, gpt_df])

    else:
        raise ValueError(f"Unsupported feature_mode: {args.feature_mode}")

    print(f" - Combined features: {combined.shape}")

    combined_out = os.path.join(args.output_dir, f"compound_features_combined_{args.feature_mode}.csv")
    combined.reset_index().to_csv(combined_out, index=False)
    print(f" - Saved: {combined_out}")

    print("[4] Merging pairs with features.")
    joined = build_pairs_with_features(pairs, combined)
    print(f" - Pairs with features: {joined.shape}")

    # CHANGE: attach exact split
    print("[5] Loading exact split file.")
    split_df = load_exact_split(args.exact_split)
    print(f" - Exact split: {split_df.shape}")

    split_out = os.path.join(args.output_dir, f"exact_split_used_{args.feature_mode}.csv")
    split_df.to_csv(split_out, index=False)
    print(f" - Saved exact split copy: {split_out}")

    joined = attach_split_to_pairs(joined, split_df)
    print(f" - Joined with split: {joined.shape}")
    print(joined["split"].value_counts(dropna=False))

    # CHANGE: sanity checks
    print("\n[SANITY CHECKS]")
    print("Unique compounds per split:")
    print(joined.groupby("split")["prestwick_ID"].nunique())

    print("\nRows per split:")
    print(joined["split"].value_counts())

    leakage = joined.groupby("prestwick_ID")["split"].nunique().max()
    print(f"\nMax number of splits per compound: {leakage}")
    if leakage > 1:
        raise ValueError("Data leakage detected: some compounds appear in multiple splits.")

    # Ensure exactly one prestwick_ID column
    if "prestwick_ID" not in joined.columns:
        if {"prestwick_ID_x", "prestwick_ID_y"} <= set(joined.columns):
            same = (
                joined["prestwick_ID_x"].astype(str).str.lower()
                == joined["prestwick_ID_y"].astype(str).str.lower()
            )
            if not same.all():
                print("[WARN] prestwick_ID_x and prestwick_ID_y differ for some rows; keeping _x.")
            joined = joined.drop(columns=["prestwick_ID_y"]).rename(columns={"prestwick_ID_x": "prestwick_ID"})
        else:
            alt = next((c for c in joined.columns if c.lower() in ("prestwick_id", "prestwick_id_x", "prestwick_id_y")), None)
            if alt:
                joined = joined.rename(columns={alt: "prestwick_ID"})
            else:
                raise KeyError(f"No prestwick_ID present. Columns={joined.columns.tolist()}")

    # CHANGE: split is now protected
    non_feat = {"prestwick_ID", "bacterium", "label", "split"}

    dead = [c for c in joined.columns
            if c not in non_feat
            and str(c).startswith("mole_")
            and joined[c].isna().all()]
    if dead:
        print("Dropping all-NaN MolE columns:", dead)
        joined = joined.drop(columns=dead)

    _NUM_PATTERN = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'
    YES = {"yes", "true", "y", "t", "1"}
    NO = {"no", "false", "n", "f", "0"}

    def _smart_to_numeric(s: pd.Series) -> pd.Series:
        if s.dtype == bool:
            return s.astype(int)

        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")

        s_num = pd.to_numeric(s, errors="coerce")
        if s_num.notna().any():
            return s_num

        extracted = s.astype(str).str.extract(_NUM_PATTERN, expand=False)
        s_num = pd.to_numeric(extracted, errors="coerce")
        if s_num.notna().any():
            return s_num

        low = s.astype(str).str.strip().str.lower()
        if low.isin(YES).any() or low.isin(NO).any():
            return low.isin(YES).astype(float)

        return pd.Series(np.nan, index=s.index, dtype="float64")

    for c in joined.columns:
        if c not in non_feat:
            joined[c] = _smart_to_numeric(joined[c])

    def _nuniq_after_round(a: pd.Series, ndigits=6) -> int:
        v = pd.to_numeric(a, errors="coerce")
        v = np.round(v.astype(float), ndigits)
        return pd.Series(v).nunique(dropna=True)

    const = [c for c in joined.columns
             if c not in non_feat
             and joined[c].isna().sum() == 0
             and _nuniq_after_round(joined[c]) <= 1]
    if const:
        print("Dropping constant feature columns:", const)
        joined = joined.drop(columns=const)

    pairs_feats_csv = os.path.join(args.output_dir, f"compound_strain_pairs_with_features_{args.feature_mode}.csv")
    joined.to_csv(pairs_feats_csv, index=False)
    print(f" - Saved: {pairs_feats_csv}")

    print("[6] Hyperparameter search + training with exact split.")

    # Seed both Python random and NumPy so that hyperparameter sampling and
    # XGBoost itself are fully reproducible (and vary across --random_state values).
    random.seed(args.random_state)
    np.random.seed(args.random_state)

    N_TRIALS = 5
    SEEDS = [0, 1, 2]

    best_val = -1
    best_params = None

    for trial in range(N_TRIALS):
        print(f"\n[SEARCH] Trial {trial+1}/{N_TRIALS}")
        params = sample_params()
        val_scores = []

        for seed in SEEDS:
            _, summary, _, _, _, _, _ = train_xgb(
                joined,
                params=params,
                random_state=seed
            )
            val_scores.append(summary["val_macro_pr_auc_by_strain"])

        mean_val = float(np.mean(val_scores))
        print(f"[SEARCH] Mean validation macro-AUPRC: {mean_val:.4f}")

        if mean_val > best_val:
            best_val = mean_val
            best_params = params

    print(f"[BEST] Validation macro-AUPRC: {best_val:.4f}")
    print(f"[BEST] Params: {best_params}")

    model, summary, preds, per_strain, dte, y_te, feature_debug = train_xgb(
        joined,
        params=best_params,
        random_state=args.random_state,
    )

    print("[FINAL] Test macro-AUPRC:", summary["test_macro_pr_auc_by_strain"])

    # Save full summary metrics as JSON so downstream scripts can aggregate them.
    summary_path = os.path.join(args.output_dir, f"summary_metrics_{args.feature_mode}.json")
    with open(summary_path, "w") as _f:
        json.dump(summary, _f, indent=2)
    print(f" - Saved summary metrics: {summary_path}")

        # -------------------- SAVE FEATURE LISTS USED --------------------
    print("[6b] Saving exact feature lists used by XGBoost...")

    # Raw feature groups
    raw_summary = feature_debug["raw_feature_summary"]

    raw_rows = []
    for group_name, feats in raw_summary.items():
        if isinstance(feats, list):
            for f in feats:
                raw_rows.append({"feature_group": group_name, "feature_name": f})

    raw_feature_df = pd.DataFrame(raw_rows)
    raw_feature_csv = os.path.join(args.output_dir, f"raw_features_used_{args.feature_mode}.csv")
    raw_feature_df.to_csv(raw_feature_csv, index=False)
    print(f" - Saved raw feature list: {raw_feature_csv}")

    # Transformed feature names actually seen by XGBoost
    transformed_feature_df = pd.DataFrame({
        "xgb_feature_index": [f"f{i}" for i in range(len(feature_debug["transformed_feature_names"]))],
        "feature_name": feature_debug["transformed_feature_names"]
    })

    transformed_feature_csv = os.path.join(args.output_dir, f"xgb_transformed_features_{args.feature_mode}.csv")
    transformed_feature_df.to_csv(transformed_feature_csv, index=False)
    print(f" - Saved transformed XGBoost feature list: {transformed_feature_csv}")
    # -------------------- FEATURE IMPORTANCE --------------------
    print("[7] Extracting feature importances...")
    best_model = model
    feature_names = list(best_model["preproc"].get_feature_names_out())
    raw_importances = best_model["booster"].get_score(importance_type="gain")
    importances = {feature_names[int(k[1:])]: v for k, v in raw_importances.items()}

    importance_df = (
        pd.DataFrame({
            "feature_name": list(importances.keys()),
            "importance": list(importances.values())
        })
        .sort_values("importance", ascending=False)
    )

    top20_all = importance_df.head(20).copy()
    print("\nTop 20 important features (all types):")
    print(top20_all[["feature_name", "importance"]])

    if args.feature_mode != "mole_only":
        ENGINEERED_PREFIXES = ("gram", "bacterium_clean")

        gpt_importance_df = importance_df[
            importance_df["feature_name"].apply(
                lambda x: (
                    not x.startswith("mole__")
                    and not x.startswith("chemdesc__")
                    and not x.startswith("cat__")
                    and not any(x.startswith(p) for p in ENGINEERED_PREFIXES)
                )
            )
        ]

        top10_gpt = gpt_importance_df.head(10)

        print("\nTop 10 important non-MolE/non-ChemDesc features:")
        print(top10_gpt[["feature_name", "importance"]])

        top_gpt_csv = os.path.join(args.output_dir, f"top_gpt_features_{args.feature_mode}.csv")
        top10_gpt.to_csv(top_gpt_csv, index=False)
        print(f" - Saved top features to: {top_gpt_csv}")

    # -------------------- SAVE OUTPUTS --------------------
    preds_csv = os.path.join(args.output_dir, f"test_predictions_{args.feature_mode}.csv")
    preds.to_csv(preds_csv, index=False)
    print(f" - Saved: {preds_csv}")

    print("Formatting predictions in MolE paper-style output...")

    df_paper = preds.copy()
    df_paper["pred_id"] = df_paper["prestwick_ID"].astype(str) + ":" + df_paper["bacterium"].astype(str)
    df_paper = df_paper.rename(columns={
        "proba": "antimicrobial_predictive_probability",
        "pred_f1": "growth_inhibition"
    })

    df_paper = df_paper[["pred_id", "antimicrobial_predictive_probability", "growth_inhibition"]]

    paper_csv = os.path.join(args.output_dir, f"paper_style_predictions_{args.feature_mode}.csv")
    df_paper.to_csv(paper_csv, index=False)
    print(f" - Saved paper-style predictions: {paper_csv}")

    per_strain_csv = os.path.join(args.output_dir, f"per_strain_metrics_{args.feature_mode}.csv")
    per_strain.to_csv(per_strain_csv, index=False)
    print(f" - Saved: {per_strain_csv}")

    try:
        import joblib
        joblib.dump(
            {
                "model": model,
                "params": best_params,
                "val_macro_pr_auc": best_val
            },
            os.path.join(args.output_dir, f"xgb_model_{args.feature_mode}.pkl")
        )

        model_path = os.path.join(args.output_dir, f"xgb_model_{args.feature_mode}.pkl")
        print(f" - Model saved: {model_path}")

    except Exception as e:
        print(f"[WARN] Could not save model pickle: {e}")

    # ── Optional AP-G scoring for external compounds ──────────────────────────
    if args.ap_score_compounds:
        ap_out = args.ap_score_output or os.path.join(args.output_dir, "ap_scores.csv")
        pkl_path = os.path.join(args.output_dir, f"xgb_model_{args.feature_mode}.pkl")
        try:
            compute_ap_scores(
                model_pkl_path=pkl_path,
                gpt_features_csv=args.ap_score_compounds,
                id_col=args.ap_score_id_col,
                output_csv=ap_out,
                mole_embedding_tsv=args.ap_score_mole_tsv,
                mole_id_map_tsv=args.ap_score_mole_id_map,
                mole_map_left_col=args.ap_score_mole_map_left,
                mole_map_right_col=args.ap_score_mole_map_right,
            )
        except Exception as e:
            print(f"[WARN] AP-G scoring failed: {e}")

    print("\n[Done]")


if __name__ == "__main__":
    main()