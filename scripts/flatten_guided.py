"""Flatten the guided (schema-constrained) GPT feature files into one flat CSV per model.

The guided generations store each schema section as a JSON-like dict per cell
(data/gpt_feature_generation/gpt_features_4o_<model>_unified.csv, one file per model, all
from the same guided prompt). The submitted guided runs flattened them inconsistently:
GPT-only runs used `section.key` scalars only (~40 columns), MolE+GPT runs used
`section__key` plus list lengths, and two runs drew on other files. For the re-run every
model is flattened once, with the project's own `flatten_dict` from
analysis/final_analysis.ipynb, and both guided rows (MolE + guided, guided only) read the
same file.

Writes data/gpt_guided_flat/gpt_features_guided_<model>.csv (prestwick_ID + numeric
features). Identifier strings and the request echo (`input`) are dropped.

Usage:
    python -I scripts/flatten_guided.py
"""

from __future__ import annotations

import ast
import os

import pandas as pd

SRC = "data/gpt_feature_generation/gpt_features_4o_{model}_unified.csv"
OUT_DIR = "data/gpt_guided_flat"
MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "gpt-5.1", "gpt-5.2"]
NON_JSON_COLS = {"prestwick_ID", "smiles_resolution", "input"}
DROP_PREFIXES = ("identifiers__",)


def safe_parse(val):
    if pd.isna(val):
        return {}
    try:
        return ast.literal_eval(str(val).strip())
    except Exception:
        return {}


def flatten_dict(d, prefix=""):
    """As in analysis/final_analysis.ipynb: nested keys joined by '__', lists -> length + first 6 numbers."""
    out = {}
    for k, v in d.items():
        full_key = f"{prefix}__{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, full_key))
        elif isinstance(v, (list, tuple)):
            out[f"{full_key}__len"] = len(v)
            for i, elem in enumerate(v[:6]):
                if isinstance(elem, (int, float, bool)):
                    out[f"{full_key}__{i}"] = float(elem)
        elif isinstance(v, bool):
            out[full_key] = float(v)
        elif isinstance(v, (int, float)):
            out[full_key] = float(v)
    return out


def flatten_file(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    records = []
    for _, row in raw.iterrows():
        flat = {"prestwick_ID": row["prestwick_ID"]}
        for col in raw.columns:
            if col in NON_JSON_COLS:
                continue
            parsed = safe_parse(row[col])
            if isinstance(parsed, dict):
                flat.update(flatten_dict(parsed, prefix=col))
            elif isinstance(parsed, (int, float, bool)):
                flat[col] = float(parsed)
        records.append(flat)
    df = pd.DataFrame(records).set_index("prestwick_ID")
    df = df.drop(columns=[c for c in df.columns if c.startswith(DROP_PREFIXES)])
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return df.reset_index()


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    for model in MODELS:
        df = flatten_file(SRC.format(model=model))
        out = os.path.join(OUT_DIR, f"gpt_features_guided_{model}.csv")
        df.to_csv(out, index=False)
        n_missing = int(df.drop(columns="prestwick_ID").isna().all(axis=1).sum())
        print(f"{model:13s} {len(df)} compounds, {df.shape[1] - 1} features "
              f"({n_missing} compounds with no parsed values) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
