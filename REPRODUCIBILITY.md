# Reproducing the results

Three reviewers raised reproducibility. This file states exactly what produces each
reported number, and pins the environment so a fresh install cannot drift.

## Why pinning matters here

The repository's `requirements.txt` gives lower bounds only (`numpy>=1.24`,
`xgboost>=2.0`, `scikit-learn>=1.3`, ...), so `pip install -r requirements.txt` resolves
to whatever is current on the day. That is not a reproducible specification, and the
code already carries a compatibility shim for a scikit-learn 1.7 to 1.8 attribute rename
(`_fit_dtype` to `_fill_dtype`, in `compute_ap_scores`), which shows the pipeline is
sensitive to the versions it runs against.

`requirements-lock.txt` pins the versions that produced the reported numbers:

| package | version |
|---|---|
| python | 3.13.5 |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| xgboost | 3.2.0 |
| rdkit | 2025.9.6 |
| openpyxl | 3.1.5 |
| joblib | 1.5.3 |

## Option A - Docker (no local Python needed)

```bash
docker build -t gpt-mole:1.0 .
docker run --rm -v "$PWD/results:/work/results" gpt-mole:1.0 \
    python -u scripts/run_table1.py --seeds 0,1,2,3,4
```

All inputs are tracked in `data/` and baked into the image, so nothing is downloaded at
run time. Results land in the mounted `results/` directory.

## Option B - local environment

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-lock.txt
python -u scripts/run_table1.py --seeds 0,1,2,3,4
python scripts/aggregate_table1.py
```

## What each script produces

| Script | Produces | Answers |
|---|---|---|
| `scripts/run_table1.py` | every Table 1 row, one protocol, seeds 0-4 | Ma8V Q1, wCKY W1, Q5Nz W2 |
| `scripts/aggregate_table1.py` | `results/table1/table1.csv` + `RESULTS_MANIFEST.csv` | the metric-mixing defect |
| `scripts/ap_g_comparison.py` | `results/mechchem/ap_g_comparison.csv` | Ma8V W3 and Q3, mMKW Q4 |
| `scripts/feature_overlap_matching.py` | strict vs lenient overlap tables | wCKY W4, 8LiW |
| `scripts/match_table_rows.py` | maps every published number to its run | provenance |
| `scripts/check_split_provenance.py` | verifies the split against MolE's published one | Ma8V Q2 |
| `scripts/check_seed_effect.py` | demonstrates the seed bug and its fix | the variance defect |

## Determinism caveats

- **The training seed now reaches XGBoost.** It previously did not, so every "seed"
  refit an identical model; see `FINDINGS.md` section 7.
- **Seed variance is not always meaningful.** `tree_method="hist"` with
  `subsample=1.0` and `colsample_bytree=1.0` is deterministic, and `PARAM_SPACE`
  includes 1.0 for both. When the selected configuration is deterministic,
  `build_train.py` prints a warning and the spread must instead come from repeated
  scaffold splits.
- **The scaffold split is fixed and is MolE's own.** Verified compound-by-compound by
  `scripts/check_split_provenance.py`; do not regenerate it.
- **GPT feature generation is not reproducible by construction** - that is the paper's
  own finding. The generated CSVs in `data/gpt_generation_without_guidance/` are the
  exact inputs used; regenerating them produces a different feature vocabulary.
