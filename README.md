# GPT + MolE: LLM-Generated Molecular Features for Antimicrobial Activity Prediction

This repository investigates whether GPT-generated molecular feature vectors can improve
antimicrobial activity prediction when combined with learned embeddings (MolE) and
traditional RDKit descriptors.

We benchmark **seven GPT model variants** (gpt-4o, gpt-4.1, gpt-4.1-mini, gpt-4o-mini,
gpt-5.1, gpt-5.2, and gpt-4.1-mini-v2) across two feature-generation strategies:

- **Guided** — structured JSON schema with fixed feature names (original pipeline)
- **Freeform** — schema-free prompting; GPT decides the features autonomously, then a
  locked schema is inferred from the first compound and applied to all others

Models are evaluated on the Maier et al. gut microbiome screening dataset
(~1 200 compounds x 40 bacterial strains). A **Figure 7 reproduction** extends the
best-performing model (gpt-5.2 freeform) to the MCE chemical library
(3 287 compounds) and compares GPT-predicted Antimicrobial Potential G (AP-G)
scores against experimentally measured MIC values.

---

## Quick start — reproduce the analysis

All pre-computed inputs are already in the repo; no API key needed.

```bash
git clone https://github.com/<your-username>/GPT+Mole.git
cd GPT+Mole

conda create -n gpt-mole python=3.10
conda activate gpt-mole
conda install -c conda-forge rdkit
pip install -r requirements.txt

# Download the Maier et al. data files into data/raw_data/  (see Data section below)

# Guided-feature analysis (all GPT model variants)
jupyter notebook analysis/final_analysis.ipynb

# Freeform-feature EDA for gpt-5.2
jupyter notebook analysis/freeform_xgboost_prep_gpt52.ipynb

# Feature overlap analysis (Venn diagrams, semantic mapping)
jupyter notebook analysis/freeform_feature_overlap_analysis.ipynb

# MCE library EDA
jupyter notebook analysis/mechchem_eda.ipynb

# Figure 7 (AP-G vs MIC)
jupyter notebook analysis/AP_VS_MIC_MCE.ipynb
```

---

## Repository layout

```
GPT+Mole/
├── analysis/
│   ├── feature_matrix.ipynb                      # GPT feature quality (guided, all models)
│   ├── final_analysis.ipynb                      # Final comparison across 7 GPT variants
│   ├── freeform_xgboost_prep_gpt52.ipynb         # Freeform EDA + XGBoost for gpt-5.2  *
│   ├── freeform_xgboost_prep_gpt51.ipynb         # Freeform EDA for gpt-5.1            *
│   ├── freeform_xgboost_prep_gpt41.ipynb         # Freeform EDA for gpt-4.1            *
│   ├── freeform_xgboost_prep_gpt4o.ipynb         # Freeform EDA for gpt-4o             *
│   ├── freeform_xgboost_prep_gpt4omini.ipynb     # Freeform EDA for gpt-4o-mini        *
│   ├── freeform_xgboost_prep.ipynb               # Base freeform prep notebook          *
│   ├── freeform_feature_overlap_analysis.ipynb   # Feature overlap: Venn, semantic map  *
│   ├── mechchem_eda.ipynb                        # MCE chemical library EDA             *
│   └── AP_VS_MIC_MCE.ipynb                       # Figure 7 AP-G vs log2(MIC) plot     *
│
├── notebooks/
│   ├── feature_generation.py     # Generate GPT features (guided or freeform)       *
│   ├── build_train.py            # XGBoost training + AP-G score computation        *
│   ├── explore_dataset.py        # Early EDA (original Google Colab script)
│   └── understand_dataset.py     # Dataset statistics (original Google Colab script)
│
├── data/
│   ├── gpt_feature_generation/              # Guided GPT feature CSVs (all models)
│   ├── gpt_generation_without_guidance/     # Freeform GPT feature CSVs             *
│   │   ├── gpt_features_4_gpt-4.1_freeform.csv
│   │   ├── gpt_features_4_gpt-4.1-mini_freeform.csv
│   │   ├── gpt_features5_gpt-5.1_freeform.csv
│   │   ├── gpt_features5_gpt-5.2_freeform.csv
│   │   ├── gpt_features4o_gpt-4o_freeform.csv
│   │   └── gpt_features4omini_gpt-4o-mini_freeform.csv
│   └── raw_data/                            # All raw input data                     *
│       ├── graphBT_R1k_E8kl1e3_representation.tsv  # MolE GNN embeddings for MCE (1000-dim)
│       ├── maier_chemdesc_representation.tsv        # RDKit descriptors for Maier compounds
│       ├── maier_scaffold_split.tsv                 # Bemis-Murcko scaffold split (fixed)
│       │   ── files below must be downloaded separately (see Data section) ──
│       ├── prestwick_library_screened.tsv.gz        # Prestwick compound library + SMILES
│       ├── maier_mole_representation_with_id.tsv.gz # MolE embeddings (Maier dataset)
│       └── screen_results_info_SF3.xlsx             # Binary antimicrobial activity labels
│
├── figures/                                  # All plots and figures                  *
│   ├── feature_analysis/                   # GPT feature quality plots (per model)
│   │   └── gpt-4.1/ … gpt-5.2/            #   self-corr, cross-corr, importance
│   ├── feature_overlap/                    # Venn diagrams + semantic mapping
│   │   └── fig1-fig6 PNGs + PDFs + tables/
│   ├── guided_model_analysis/              # Guided-mode comparison plots (per model)
│   │   └── gpt-4.1/ … gpt-5.2/
│   ├── freeform_comparison/                # Cross-model freeform comparison plots
│   └── figure7_ap_g_vs_mic/               # AP-G vs MIC scatter figures
│       ├── fig5_original.png              #   MolE baseline (rho=-0.493)
│       ├── fig5_gpt_scores.png            #   MolE+GPT-5.2 XGB (rho=-0.480)
│       ├── fig5_gpt_only_scores.png       #   GPT-5.2 only XGB (rho=-0.338)
│       └── fig5_comparison.png            #   Side-by-side comparison
│
├── results/                                  # All training results and MCE outputs   *
│   ├── freeform/                            # Freeform-feature XGBoost runs
│   │   ├── model_importance/               #   top_gpt_features.csv per freeform model
│   │   ├── gpt-4.1/                        #   Each folder has both modes:
│   │   ├── gpt-4.1-mini/                   #   *_mole_plus_gpt.* + *_gpt_only.*
│   │   ├── gpt-4o/                         #   per_strain_metrics, top_gpt_features,
│   │   ├── gpt-4o-mini/                    #   xgb_model_*.pkl, test_predictions, …
│   │   ├── gpt-5.1/
│   │   └── gpt-5.2/
│   ├── guided/                              # Guided-feature XGBoost runs
│   │   ├── gpt-4.1/
│   │   │   ├── mole_plus_gpt/              #   MolE + guided GPT features
│   │   │   └── gpt_only/                   #   Guided GPT features only
│   │   ├── gpt-4.1-mini/
│   │   ├── gpt-4o/
│   │   ├── gpt-4o-mini/
│   │   ├── gpt-5.1/
│   │   └── gpt-5.2/
│   ├── baselines/
│   │   ├── mole_only/                      # MolE embeddings only
│   │   └── rdkit_only/                     # RDKit/ChemDesc descriptors only
│   ├── mechchem/                            # MCE library extension outputs           *
│   │   ├── SupplementaryTable10.xlsx       # ST10 from Olayo-Alarcon et al. (2025)
│   │   ├── mole_mce_predictions.xlsx       # Original MolE AP-G scores for MCE cpds
│   │   ├── fig5_compound_catalog_map.csv   # ProductName → HY catalog number bridge
│   │   ├── gpt_features_chemical_library_fig5_compounds_gpt-5.2_freeform.csv
│   │   ├── ap_scores_gpt52.csv             # MolE+GPT-5.2 AP-G scores (Fig5 compounds)
│   │   ├── ap_scores_gpt52_gptonly.csv     # GPT-5.2 only AP-G scores (Fig5 compounds)
│   │   │   ── excluded by .gitignore; download separately (see Data section) ──
│   │   ├── chemical_library_cleaned.tsv    # 3287 MCE compounds + SMILES
│   │   └── chemical_library_fig5_compounds.tsv      # Fig5 compound subset (~39)
│   └── misc/                               # Extra/exploratory runs (not in paper)
│
├── .gitignore
├── requirements.txt
└── README.md

* = added or significantly updated in the freeform / MCE extension phase
```

---

## Installation

### 1. Clone and create environment

```bash
git clone https://github.com/<your-username>/GPT+Mole.git
cd GPT+Mole

conda create -n gpt-mole python=3.10
conda activate gpt-mole
conda install -c conda-forge rdkit
pip install -r requirements.txt
```

### 2. Set your OpenAI API key

Required only if re-generating GPT features. Never hardcode the key in source files.

```bash
# Linux / macOS
export OPENAI_API_KEY="sk-..."

# Windows PowerShell
$env:OPENAI_API_KEY = "sk-..."
```

---

## Data

### Maier et al. dataset (main training data)

Three files must be placed in `data/raw_data/`:

| File | Description | Source |
|---|---|---|
| `prestwick_library_screened.tsv.gz` | Prestwick compound library + SMILES | Maier et al. supplementary |
| `maier_mole_representation_with_id.tsv.gz` | Pre-computed MolE embeddings | Maier et al. supplementary |
| `screen_results_info_SF3.xlsx` | Binary antimicrobial activity labels (p-value matrix) | Maier et al. SF3 |

> Maier, L. et al. "Unravelling the collateral damage of antibiotics on gut bacteria."
> *Nature* 599, 120-124 (2021). https://doi.org/10.1038/s41586-021-03986-2

**Important:** `prestwick_library_screened.tsv.gz` is a compound metadata file (SMILES,
molecular weight, ATC codes, etc.). Always pass `screen_results_info_SF3.xlsx` as `--sf3`
— it is the binary activity label matrix.

### MCE chemical library (Figure 7 extension)

Already committed in the cloned repo:

| File | Description |
|---|---|
| `results/mechchem/SupplementaryTable10.xlsx` | ST10 from Olayo-Alarcon et al. — MIC data + AP-G scores |
| `data/raw_data/graphBT_R1k_E8kl1e3_representation.tsv` | MolE GNN embeddings for MCE library (1000-dim, 3286 cpds) |

Excluded by `.gitignore` — place in `results/mechchem/` before running the MCE pipeline:

| File | Description |
|---|---|
| `chemical_library_cleaned.tsv` | 3287 MCE compounds with `ProductName` + `rdkit_no_salt` SMILES |
| `chemical_library_fig5_compounds.tsv` | Subset of ~39 Fig. 5 compounds with MIC data |

> Olayo-Alarcon, R. et al. "Predicting antimicrobial activity of natural compounds using
> machine learning." *Nature Communications* (2025).

---

## Running the pipeline

### Guided-feature pipeline (original)

#### Step 1 — Generate guided GPT features *(optional)*

```bash
python notebooks/feature_generation.py --model_name gpt-4.1
```

Repeat for each model (`gpt-4o`, `gpt-4.1-mini`, `gpt-4o-mini`, `gpt-5.1`, `gpt-5.2`).
Output: `data/gpt_feature_generation/gpt_features_*_<model>_unified.csv`

#### Step 2 — Train XGBoost (guided) *(optional)*

```bash
# MolE + GPT features
python notebooks/build_train.py \
    --sf3                data/raw_data/screen_results_info_SF3.xlsx \
    --gpt_features       data/gpt_feature_generation/gpt_features_4o_gpt-5.2_unified.csv \
    --mole_mode          load \
    --mole_tsv           data/raw_data/maier_mole_representation_with_id.tsv.gz \
    --prestwick_screened data/raw_data/prestwick_library_screened.tsv.gz \
    --exact_split        data/raw_data/maier_scaffold_split.tsv \
    --feature_mode       mole_plus_gpt \
    --output_dir         results/guided/gpt-5.2/mole_plus_gpt

# GPT-only
python notebooks/build_train.py \
    --sf3                data/raw_data/screen_results_info_SF3.xlsx \
    --gpt_features       data/gpt_feature_generation/gpt_features_4o_gpt-5.2_unified.csv \
    --prestwick_screened data/raw_data/prestwick_library_screened.tsv.gz \
    --exact_split        data/raw_data/maier_scaffold_split.tsv \
    --feature_mode       gpt_only \
    --output_dir         results/guided/gpt-5.2/gpt_only

# RDKit-only baseline
python notebooks/build_train.py \
    --sf3                data/raw_data/screen_results_info_SF3.xlsx \
    --gpt_features       data/raw_data/maier_chemdesc_representation.tsv \
    --prestwick_screened data/raw_data/prestwick_library_screened.tsv.gz \
    --exact_split        data/raw_data/maier_scaffold_split.tsv \
    --feature_mode       chemdesc_only \
    --output_dir         results/baselines/rdkit_only
```

#### Step 3 — GPT feature quality analysis

```bash
jupyter notebook analysis/feature_matrix.ipynb
```

Produces self-correlation heatmaps, cross-correlation with RDKit, clustered heatmaps,
molecular weight scatter, and feature novelty bar charts per model.
Plots saved to `figures/feature_analysis/<model>/`.

#### Step 4 — Final model comparison

```bash
jupyter notebook analysis/final_analysis.ipynb
```

Loops over all model variants, reads importance rankings from `results/freeform/model_importance/`,
and generates comparison plots saved to `figures/guided_model_analysis/<model>/`.

---

### Freeform-feature pipeline (new)

In the freeform pipeline, GPT receives only a SMILES string and a blank-slate prompt.
It autonomously decides which molecular properties to report. The feature schema is
inferred from the first compound's response and locked for all subsequent compounds,
ensuring a consistent feature matrix.

#### Step 1 — Generate freeform GPT features *(optional)*

`feature_generation.py` now accepts three additional arguments:

| Argument | Default | Purpose |
|---|---|---|
| `--input_file` | Prestwick TSV | Path to any compound TSV/CSV |
| `--id_column` | `prestwick_ID` | Column containing compound identifiers |
| `--lock_schema_from` | None | Path to an existing freeform CSV; its columns become the locked schema for all new compounds |

```bash
# Generate freeform features for the Maier/Prestwick dataset
python notebooks/feature_generation.py \
    --mode       freeform \
    --model_name gpt-5.2

# Generate freeform features for a new compound library using the locked schema
# from an existing freeform CSV (ensures identical columns)
python notebooks/feature_generation.py \
    --mode            freeform \
    --model_name      gpt-5.2 \
    --input_file      results/mechchem/chemical_library_cleaned.tsv \
    --id_column       ProductName \
    --lock_schema_from data/gpt_generation_without_guidance/gpt_features5_gpt-5.2_freeform.csv
```

Output goes to `data/gpt_generation_without_guidance/gpt_features*_<model>_freeform.csv`.

#### Step 2 — Train XGBoost (freeform) *(optional)*

```bash
# mole_plus_gpt
python notebooks/build_train.py \
    --sf3                data/raw_data/screen_results_info_SF3.xlsx \
    --gpt_features       data/gpt_generation_without_guidance/gpt_features5_gpt-5.2_freeform.csv \
    --mole_mode          load \
    --mole_tsv           data/raw_data/maier_mole_representation_with_id.tsv.gz \
    --prestwick_screened data/raw_data/prestwick_library_screened.tsv.gz \
    --exact_split        data/raw_data/maier_scaffold_split.tsv \
    --feature_mode       mole_plus_gpt \
    --output_dir         results/freeform/gpt-5.2

# gpt_only
python notebooks/build_train.py \
    --sf3                data/raw_data/screen_results_info_SF3.xlsx \
    --gpt_features       data/gpt_generation_without_guidance/gpt_features5_gpt-5.2_freeform.csv \
    --prestwick_screened data/raw_data/prestwick_library_screened.tsv.gz \
    --exact_split        data/raw_data/maier_scaffold_split.tsv \
    --feature_mode       gpt_only \
    --output_dir         results/freeform/gpt-5.2
```

Run the same commands for all six freeform CSVs in `data/gpt_generation_without_guidance/`.

#### Step 3 — Freeform EDA notebook

Each model has a dedicated EDA notebook (`freeform_xgboost_prep_gpt52.ipynb`, etc.)
that produces:

- **Cross-correlation heatmap** (`c-crosscorr`): GPT freeform features vs RDKit descriptors
  (Spearman). Top GPT features (by XGBoost gain from the mole_plus_gpt model) are
  highlighted in red — red axhspan bands + bold red tick labels.
- **Self-correlation heatmap** (`c-selfcorr`): Spearman correlations among GPT features.
  Top features highlighted in red on both axes (axhspan rows + axvspan columns).
- XGBoost feature importance bar charts for both `gpt_only` and `mole_plus_gpt` modes.

Top features are loaded from `results/freeform/gpt-5.2/top_gpt_features_mole_plus_gpt.csv`.

```bash
jupyter notebook analysis/freeform_xgboost_prep_gpt52.ipynb
```

---

### MCE library extension — Figure 7

#### Step 1 — MCE EDA

```bash
jupyter notebook analysis/mechchem_eda.ipynb
```

Explores the 3287-compound MCE library: SMILES coverage, overlap with ST10,
distribution of Research Area / Target, and the ~39 Figure 7 compounds (those with
MIC data and MolE predictions in ST10).

#### Step 2 — Generate GPT features for Figure 7 compounds *(optional)*

```bash
python notebooks/feature_generation.py \
    --mode             freeform \
    --model_name       gpt-5.2 \
    --input_file       results/mechchem/chemical_library_fig5_compounds.tsv \
    --id_column        ProductName \
    --lock_schema_from data/gpt_generation_without_guidance/gpt_features5_gpt-5.2_freeform.csv
```

Output: `results/mechchem/gpt_features_chemical_library_fig5_compounds_gpt-5.2_freeform.csv`

#### Step 3 — Compute AP-G scores *(optional)*

`build_train.py` now exposes a `compute_ap_scores()` function. After training, pass
`--ap_score_compounds` to score a new compound library with the saved model:

```bash
python notebooks/build_train.py \
    --sf3                data/raw_data/screen_results_info_SF3.xlsx \
    --gpt_features       data/gpt_generation_without_guidance/gpt_features5_gpt-5.2_freeform.csv \
    --mole_mode          load \
    --mole_tsv           data/raw_data/maier_mole_representation_with_id.tsv.gz \
    --prestwick_screened data/raw_data/prestwick_library_screened.tsv.gz \
    --exact_split        data/raw_data/maier_scaffold_split.tsv \
    --feature_mode       mole_plus_gpt \
    --output_dir         results/freeform/gpt-5.2 \
    --ap_score_compounds results/mechchem/gpt_features_chemical_library_fig5_compounds_gpt-5.2_freeform.csv \
    --ap_score_id_col    ProductName \
    --ap_score_output    results/mechchem/ap_scores_gpt52.csv
```

AP-G formula: `AP-G = mean(log(p_strain + ε))` = log of geometric mean across all 40 strains.
Output `ap_scores_gpt52.csv` has columns `ProductName`, `ap_score_gpt`.

#### Step 4 — Figure 7

```bash
jupyter notebook analysis/AP_VS_MIC_MCE.ipynb
```

- **Step 1** (verification): plots AP-G vs log2(MIC) using original MolE scores from ST10.
  Expected: Spearman ρ ≈ −0.493, p ≈ 0.0009.
- **Step 2**: replaces MolE scores with GPT-5.2 model AP-G scores and generates the
  comparison figure. Side-by-side statistics are printed to the notebook.

---

## Feature modes in `build_train.py`

| `--feature_mode` | Feature blocks used |
|---|---|
| `mole_plus_gpt` | MolE embeddings (935-dim) + GPT features |
| `mole_only` | MolE embeddings only |
| `chemdesc_only` | RDKit/ChemDesc descriptors only |
| `gpt_only` | GPT features only |

---

## Output files from `build_train.py`

Each `--output_dir` will contain (suffix = `_<feature_mode>`):

| File | Description |
|---|---|
| `per_strain_metrics_*.csv` | AUC-ROC, AUC-PR, MCC, F1 per strain |
| `paper_style_predictions_*.csv` | Predicted scores in Maier-paper format |
| `test_predictions_*.csv` | Raw test-set predictions |
| `top_gpt_features_*.csv` | Top GPT features by XGBoost gain |
| `xgb_model_*.pkl` | Saved XGBoost pipeline (preproc + booster + threshold) |

When `--ap_score_compounds` is provided, an additional file is written to
`--ap_score_output` (default: `<output_dir>/ap_scores.csv`):

| File | Description |
|---|---|
| `ap_scores_gpt52.csv` | `ProductName`, `ap_score_gpt` for the new compound library |

---

## Data flow

```
OpenAI API
    |
    v
notebooks/feature_generation.py
    |  guided  -> data/gpt_feature_generation/gpt_features_*_unified.csv
    |  freeform -> data/gpt_generation_without_guidance/gpt_features*_freeform.csv
    v
notebooks/build_train.py
    |  reads  GPT CSV + data/raw_data/ (MolE TSV, SF3 labels, scaffold split)
    |  writes results/<run>/per_strain_metrics_*.csv
    |          results/<run>/top_gpt_features_*.csv
    |          results/<run>/xgb_model_*.pkl
    |  (optional) writes results/mechchem/ap_scores_gpt52.csv
    v
analysis/freeform_xgboost_prep_gpt52.ipynb        <- freeform EDA + heatmaps
analysis/freeform_feature_overlap_analysis.ipynb  <- Venn diagrams, semantic mapping
analysis/final_analysis.ipynb                     <- guided model comparison
    |
    v
analysis/mechchem_eda.ipynb                  <- MCE library overview
    |
    v
analysis/AP_VS_MIC_MCE.ipynb                 <- AP-G vs MIC scatter (Figure 7)
    |  fig5_original.png      (MolE baseline,      rho=-0.493)
    |  fig5_gpt_scores.png    (MolE+GPT-5.2 XGB,  rho=-0.480)
    |  fig5_gpt_only_scores.png (GPT-5.2 only XGB, rho=-0.338)
    |  fig5_comparison.png    (side-by-side)
    v
figures/figure7_ap_g_vs_mic/
```

---

## Navigating the repo

| What you want | Where to look |
|---|---|
| Understand the project | This README + `analysis/final_analysis.ipynb` |
| See freeform GPT feature structure | `data/gpt_generation_without_guidance/*.csv` |
| Understand freeform EDA plots | `analysis/freeform_xgboost_prep_gpt52.ipynb` |
| Re-run training for one freeform model | `notebooks/build_train.py` — see Freeform Step 2 |
| Generate features for a new library | `feature_generation.py --lock_schema_from <existing_csv>` |
| Compute AP-G scores for new compounds | `build_train.py --ap_score_compounds <gpt_csv>` |
| Reproduce Figure 7 (AP-G vs MIC) | `analysis/AP_VS_MIC_MCE.ipynb` |
| Feature overlap Venn diagrams & tables | `analysis/freeform_feature_overlap_analysis.ipynb` |
| Add a new GPT model variant | Run `feature_generation.py --model_name <new-model>`, add to `MODEL_CONFIGS` in `final_analysis.ipynb` |

---

## Figure 7 — AP-G vs log₂(MIC) for MCE library compounds

All four figures are saved in `figures/figure7_ap_g_vs_mic/`.

### fig5_original.png — MolE baseline (paper reproduction)

Reproduces Figure 5 from Olayo-Alarcon et al. (2025) using the original MolE model's
per-compound AP-G scores from ST10. Serves as the ground-truth reference.

- **Spearman ρ = −0.493**, p = 0.0009, slope = −0.329 (p = 0.0025)
- 42 compounds with MIC ≤ 128 µg/mL (Gram-positive and Gram-negative combined)
- Confirms that higher AP-G (less negative) correlates with lower MIC (more active)

---

### fig5_gpt_scores.png — MolE + GPT-5.2 freeform XGBoost

AP-G scores computed by the **MolE + GPT-5.2 freeform** XGBoost model trained on the
Maier et al. Prestwick dataset, then applied to the 29 MCE Fig. 7 compounds.

- MolE embeddings sourced from `data/raw_data/graphBT_R1k_E8kl1e3_representation.tsv`
  (all 29 compounds present; indexed by HY catalog number, bridged to `ProductName`
  via `SupplementaryTable10.xlsx`)
- GPT features from `results/mechchem/gpt_features_chemical_library_fig5_compounds_gpt-5.2_freeform.csv`
- **AP-G formula**: log of geometric mean = `mean(log(p_strain))` across 40 strains
- **Spearman ρ = −0.480**, p = 0.0023, slope = −0.200 (p = 0.0029)
- 38 compounds matched (29 GPT-scored × overlap with MIC data in ST10)

This model achieves correlation nearly as strong as the original MolE paper, demonstrating
that GPT-generated freeform features complement MolE embeddings effectively.

---

### fig5_gpt_only_scores.png — GPT-5.2 freeform only XGBoost

AP-G scores from the **GPT-5.2 freeform only** XGBoost model (no MolE embeddings).
Uses only the 23 GPT-generated physicochemical descriptors.

- **Spearman ρ = −0.338**, p = 0.0377, slope = −0.152 (p = 0.0530)
- 38 compounds matched
- Weaker correlation than MolE+GPT, but statistically significant (p < 0.05),
  confirming that GPT features alone carry meaningful antimicrobial signal

---

### fig5_comparison.png — Side-by-side: MolE+GPT vs GPT-only

Two-panel comparison of the two GPT-5.2 XGBoost models on the same 38 compounds.
Blue dots = Gram-positive; yellow dots = Gram-negative. Pink regression line with
95 % confidence band.

| Model | Spearman ρ | p-value | Slope |
|---|---|---|---|
| MolE-XGB (paper) | −0.493 | 0.0009 | −0.329 |
| MolE + GPT-5.2 XGB | −0.480 | 0.0023 | −0.200 |
| GPT-5.2 only XGB | −0.338 | 0.0377 | −0.152 |

Adding GPT-5.2 freeform features to MolE embeddings reproduces the original paper's
correlation (ρ = −0.480 vs −0.493) while GPT features alone still yield a significant
negative association (ρ = −0.338).

> **Note on AP-G formula**: AP-G is defined as the log of the geometric mean of
> per-strain predicted probabilities across all 40 gut bacterial strains:
> `AP-G = mean(log(p_strain + ε))`. This equals `log(geometric_mean(p_strain))` and
> is more sensitive to near-zero per-strain predictions than the arithmetic-mean variant
> `log(mean(p_strain))`.

---

## Reproducing manuscript results

| Result | File(s) / Notebook |
|---|---|
| **Table 1** (guided GPT models) | `results/guided/gpt-*/mole_plus_gpt/per_strain_metrics_mole_plus_gpt.csv` |
| **Table 3** (freeform GPT models) | `results/freeform/gpt-*/per_strain_metrics_mole_plus_gpt.csv` |
| **Figure 3** (feature overlap, Venn diagrams) | `analysis/freeform_feature_overlap_analysis.ipynb` |
| **Figure 7** (AP-G vs MIC, MCE library) | `analysis/AP_VS_MIC_MCE.ipynb` + `results/mechchem/ap_scores_gpt52.csv` |

> All results use the scaffold-held-out split in `data/raw_data/maier_scaffold_split.tsv` (120 test compounds).
> The `5 trials` in the paper refers to 5 random hyperparameter configurations, each evaluated
> over 3 random seeds, with one final test evaluation using the best configuration.
> API calls used `temperature=0.0` — no tools, retrieval, or external databases were enabled.

---

## Project notes

- The scaffold split (`data/raw_data/maier_scaffold_split.tsv`) is fixed across all runs.
- GPT feature generation uses `temperature=0.0` (deterministic). Pre-generated CSVs are
  the exact inputs used for reported results.
- `prestwick_library_screened.tsv.gz` is compound metadata only. The binary activity
  labels are in `screen_results_info_SF3.xlsx` — always use that as `--sf3`.
- Large files (`.gz`, `.pkl`, intermediate CSVs) are excluded from git.
- The OpenAI API key must be set via the `OPENAI_API_KEY` environment variable and
  must never be hardcoded in source files.
