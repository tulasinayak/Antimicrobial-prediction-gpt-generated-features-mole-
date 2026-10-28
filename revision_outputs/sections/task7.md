## Task 7. Facts to confirm

**1. Original MolE XGBoost search.** Source: `rolayoalarcon/mole_antimicrobial_potential`, `workflow/02.model_training.ipynb` and the committed result file `data/02.model_training/strain_performance.tsv.gz`; the search space is also Supplementary Table 5 of the MolE paper (the supplement gives no count).

- Search space: `n_estimators` {30, 100, 300, 500, 1000}, `max_depth` {5, 10, 50, 100}, `eta` {0.3, 0.1, 0.05, 1}, `subsample` {0.3, 0.5, 0.8, 1.0}; 320 combinations. Everything else XGBoost 1.6.2 defaults (no column subsampling, no early stopping, no class weighting, `tree_method` exact for this data size).
- Number of configurations: **5 random draws, each fitted 5 times** (unseeded; two of the five draws were the same configuration, so 4 distinct). Selection on the mean validation PR-AUC over the 5 fits.
- Metric: pooled PR-AUC over all compound-strain pairs, `sklearn.metrics.auc(recall, precision)` (differs from average precision by about 0.0001 here).
- Selected: `n_estimators=1000, max_depth=100, eta=0.1, subsample=1.0` -> validation 0.7010, test 0.7797. Reproduced here at 0.7802 / 0.7021 (`scripts/reproduce_mole_original.py`, `tree_method=exact`).
- So yes, `max_depth` goes up to 100; `n_estimators` up to 1000 (fixed, no early stopping); learning rate 0.05-1.

**2. Our final grid and boosting setup** (`notebooks/build_train.py`, `PARAM_SPACE`, `BASE_PARAMS`, `train_xgb`). Confirmed as stated:

- `eta` {0.10, 0.05, 0.03}, `max_depth` {5, 8, 10}, `subsample` {0.5, 0.8, 1.0}, `colsample_bytree` {0.5, 0.8, 1.0}, `min_child_weight` {1, 5}, `gamma` {0, 0.1}.
- `tree_method="hist"`, `max_bin=512`, `lambda=1`, `alpha=0`, `max_delta_step=1`, `objective="binary:logistic"`, up to 3000 boosting rounds, early stopping after 300 rounds without improvement, `scale_pos_weight = (1 - p) / p` with p the pooled positive rate of the training pairs (12.9%, so about 6.8).
- Per trial k: 5 configurations sampled with `random.seed(k)` (identical for all rows of that trial), each fitted with seeds 0, 1, 2; the configuration with the best mean validation per-strain PR-AUC is refitted with seed k and tested once. The 5-trial runs used `device="cuda"` on a Kaggle GPU.
- Three details worth stating in the paper: (i) early stopping monitors **pooled** validation AUCPR, while selection uses **mean per-strain** validation PR-AUC; (ii) predictions use **all** trees, i.e. up to 300 rounds past the best iteration (e.g. trial-0 MolE+GPT-5.2: 814 rounds, best at 513); (iii) the same validation set is used for early stopping, configuration selection and the F1 threshold.
- Preprocessing (fit on training pairs): MolE and RDKit blocks median-imputed with missingness indicators and standardized; GPT block zero-imputed, not standardized, and GPT columns non-zero in fewer than 1% of training pairs dropped; strain one-hot. Constant MolE dimensions are dropped (935 of 1000 remain).

**3. Definition of "mean per-strain PR-AUC"** (`per_strain_metrics` in `build_train.py`). For each of the 40 strains, `sklearn.metrics.average_precision_score` is computed on that strain's pairs in the split (120 test or 120 validation compounds), and the 40 values are averaged with equal weight. A strain whose AP cannot be computed (e.g. no positives) would be returned as NaN and skipped by the mean; this never happens here: every strain has positives in both splits (validation 11-35 per strain, median 20; test 10-34, median 18). It is average precision (step-wise), not trapezoidal area. The pooled PR-AUC is `average_precision_score` over all 4,800 pairs at once.

**4. Guided GPT-only table (Val F1 == Test F1).** Yes, a copy error. Accuracy in that table matches the stored test predictions thresholded at the validation-optimal threshold (`pred_f1`) exactly for all six rows, but the F1 shown in both columns is not the test F1 of those predictions. Correct test F1 (pooled, same threshold, recomputed from `results/guided/<model>/gpt_only/test_predictions.csv`):

| GPT-only (guided) | F1 as submitted (val = test) | Correct test F1 | Test acc. (%) (matches) |
|---|---|---|---|
| GPT-4.1 | 0.423 | 0.498 | 83.32 |
| GPT-4.1-mini | 0.524 | 0.469 | 78.10 |
| GPT-4o | 0.422 | 0.445 | 80.14 |
| GPT-4o-mini | 0.405 | 0.439 | 78.23 |
| GPT-5.1 | 0.395 | 0.432 | 79.32 |
| GPT-5.2 | 0.508 | 0.507 | 82.58 |

The submitted value was most likely the validation F1 copied into the test column, but this cannot be verified: these runs stored no validation predictions, threshold or split file. They also used the **old 195-test-compound split**, not MolE's scaffold split, so the whole guided table is not comparable with the main table and should be re-run (Task 8b) before it is reported.
