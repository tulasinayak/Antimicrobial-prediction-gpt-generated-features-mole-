# Verified findings for the reviewer response

Everything below is reproduced by a script in `scripts/`. Numbers marked *pending*
are waiting on the Table 1 re-runs.

## 1. Table 1 mixes two metrics, and one row uses a different split

`notebooks/build_train.py` prints two test numbers per run: pooled PR-AUC over all
compound-strain pairs, and the mean per-strain PR-AUC it labels "PAPER METRIC".
Matching the submitted table's PR-AUC, F1 *and* accuracy against every stored run
(`scripts/match_table_rows.py`) identifies each row unambiguously:

| Submitted row | Test PR-AUC / F1 / Acc | Run that produced it | Metric used |
|---|---|---|---|
| MolE only (reported [9]) | 0.7797 / 0.595 / — | MolE paper, their protocol | theirs |
| MolE only (reproduced) | 0.6948 / 0.5950 / 88.38 | `results/misc/mole_baseline` | **pooled**, 195-compound split |
| MolE + Expert RDKit | 0.6882 / 0.5676 / 84.54 | `results/guided/gpt-4.1/mole_plus_gpt` | **pooled** |
| Expert RDKit only | 0.7093 / 0.5411 / 81.06 | `results/baselines/rdkit_only` | **pooled** |
| MolE + GPT-4o | 0.7117 / 0.6272 / 89.38 | `results/freeform/gpt-4o` | **per-strain** |
| MolE + GPT-4o-mini | 0.6914 / 0.6380 / 89.33 | `results/freeform/gpt-4o-mini` | **per-strain** |
| MolE + GPT-4.1-mini | 0.6940 / 0.6185 / 88.77 | `results/freeform/gpt-4.1-mini` | **per-strain** |
| MolE + GPT-4.1 | 0.6939 / 0.6172 / 88.29 | `results/freeform/gpt-4.1` | **per-strain** |
| MolE + GPT-5.1 | 0.6651 / 0.6091 / 88.31 | `results/freeform/gpt-5.1` | **per-strain** |
| MolE + GPT-5.2 | 0.7233 / 0.6217 / 87.38 | `results/freeform/gpt-5.2` | **per-strain** |

Per-strain PR-AUC exceeds pooled in every run (by 0.003-0.012), so the mixture
systematically favours the GPT rows. Table 3 (guided) is pooled throughout - the
0.7178 that reviewer Ma8V quoted as "the best guided configuration" is the pooled
value of `results/guided/gpt-4.1-mini/mole_plus_gpt`, whose per-strain value is 0.7211.

**Transcription error:** the reproduced-MolE row is given Test F1 = 0.5950, identical
to the published row above it. The actual F1 of that run is **0.6177**.

## 2. The "MolE + Expert RDKit" row is not that combination

`results/guided/gpt-4.1/mole_plus_gpt/raw_features_used_mole_plus_gpt.csv`: 935 MolE
features, **zero** chemdesc features, and a 143-column block holding **108 RDKit-style
names** (`MolWt`, `TPSA`, `MolLogP`, `fr_*`, …) **plus 35 guided GPT-4.1 names**
(`atom_count_total`, `heterocycle_count`, `spiro_center_count`, …). The row reported as
MolE+RDKit is MolE + RDKit + guided-GPT-4.1, which is why reviewer wCKY saw it fall
below both of its supposed components. No `mole_plus_chemdesc` run exists in the
repository; a clean MolE+RDKit row has never been produced. *(pending: Table 1 re-run)*

## 3. The 8.5-point MolE drop is a split and metric artifact

`scripts/check_split_provenance.py` downloads MolE's published split and compares it:

```
published : {'train': 957, 'test': 120, 'valid': 120}
ours      : {'train': 957, 'test': 120, 'valid': 120}
identical compound set: yes (1197 compounds)
split labels agree    : 1197/1197
```

Our split **is** MolE's published split. But `results/misc/mole_baseline` - the run
behind the reported 0.6948 - has **195** test compounds, so that run did not use it.
The submission therefore compares a non-MolE-split, pooled reproduction against a
MolE-split published number.

On MolE's own split with one metric, the baseline already in the repository
(`results/baselines/mole_only`) scores **0.7231** per-strain / 0.7198 pooled. The gap to
MolE's published 0.7797 narrows from 8.5 points to ~5.7, and what remains is
attributable to: only 935 of 1000 MolE dimensions surviving the constant-column filter,
a 5-trial random search against MolE's broader one, and MolE's F1-optimal 0.044 cutoff
being carried over rather than re-derived.

**MolE's 0.7797 reproduced** (`scripts/reproduce_mole_original.py`). MolE's committed
`strain_performance.tsv.gz` shows their search was 5 configs x 5 fits, unseeded, over
n_estimators {30..1000} x max_depth {5,10,50,100} x eta x subsample. The winner was
`n_estimators=1000, max_depth=100, eta=0.1, subsample=1.0`, a depth our search space
(max_depth <= 10) cannot reach. Trained exactly their way (raw 1000-d features, no early
stopping, no class weighting, no scaling) on identical inputs:

| tree_method | val PR-AUC | test PR-AUC | r vs published per-pair probabilities |
|---|---|---|---|
| exact (xgboost 1.6.2's default here) | 0.7021 | **0.7802** | 0.998 |
| hist (xgboost >= 2 default) | 0.6905 | 0.7655 | 0.968 |
| published | 0.7010 | 0.7798 | - |

So the gap is a protocol difference (search space, fixed rounds, tree method), not a data or
split difference. Their other four configs scored test 0.56-0.80 with validation 0.54-0.68:
on this 120-compound test set, the configuration alone moves test PR-AUC by about 0.2.

**Consequence:** MolE alone (0.7231) ties the best GPT configuration (0.7233), and
RDKit alone (0.7156) beats five of six. *(pending: seed spreads)*

## 3b. Table 1 over 5 full-pipeline trials (Kaggle GPU)

`scripts/run_table1.py --trials 0,1,2,3,4 --device cuda`, MolE's split, our protocol with
the seed fix. Each trial re-runs the 5-config search (configs shared by all rows in that
trial), selection on validation, and one test run. 45/45 runs completed on GPU.
Results: `results/table1_trials/` (`table1.csv`, `RESULTS_MANIFEST.csv`, per-trial metrics).

| Representation | Val PR-AUC | Test PR-AUC (per-strain) | Test PR-AUC (pooled) |
|---|---|---|---|
| MolE | 0.6885 ± 0.0339 | 0.6818 ± 0.0234 | 0.6734 ± 0.0259 |
| RDKit descriptors | 0.6917 ± 0.0068 | 0.6977 ± 0.0083 | 0.6905 ± 0.0069 |
| MolE + RDKit | 0.7126 ± 0.0051 | 0.6861 ± 0.0153 | 0.6809 ± 0.0156 |
| MolE + GPT-4o | 0.7088 ± 0.0104 | 0.6898 ± 0.0075 | 0.6845 ± 0.0075 |
| MolE + GPT-4o-mini | 0.6998 ± 0.0164 | 0.6825 ± 0.0251 | 0.6745 ± 0.0298 |
| MolE + GPT-4.1 | 0.7183 ± 0.0071 | 0.6962 ± 0.0153 | 0.6892 ± 0.0159 |
| MolE + GPT-4.1-mini | 0.7118 ± 0.0062 | 0.6844 ± 0.0126 | 0.6751 ± 0.0136 |
| MolE + GPT-5.1 | 0.7311 ± 0.0075 | 0.6594 ± 0.0228 | 0.6439 ± 0.0209 |
| MolE + GPT-5.2 | 0.7064 ± 0.0069 | **0.7102 ± 0.0164** | **0.7055 ± 0.0171** |

Paired by trial (per-strain test, n = 5):
- MolE + GPT-5.2 beats MolE in 5/5 trials, +0.028 (paired t p = 0.011; Wilcoxon p = 0.062,
  its minimum at n = 5). Not significant after correcting for 8 comparisons.
- MolE + GPT-5.2 vs RDKit alone: +0.013, 3/5 wins, p = 0.12. No GPT row beats RDKit
  significantly; RDKit alone ranks 2nd on test.
- MolE + GPT-5.1 is worse than MolE in 5/5 trials (-0.023, p = 0.025).
- Validation and test rankings still disagree (Spearman -0.13 across the 9 rows):
  GPT-5.1 ranks 1st on validation and last on test in every trial. The validation set is
  used for early stopping, threshold and config selection, so it is optimistic, most of all
  for the largest feature set (GPT-5.1, 1481 features).

## 3c. Every row trained with MolE's recipe (Kaggle GPU)

`scripts/run_mole_method.py --tree_method hist --device cuda` (xgboost 3.4.1): every row gets
MolE's selected configuration (1000 trees, depth 100, eta 0.1, subsample 1.0, raw features,
no early stopping, no class weighting). The configuration is deterministic, so the 95% CIs
come from 1000 compound-level bootstrap resamples of the 120 test compounds, shared by all
rows so differences against MolE are paired. Results: `results/mole_method_all_rows/`.

| Representation | Val PR-AUC | Test PR-AUC | Test 95% CI | Diff vs MolE [95% CI] |
|---|---|---|---|---|
| MolE | 0.6960 | **0.7527** | [0.609, 0.849] | - |
| RDKit descriptors | 0.6613 | 0.7146 | [0.562, 0.835] | -0.037 [-0.098, +0.017] |
| MolE + RDKit | 0.6972 | 0.7263 | [0.576, 0.835] | -0.026 [-0.066, +0.010] |
| MolE + GPT-4o | 0.6845 | 0.7301 | [0.580, 0.837] | -0.022 [-0.062, +0.010] |
| MolE + GPT-4o-mini | 0.6956 | 0.7436 | [0.595, 0.848] | -0.009 [-0.033, +0.012] |
| MolE + GPT-4.1 | 0.6589 | 0.7033 | [0.549, 0.820] | -0.050 [-0.122, +0.014] |
| MolE + GPT-4.1-mini | 0.6993 | 0.7428 | [0.598, 0.850] | -0.009 [-0.037, +0.016] |
| MolE + GPT-5.1 | 0.7262 | 0.6243 | [0.463, 0.759] | **-0.126 [-0.213, -0.056]** |
| MolE + GPT-5.2 | 0.6744 | 0.7331 | [0.592, 0.838] | -0.019 [-0.059, +0.022] |

- Under MolE's recipe nothing beats MolE alone; every difference except GPT-5.1's has a CI
  containing 0. GPT-5.1 is clearly worse, as in the 5-trial run.
- This recipe favours MolE by construction: the configuration was chosen on MolE's own
  validation set. A per-row search over MolE's space would be the fair version.
- A single deterministic fit is fragile across implementations. MolE alone: exact CPU 0.7802,
  hist CPU 0.7655, hist GPU 0.7527. MolE + RDKit: hist CPU 0.7470 vs hist GPU 0.7263. These
  shifts (0.013-0.03) are the size of the differences between rows.
- Test CIs are about +/-0.12 wide for every row: 120 test compounds cannot separate
  representations that differ by 0.01-0.05.

## 3d. GPT-only and guided rows, 5 trials (Kaggle GPU)

`scripts/run_table1.py --trials 0,1,2,3,4 --include-gpt-only --include-guided --only-extra
--device cuda`: 18 rows x 5 trials, all on GPU, same protocol and split as 3b.
Results: `results/table1_trials_extra/`. Guided inputs rebuilt by `scripts/flatten_guided.py`
(one schema, 46 features per model, shared by both guided rows); the submitted guided runs had
used inconsistent flattenings and the old 195-compound split, so their numbers are not
comparable.

Mean per-strain test PR-AUC +/- SD (reference: MolE 0.6818, RDKit 0.6977, MolE+GPT-5.2 0.7102):

| Model | GPT only (free-form) | MolE + guided | Guided only | Guided only, submitted |
|---|---|---|---|---|
| GPT-4o-mini | 0.6436 ± 0.0096 | 0.6812 ± 0.0094 | 0.5275 ± 0.0174 | 0.4779 |
| GPT-4.1-mini | 0.6704 ± 0.0086 | **0.6935 ± 0.0207** | 0.5600 ± 0.0259 | 0.5220 |
| GPT-4o | 0.6576 ± 0.0254 | 0.6784 ± 0.0219 | 0.5379 ± 0.0367 | 0.4957 |
| GPT-4.1 | **0.6850 ± 0.0114** | 0.6926 ± 0.0090 | **0.6035 ± 0.0136** | 0.5289 |
| GPT-5.1 | 0.6328 ± 0.0091 | 0.6928 ± 0.0210 | 0.5883 ± 0.0376 | 0.4239 |
| GPT-5.2 | 0.6713 ± 0.0080 | 0.6868 ± 0.0147 | 0.6158 ± 0.0140 | 0.5919 |

- **The guided schema costs signal when GPT stands alone**: free-form beats guided by
  0.044-0.120 for every model, 5 of 6 significant (paired t, n = 5).
- **GPT-4.1 features alone match MolE alone** (+0.003, 3/5 wins, p = 0.80). The submission's
  "GPT descriptors alone do not substitute for the pretrained graph representation" was drawn
  from the guided runs and does not hold free-form.
- Added to MolE, guided and free-form are indistinguishable (4 of 6 with |diff| < 0.01), except
  GPT-5.1 where guided wins 5/5 (+0.033, p = 0.003) - its 509 free-form features hurt.
- **No guided row beats RDKit**: the best (MolE + GPT-4.1-mini, 0.6935) is 0.004 below it
  (p = 0.69), and all six now sit within 0.678-0.694.

## 3e. MolE's full original protocol, 5 trials (Kaggle GPU)

`scripts/run_mole_protocol_search.py --rows mole_only,mole_plus_gpt-5.2,gpt_only_gpt-5.2
--trials 0,1,2,3,4 --device cuda`: MolE's own search, not ours. Per trial k, 5 configurations are
drawn with seed k from MolE's space (n_estimators {30,100,300,500,1000}, max_depth {5,10,50,100},
eta {0.05,0.1,0.3,1.0}, subsample {0.3,0.5,0.8,1.0}) and shared by all three rows; each is fitted
5 times; the configuration with the best mean validation PR-AUC is selected. Raw features with
fillna(0), no scaling, no early stopping, no class weighting, pooled PR-AUC = `auc(recall,
precision)`. 375 fits, 1.3 GPU-hours. Results: `results/mole_protocol_trials/`.

Pooled test PR-AUC +/- SD over 5 trials, with the same rows under our protocol (3b/3d, pooled
column) for comparison:

| Row | MolE's protocol | Our protocol | Protocol effect |
|---|---|---|---|
| MolE | 0.7146 +/- 0.0211 | 0.6734 +/- 0.0259 | **+0.041** |
| MolE + GPT-5.2 | **0.7361 +/- 0.0291** | 0.7055 +/- 0.0171 | +0.031 |
| GPT-5.2 only | 0.6362 +/- 0.0145 | 0.6619 +/- 0.0062 | -0.026 |

Paired by trial against MolE alone (n = 5):

| Comparison | Mean diff | SD | Wins | p (paired t) | p (Wilcoxon) |
|---|---|---|---|---|---|
| MolE + GPT-5.2 vs MolE | +0.0216 | 0.0335 | 4/5 | 0.223 | 0.313 |
| GPT-5.2 only vs MolE | **-0.0784** | 0.0316 | 0/5 | **0.005** | 0.063 |

- **The GPT gain survives the protocol change, but is not significant under it.** MolE + GPT-5.2
  beats MolE by +0.022 in 4 of 5 trials here, against +0.032 pooled (5/5, p = 0.011 per-strain)
  under ours. MolE's protocol roughly doubles the trial-to-trial SD of the difference, so the
  same effect no longer separates from zero at n = 5.
- **MolE's protocol is worth about +0.04 to MolE and -0.03 to GPT alone.** Deep unregularised
  trees on raw features suit 1000-d embeddings; the 45 GPT features do better with our scaling,
  class weighting and early stopping. Protocol choice is not neutral between representations,
  which is an argument for reporting both rather than adopting either.
- **The published 0.7797 is a draw from a very wide distribution, not a protocol advantage we
  failed to match.** Across the 125 MolE-alone fits, test PR-AUC spans 0.229 to 0.810 and
  validation 0.230 to 0.713; the best (trial, configuration) cell averages 0.7747. MolE's winning
  configuration (1000 trees, depth 100, eta 0.1, subsample 1.0) was never drawn in 25 draws,
  consistent with the ~1.6% chance per 5-draw trial.
- **Validation barely ranks test.** Within a trial, across the 5 configurations, Spearman
  rho(val, test) averages 0.44 for MolE, 0.78 for MolE + GPT-5.2 and 0.72 for GPT-5.2 only.
  Selecting on test instead of validation would gain MolE up to +0.066 in a single trial. The
  inverted validation/test ordering the reviewers objected to is a property of a 120-compound
  test set under this search, not of GPT features.
- **A third of MolE's space is unusable.** 50 of 375 fits score below 0.40, all of them at
  eta = 1.0 or at eta = 0.3 with 500 deep trees; mean test by eta is 0.697 (0.05), 0.697 (0.1),
  0.646 (0.3), 0.447 (1.0). MolE's 5-draw search therefore spends a large share of its budget on
  configurations that cannot work, which is why its selected test values swing so widely.
- Three of the 5 MolE trials selected a subsample = 1.0 configuration, where the fit is
  deterministic and fit-to-fit SD is exactly 0; the variance in the table above is from the
  search, not from seeding. For the selected configurations `auc(recall, precision)` and
  `average_precision_score` agree to 4e-4, so the metric choice does not affect these numbers.
- GPU `hist` is used throughout; MolE used CPU `exact`, which was worth +0.015 on MolE's own
  configuration (0.7802 vs 0.7655, section 3). The protocol effects above are larger than that.

## 4. External MIC validation: GPT does not improve attribution

`scripts/ap_g_comparison.py`. Supplementary Table 10 already carries the published MolE
and ECFP4 antimicrobial-potential scores for these compounds, so the matched comparison
reviewer Ma8V asked for needs no retraining. On the **identical 38 observations from 29
compounds**:

| Score | rho | p |
|---|---|---|
| MolE, published (total) | **-0.5353** | 0.00053 |
| MolE + GPT-5.2 (ours) | -0.4802 | 0.00228 |
| GPT-5.2 only (ours) | -0.3384 | 0.0377 |
| ECFP4, published (total) | -0.2838 | 0.305 |

Our -0.4802 / p = 0.0023 reproduces the submitted 3.5 exactly. But the published MolE
score alone correlates *better* with external MIC than our MolE+GPT model does. Adding
GPT features does not improve external validity on this set. GPT features alone do
carry real signal (rho = -0.34, p = 0.038), which is the honest positive statement.

Caveat: MolE's published score comes from their fully-trained model, not our
reproduction. *(pending: our own MolE-only AP-G, once the Table 1 model exists)*

**Gram-specific scores (Ma8V question 3)**, using MolE's published G+/G- against
gram-matched MICs: G+ rho = -0.307 (p = 0.093, n = 31); G- rho = +0.029 (p = 0.914,
n = 16). The gram-negative score has no detectable relationship with gram-negative MICs
in this set.

**Counts (mMKW question 4):** ST10 holds 44 compounds and 54 MIC observations (36 gram+,
18 gram-), of which 41 compounds have at least one MIC. The matched subset is **38
observations from 29 compounds**, containing 1 exactly coincident pair, so the figure
shows **37** distinct marker positions. The caption should state all three numbers.

## 5. Vocabulary divergence survives synonym-aware matching

`scripts/feature_overlap_matching.py` reproduces the submission exactly under strict
matching and adds a lenient bound (case/punctuation folded, estimator affixes stripped,
synonym table for the common physicochemical descriptors):

| | strict | lenient |
|---|---|---|
| max pairwise Jaccard | **0.1193** | 0.1387 |
| mean pairwise Jaccard | 0.0249 | 0.0358 |
| model-specific fraction | **0.626 - 0.945** | 0.540 - 0.920 |

The headline finding is robust: even when synonyms are merged, 54-92% of each model's
descriptor vocabulary appears in no other model, and no pair exceeds 0.14 Jaccard.
Figure 1 should be published as this range.

Minor: the submission says feature sets range from 38 to **509**; the files give 38 to
**508** after metadata removal. Worth reconciling.

## 6. Definition mismatches against the MolE paper

- **AP score.** MolE define `G_i = log2` of the geometric mean of `p_ij`.
  `build_train.py` uses natural log. Spearman rho is unaffected, so 3.5 stands, but the
  values are not on MolE's scale; `scripts/ap_g_comparison.py` divides by ln 2 before
  comparing. This also settles wCKY's minor comment about Equation 1.
- **G+ / G-** are defined over a fixed taxon ordering (first 22 gram-positive, last 18
  gram-negative).
- **The 0.044 threshold** hardcoded as `pred_0044` is MolE's F1-optimal cutoff for
  *their* MolE model (they report 0.044 / 0.068 / 0.209 for MolE / ChemDesc / ECFP4).
  Using it for our models should be stated or re-derived.
- **The 98 descriptors** are the Algavi & Borenstein chemical-descriptor set (d = 98).
  Our `rdkit_only` run carries **96** after constant-column dropping, and the
  contaminated row carries 108 RDKit-style names. All three need reconciling.
- **Only 935 of MolE's 1000 dimensions** reach the model.
- MolE's selection rule (random search, best validation PR-AUC, then test) matches ours,
  so it is not a source of the gap.

## 7. The seed never reached XGBoost — there was no variance to measure

Found while validating the re-runs. `train_xgb(joined, params, random_state)` accepts
`random_state` but **never applies it**: the current `notebooks/build_train.py` builds its
parameter dict without a `seed` entry, so every call fits the identical model.

Evidence from the first validation run's log, three "seeds" inside one search trial:

```
[SEARCH] Trial 1/5
Mean per-strain AUPRC (paper-style): 0.7103 and accuracy: 0.9070833333333334
Mean per-strain AUPRC (paper-style): 0.7103 and accuracy: 0.9070833333333334
Mean per-strain AUPRC (paper-style): 0.7103 and accuracy: 0.9070833333333334
```

Identical to four decimals in all five trials.

Consequences:

- Table 1's caption says "Validation values are means over seeds 0, 1 and 2". Those three
  fits were the same model, so the reported validation values are **single runs averaged
  with themselves**, not seed means. The caption overstates what was done.
- Two thirds of the hyperparameter-search compute was spent refitting identical models.
- A seed sweep on the unpatched code would have produced a standard deviation of exactly
  0.0000 — a precise-looking error bar carrying no information.

**Fix applied:** `params["seed"] = int(random_state)` in `train_xgb`.
`scripts/check_seed_effect.py` verifies the mechanism:

```
subsample=colsample=1.0: IDENTICAL (seed has no effect), max|delta| = 0.000e+00
subsample=colsample=0.8: differs, max|delta| = 4.243e-01
subsample=colsample=0.5: differs, max|delta| = 5.394e-01
```

So seeds now produce genuine variation — but only when the selected configuration
subsamples. `tree_method="hist"` with `subsample=1.0` and `colsample_bytree=1.0` is
deterministic, and `PARAM_SPACE` includes 1.0 for both. `build_train.py` now warns
explicitly when a zero spread is deterministic rather than evidential, in which case the
variance estimate must come from **repeated scaffold splits** instead — which is the
alternative reviewer wCKY offered in W1.
