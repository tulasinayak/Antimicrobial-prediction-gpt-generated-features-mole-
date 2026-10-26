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
