## Task 8. Additional runs

### 8b. GPT-only and guided rows under the 5-trial protocol — DONE

`scripts/run_table1.py --trials 0,1,2,3,4 --include-gpt-only --include-guided --only-extra
--device cuda` on a Kaggle GPU: 18 rows x 5 trials = 90 runs, all on GPU, same protocol as the
main table (MolE's scaffold split, trial k = search seed and final training seed k, configurations
shared by every row within a trial). Results: `results/table1_trials_extra/`.

Guided inputs were rebuilt first: `scripts/flatten_guided.py` flattens the six
`gpt_features_4o_<model>_unified.csv` files with the project's own `flatten_dict`, giving one
schema of 46 features per model used by both guided rows. The submitted guided runs had
flattened the same files inconsistently (about 40 columns for GPT-only vs 46-144 for MolE+GPT)
and two of them drew on different source files, so those numbers are not comparable with these.

**Mean per-strain test PR-AUC +/- SD over 5 trials** (main-table rows repeated for reference:
MolE 0.6818 +/- 0.0234, RDKit 0.6977 +/- 0.0083, best free-form MolE + GPT-5.2 0.7102 +/- 0.0164):

| Model | GPT only (free-form) | MolE + GPT (guided) | GPT only (guided) | Guided GPT-only, as submitted |
|---|---|---|---|---|
| GPT-4o-mini | 0.6436 +/- 0.0096 | 0.6812 +/- 0.0094 | 0.5275 +/- 0.0174 | 0.4779 |
| GPT-4.1-mini | 0.6704 +/- 0.0086 | **0.6935 +/- 0.0207** | 0.5600 +/- 0.0259 | 0.5220 |
| GPT-4o | 0.6576 +/- 0.0254 | 0.6784 +/- 0.0219 | 0.5379 +/- 0.0367 | 0.4957 |
| GPT-4.1 | **0.6850 +/- 0.0114** | 0.6926 +/- 0.0090 | **0.6035 +/- 0.0136** | 0.5289 |
| GPT-5.1 | 0.6328 +/- 0.0091 | 0.6928 +/- 0.0210 | 0.5883 +/- 0.0376 | 0.4239 |
| GPT-5.2 | 0.6713 +/- 0.0080 | 0.6868 +/- 0.0147 | 0.6158 +/- 0.0140 | 0.5919 |

Paired by trial (n = 5, per-strain test PR-AUC):

| Comparison | Mean diff | Wins | p (paired t) |
|---|---|---|---|
| Free-form GPT-only vs MolE, GPT-4.1 | +0.0032 | 3/5 | 0.803 |
| Free-form GPT-only vs MolE, GPT-5.2 | -0.0105 | 1/5 | 0.384 |
| Free-form GPT-only vs MolE, GPT-4.1-mini | -0.0114 | 1/5 | 0.346 |
| Free-form GPT-only vs MolE, GPT-4o | -0.0242 | 1/5 | 0.108 |
| Free-form GPT-only vs MolE, GPT-4o-mini | -0.0383 | 0/5 | 0.009 |
| Free-form GPT-only vs MolE, GPT-5.1 | -0.0491 | 0/5 | 0.011 |
| Guided vs free-form (MolE + GPT-5.1) | **+0.0334** | 5/5 | **0.003** |
| Guided vs free-form (MolE + GPT-4.1-mini) | +0.0091 | 4/5 | 0.266 |
| Guided vs free-form (MolE + GPT-5.2) | -0.0234 | 1/5 | 0.090 |
| Guided-only vs free-form-only (worst, GPT-4o) | **-0.1198** | 0/5 | **0.001** |
| Guided-only vs free-form-only (best, GPT-5.1) | -0.0445 | 1/5 | 0.058 |
| Best guided (MolE + GPT-4.1-mini) vs RDKit | -0.0042 | 3/5 | 0.687 |
| Best free-form (MolE + GPT-5.2) vs best guided | +0.0167 | 3/5 | 0.114 |

**What this changes.**

1. **Free-form GPT features alone are far stronger than the submitted guided-only numbers
   suggest.** The paper's "guided GPT-only configurations fall to 0.4239-0.5919" describes the
   guided schema; free-form GPT alone reaches 0.633-0.685. Every model gains 0.04-0.12 by
   dropping the RDKit-shaped schema, and five of the six comparisons are significant at n = 5.
   The honest statement is that constraining GPT to an RDKit-style schema **costs** predictive
   signal when GPT features stand alone.
2. **GPT-4.1 features alone match MolE alone** (0.6850 vs 0.6818; +0.003, 3/5 wins, p = 0.80),
   without a pretrained graph encoder. The submission's conclusion that "GPT descriptors alone do
   not substitute for the pretrained graph representation" was drawn from the guided runs and
   does not hold for free-form features.
3. **Combined with MolE, guided and free-form are indistinguishable overall** (four of six
   comparisons |diff| < 0.01, p > 0.25), with one real exception: GPT-5.1, the weakest free-form
   row, improves by +0.033 in 5/5 trials under the guided schema. Its 509 free-form features
   evidently hurt, and constraining them helps.
4. **No guided row beats RDKit.** The best, MolE + GPT-4.1-mini at 0.6935, is 0.004 below RDKit
   alone (p = 0.69), and all six guided rows now sit in 0.678-0.694, inside each other's spreads.
   The submitted guided spread (0.670-0.718) was partly split and flattening artefacts.
5. The submitted guided numbers are all 0.01-0.03 higher than these for MolE + GPT and 0.03-0.16
   lower for GPT-only, consistent with the old 195-compound test split and the inconsistent
   flattening rather than with any property of the features.

### 8a. MolE's full original protocol, 5 trials - DONE

`scripts/run_mole_protocol_search.py --rows mole_only,mole_plus_gpt-5.2,gpt_only_gpt-5.2
--trials 0,1,2,3,4 --device cuda` on a Kaggle GPU: 375 fits, 1.3 GPU-hours. Results:
`results/mole_protocol_trials/` (`fits.csv` has every fit, `selected_per_trial.csv` the
selection, `table_mole_protocol.csv` the summary).

This is MolE's search, not ours. Per trial k, 5 configurations are drawn with seed k from MolE's
space (n_estimators {30,100,300,500,1000}, max_depth {5,10,50,100}, eta {0.05,0.1,0.3,1.0},
subsample {0.3,0.5,0.8,1.0}) and shared by all three rows so comparisons stay paired; each
configuration is fitted 5 times; the one with the best mean validation PR-AUC is selected. Raw
features with fillna(0), no scaling, no early stopping, no class weighting, pooled
PR-AUC = `auc(recall, precision)`, exactly as in MolE's notebooks.

**Pooled test PR-AUC +/- SD over 5 trials.** The same rows under our protocol (Task 1 / 8b,
pooled column, so the metric matches) are shown beside them:

| Row | MolE's protocol | Our protocol | Protocol effect |
|---|---|---|---|
| MolE | 0.7146 +/- 0.0211 | 0.6734 +/- 0.0259 | **+0.041** |
| MolE + GPT-5.2 | **0.7361 +/- 0.0291** | 0.7055 +/- 0.0171 | +0.031 |
| GPT-5.2 only | 0.6362 +/- 0.0145 | 0.6619 +/- 0.0062 | -0.026 |

Per trial (selected configuration's mean test PR-AUC over its 5 fits):

| Trial | MolE | MolE + GPT-5.2 | GPT-5.2 only |
|---|---|---|---|
| 0 | 0.6968 | 0.7020 | 0.6253 |
| 1 | 0.7395 | 0.7201 | 0.6215 |
| 2 | 0.6948 | 0.7652 | 0.6512 |
| 3 | 0.7073 | 0.7254 | 0.6520 |
| 4 | 0.7344 | 0.7679 | 0.6309 |

Paired against MolE alone (n = 5):

| Comparison | Mean diff | SD | Wins | p (paired t) | p (Wilcoxon) |
|---|---|---|---|---|---|
| MolE + GPT-5.2 vs MolE | +0.0216 | 0.0335 | 4/5 | 0.223 | 0.313 |
| GPT-5.2 only vs MolE | **-0.0784** | 0.0316 | 0/5 | **0.005** | 0.063 |

**What this changes.**

1. **The GPT gain survives the protocol change, but stops being significant under it.**
   MolE + GPT-5.2 beats MolE by +0.022 in 4 of 5 trials here, against +0.032 pooled (5/5,
   p = 0.011 on per-strain AP) under our protocol. MolE's protocol roughly doubles the
   trial-to-trial SD of the paired difference (0.034 vs 0.016), so the same-sized effect no
   longer separates from zero at n = 5. The honest claim is a small, consistently positive
   effect that one protocol resolves and the other does not.
2. **Protocol choice is not neutral between representations.** MolE's recipe is worth about
   +0.04 to MolE and -0.03 to GPT-5.2 alone. Deep unregularised trees on raw features suit the
   1000-d embeddings; the 45 GPT features need the scaling, class weighting and early stopping
   our pipeline provides. This is a reason to report both protocols rather than adopt either as
   "the" protocol.
3. **MolE's published 0.7797 is a draw from a very wide distribution, not a protocol advantage
   we failed to match.** Across the 125 MolE-alone fits, test PR-AUC spans 0.229 to 0.810
   (validation 0.230 to 0.713); the best (trial, configuration) cell averages 0.7747. MolE's
   winning configuration (1000 trees, depth 100, eta 0.1, subsample 1.0) was not drawn once in
   25 draws, matching the ~1.6% per-trial chance estimated in Task 0. Our 5 trials of their own
   protocol average 0.7146.
4. **Validation barely ranks test, under their protocol too.** Within a trial, across the 5
   configurations, Spearman rho(val, test) averages 0.44 for MolE, 0.78 for MolE + GPT-5.2 and
   0.72 for GPT-5.2 only; selecting on test rather than validation would gain MolE up to +0.066
   in a single trial. The inverted validation/test ordering the reviewers objected to is a
   property of a 120-compound test set under a 5-draw search, not of GPT features.
5. **A seventh of MolE's space cannot work.** 50 of 375 fits score below 0.40, all at eta = 1.0
   or eta = 0.3 with 500 deep trees. Mean test by eta: 0.697 (0.05), 0.697 (0.10), 0.646 (0.30),
   0.447 (1.00). A 5-draw search spends much of its budget there, which is why its selected test
   values swing by +/-0.1.

**Preprocessing, against Table 1.** The per-column transform of the GPT block is identical in
the two protocols: both zero-fill and neither scales nor adds missingness indicators, and on the
columns both keep the matrices are bit-identical (GPT-5.2 has no missing values, so neither
imputer fires). The feature *set* differs by nine columns of 220: Table 1 drops seven
near-constant descriptors through its 1%-non-zero filter (three of them zero for every compound,
the other four non-zero for 1--8 of 1,193) and carries two metadata columns, `smiles_resolution`
(constant, imputed to zero, never split on) and `input` (a two-valued artefact of coercing the
generation payload to a number; 0.01% of total gain). The MolE and RDKit blocks are
median-imputed, indicator-augmented and standardised in Table 1 and only zero-filled here, but
MolE has no missing training values, the 65 dimensions Table 1 drops are exactly its
training-constant ones, and standardising is monotone per feature, so it cannot move a tree
model. The protocol effect is therefore not a preprocessing artefact. Details and the
column-by-column difference: FINDINGS 3f and `data/gpt_feature_set_table1_vs_task8a.csv`.

**Caveats.** GPU `hist` throughout; MolE used CPU `exact`, worth +0.015 on their own
configuration (0.7802 vs 0.7655). Three of the 5 MolE trials selected a subsample = 1.0
configuration, which is deterministic (fit-to-fit SD exactly 0), so the SD in the table is
search variance, not seed variance. For the selected configurations `auc(recall, precision)` and
`average_precision_score` agree to 4e-4.

Single-configuration comparison for reference (`results/mole_method_all_rows/`, FINDINGS 3c):
with MolE's one published configuration applied to every row, MolE + GPT-5.2 scores 0.7331
against MolE's 0.7527, -0.019 with a bootstrap 95% CI of [-0.059, +0.022] - the same small
difference, with the sign flipped by which single configuration is used.
