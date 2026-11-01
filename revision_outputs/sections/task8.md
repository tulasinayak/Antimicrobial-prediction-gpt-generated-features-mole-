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

### 8a. MolE's full original protocol — not yet run

`scripts/run_mole_protocol_search.py` and `kaggle/task8a_mole_protocol.ipynb` are ready: per
trial, 5 configurations drawn from MolE's space (n_estimators 30-1000, max_depth 5-100, eta
0.05-1, subsample 0.3-1), shared across rows, 5 fits each, selection on mean validation PR-AUC,
raw features, no early stopping, pooled PR-AUC. Rows: MolE, MolE + GPT-5.2, GPT-5.2 only.

Closest existing evidence (`results/mole_method_all_rows/`, FINDINGS 3c): with MolE's single
published configuration, MolE + GPT-5.2 scores 0.7331 against MolE's 0.7527, a difference of
-0.019 with a bootstrap 95% CI of [-0.059, +0.022].
