## Task 8. Optional new runs

**Not run in this round.** Both need a GPU (Kaggle) rather than the local CPU, and 8b has an input-provenance problem that must be settled first.

### 8a. MolE-only vs MolE+GPT-5.2 under the original MolE protocol

Closest existing evidence (already committed, `results/mole_method_all_rows/`, FINDINGS section 3c): every row trained with **MolE's selected configuration** (1000 trees, depth 100, eta 0.1, subsample 1.0, raw features, no early stopping, pooled PR-AUC) on the Kaggle GPU (`hist`). That configuration is deterministic, so uncertainty comes from a compound-level bootstrap of the 120 test compounds (1000 resamples, shared across rows):

| Representation | Val PR-AUC | Test PR-AUC (pooled) | Test 95% CI | Diff vs MolE [95% CI] |
|---|---|---|---|---|
| MolE | 0.6960 | 0.7527 | [0.609, 0.849] | - |
| MolE + GPT-5.2 | 0.6744 | 0.7331 | [0.592, 0.838] | -0.019 [-0.059, +0.022] |

So with MolE's own tuned configuration, adding GPT-5.2 features does **not** help (a small, non-significant drop). Caveat: that configuration was selected on MolE's validation set, which favours MolE. The fair version (MolE's search space and 5x5 search run per row, 3-5 trials) needs a new driver around `scripts/reproduce_mole_original.py`; estimated 3-5 GPU-hours for the two rows. Ready to prepare if wanted.

### 8b. Guided variants under the 5-trial protocol

Blocked on provenance. The submitted guided runs used the old 195-compound split, and their feature inputs cannot all be traced: `data/gpt_feature_generation/` holds two candidate files for GPT-4.1-mini and GPT-5.1; the GPT-5.1 run matches `gpt_features_5_1_gpt-5.1_unified.csv`, the GPT-4.1-mini run matches neither file by feature name, and the GPT-4o run kept no feature list. Once the six guided input files are confirmed, `scripts/run_table1.py --trials 0,1,2,3,4 --device cuda` can run them as 12 extra rows (6 MolE+guided, 6 guided-only), about the same GPU time as the main table.
