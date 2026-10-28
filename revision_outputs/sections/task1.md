## Task 1. Per-trial values and paired tests

Per-trial values for all 9 rows x 5 trials: `data/per_trial_results.csv`. `val_prauc_perstrain` is the validation PR-AUC of the final trial-k model (the value behind the main table's validation column); the extra column `val_prauc_selection_mean` is the mean over the three search fits (seeds 0,1,2) that selected the configuration.

**Is the train/val/test scaffold split identical across all 5 trials?** **Yes.** All 45 runs (9 rows x 5 trials) wrote the split they used; 1 distinct assignment(s). It is MolE's published scaffold split (`maier_scaffold_split.tsv`, 957 train / 120 valid / 120 test compounds), fixed for every trial. Trials differ only in the hyperparameter configurations sampled and the training seed.

Per-trial test PR-AUC (mean per-strain):

| Representation | Trial 0 | Trial 1 | Trial 2 | Trial 3 | Trial 4 | Mean ± SD |
|---|---|---|---|---|---|---|
| MolE | 0.6842 | 0.6822 | 0.7093 | 0.6887 | 0.6447 | 0.6818 ± 0.0234 |
| RDKit descriptors | 0.7002 | 0.7032 | 0.6918 | 0.7066 | 0.6866 | 0.6977 ± 0.0083 |
| MolE + RDKit | 0.6803 | 0.6852 | 0.6766 | 0.7126 | 0.6757 | 0.6861 ± 0.0153 |
| MolE + GPT-4o-mini | 0.6944 | 0.6734 | 0.7028 | 0.6993 | 0.6425 | 0.6825 ± 0.0251 |
| MolE + GPT-4.1-mini | 0.6764 | 0.6785 | 0.6763 | 0.7061 | 0.6850 | 0.6844 ± 0.0126 |
| MolE + GPT-4o | 0.6988 | 0.6825 | 0.6865 | 0.6968 | 0.6841 | 0.6898 ± 0.0075 |
| MolE + GPT-4.1 | 0.6898 | 0.6846 | 0.6996 | 0.7214 | 0.6858 | 0.6962 ± 0.0153 |
| MolE + GPT-5.1 | 0.6434 | 0.6486 | 0.6933 | 0.6721 | 0.6394 | 0.6594 ± 0.0228 |
| MolE + GPT-5.2 | 0.7263 | 0.7004 | 0.7192 | 0.7188 | 0.6864 | 0.7102 ± 0.0164 |

Paired comparisons on test PR-AUC (per-strain), pairs = same trial index, n = 5:

| Comparison | Mean diff | SD of diff | Wins | t | p (paired t) | p (Wilcoxon) |
|---|---|---|---|---|---|---|
| MolE + GPT-5.2 vs RDKit descriptors | +0.0126 | 0.0142 | 3/5 | 1.976 | 0.119 | 0.312 |
| MolE + GPT-5.2 vs MolE | +0.0284 | 0.0142 | 5/5 | 4.459 | 0.011 | 0.062 |
| MolE + GPT-4.1 vs RDKit descriptors | -0.0014 | 0.0135 | 2/5 | -0.236 | 0.825 | 0.812 |
| MolE + GPT-5.1 vs RDKit descriptors | -0.0383 | 0.0239 | 1/5 | -3.578 | 0.023 | 0.125 |
| MolE + RDKit vs RDKit descriptors | -0.0116 | 0.0104 | 1/5 | -2.491 | 0.067 | 0.125 |

With n = 5 the two-sided exact Wilcoxon test cannot go below p = 0.0625 (all five differences with the same sign), so it can never reach 0.05. No correction for multiple comparisons is applied above.
