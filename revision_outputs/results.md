# Revision analyses: GPT as a Molecular Feature Proposer

All numbers below come from the scripts in `scripts/` (run from the repository root, in order `t1` to `t6`, then `assemble.py`). The main 5-trial results were not changed.

## Summary: what changed or surprised us

1. **Figure 2 compared GPT with a different molecule for salts.** The 98 RDKit descriptors (MolE's chemDesc set) were computed on PubChem's *salted* SMILES, while GPT-5.2 received the *desalted* SMILES; for 433 of 1,193 compounds the molecular weights differ by more than 1 Da. Against RDKit on the same input, GPT-5.2's fidelity is much higher: Spearman rho 0.920 (MW, was 0.799), 0.935 (TPSA, was 0.753), 0.937 (FractionCSP3, was 0.927); MAE 59 g/mol (was 94) and 17 A^2 (was 30). `fig2_fidelity` uses the same-input reference; `fig2b` keeps the submitted one. (Task 5)
2. **The submitted MIC result (rho = -0.480) came from an older single-run model**, not from the 5-trial protocol. It reproduces exactly. With the 5 trial models, MolE alone (rho = -0.526 +/- 0.045) and RDKit alone (-0.533 +/- 0.043) correlate with MIC at least as well as MolE+GPT-5.2 (-0.479 +/- 0.054): the external check gives no evidence that GPT features add signal. Gram-matched G+/G- scores strengthen every correlation (about -0.55 to -0.60). (Task 3)
3. **The paper's OLS slope (-0.200) was in natural-log units**; in the log2 units of the paper's own equation it is -0.288. MolE's released AP scores also use the natural log, despite log2 in their equation. Spearman rho is unaffected. (Task 3)
4. **3 of the 38 MIC observations are not in ug/mL** according to ST10's notes (Everolimus in nM, Moxidectin twice in mM). They were included unconverted, as in the submitted and MolE analyses. (Task 3)
5. **Point count explained:** 38 observations, 37 distinct positions (Mequitazine has the same MIC for both gram classes), and 4 more pairs overlap at marker size, which leaves the 33-35 points a reader can count. Nothing was dropped. (Task 3)
6. **Paired tests (n = 5):** MolE+GPT-5.2 beats MolE in 5/5 trials (+0.028, paired t p = 0.011, Wilcoxon p = 0.062, its floor at n = 5) but not RDKit (+0.013, 3/5, p = 0.119). MolE+GPT-5.1 is worse than RDKit (-0.038, p = 0.023). The scaffold split is identical across all 45 runs. (Task 1)
7. **Vocabulary divergence survives lenient matching:** max pairwise Jaccard 0.119 exact, 0.133 normalized, 0.190 embedding (cosine >= 0.85); 33-82% of each model's features are model-specific even under embedding matching. The embedding level also over-matches some pairs (e.g. `num_secondary_nitrogens` ~ `num_basic_nitrogens`), so it is a lenient bound. (Task 2)
8. **GPT features carry a modest share of the model:** in MolE+GPT-5.2, GPT features hold 15.4% of total gain (top 10: 7.0%) vs 74.5% for MolE, although by per-split gain the top GPT features rank next to the best MolE dimensions. (Task 4)
9. **`exact_mass_est` is GPT's molecular weight restated** (rho = 1.000 with its own MW estimate); it does match RDKit `ExactMolWt` (rho = 0.934), which is not among the 98 descriptors. (Task 5)
10. **Guided GPT-only table: the Test F1 column was a copy of another F1 value**; correct test F1 values are in Task 7. These guided runs also used the old 195-compound split, so the guided tables are not comparable with the main table. (Task 7)
11. **Small corrections:** FractionCSP3 rho is 0.9275 (caption 0.928 vs panel 0.927 is rounding); GPT-5.1's 509 "features" include a non-descriptor flag (`smiles_parsed_successfully`); 7 GPT-5.2 compounds are failed generations (every feature 0); one compound (Prestw-1180) sits on a float-precision split boundary, so re-scoring the GPU models on CPU shifts its probabilities by up to 0.05 (per-strain AP unchanged to 1e-4).
12. **The guided schema costs signal, and GPT-4.1 features alone match MolE alone** (Task 8b, 18 rows x 5 trials on GPU). Free-form GPT-only reaches 0.633-0.685 against the submitted guided-only 0.424-0.592; dropping the RDKit-shaped schema gains 0.04-0.12 per model (5 of 6 significant at n = 5). GPT-4.1 alone scores 0.6850 vs MolE's 0.6818 (p = 0.80), so the submission's "GPT descriptors alone do not substitute for the pretrained graph representation" holds only for the guided variant. Combined with MolE, guided and free-form are indistinguishable except GPT-5.1, where guided wins 5/5 (+0.033, p = 0.003). No guided row beats RDKit.
13. **Under MolE's own full protocol the GPT gain holds but loses significance, and 0.7797 is a lucky draw** (Task 8a, 375 fits on GPU). MolE 0.7146 +/- 0.0211 pooled, MolE+GPT-5.2 0.7361 +/- 0.0291 (+0.022, 4/5, p = 0.22), GPT-5.2 only 0.6362 +/- 0.0145 (-0.078, 0/5, p = 0.005). Their protocol is worth +0.041 to MolE and -0.026 to GPT alone, so it is not neutral between representations, and it doubles the SD of the paired difference. Across the 125 MolE-alone fits test PR-AUC spans 0.229-0.810; MolE's winning configuration was never drawn in 25 draws. Within-trial Spearman(val, test) averages 0.44 for MolE, so validation barely ranks test under their protocol either. (Task 8a)

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

## Task 2. Vocabulary overlap at three levels of strictness

Feature-name lists: `data/feature_names_<model>.txt`. Metadata columns removed: `prestwick_ID`, `ProductName`, `smiles_resolution`, `input` (`ProductName` is absent from the free-form files). Matrices: `data/overlap_<level>.csv` (Jaccard) and `data/overlap_<level>_shared_counts.csv` (shared counts; diagonal = total).

### (a) EXACT

| Model | Total | Model-specific | % specific | Shared with >=1 other |
|---|---|---|---|---|
| GPT-4o-mini | 38 | 24 | 63.2% | 14 |
| GPT-4.1-mini | 123 | 77 | 62.6% | 46 |
| GPT-4o | 266 | 245 | 92.1% | 21 |
| GPT-4.1 | 196 | 144 | 73.5% | 52 |
| GPT-5.1 | 509 | 481 | 94.5% | 28 |
| GPT-5.2 | 218 | 200 | 91.7% | 18 |

Pairwise Jaccard:

| | GPT-4o-mini | GPT-4.1-mini | GPT-4o | GPT-4.1 | GPT-5.1 | GPT-5.2 |
|---|---|---|---|---|---|---|
| GPT-4o-mini | - | 0.032 | 0.031 | 0.045 | 0.002 | 0.008 |
| GPT-4.1-mini | 0.032 | - | 0.024 | 0.119 | 0.018 | 0.018 |
| GPT-4o | 0.031 | 0.024 | - | 0.034 | 0.003 | 0.004 |
| GPT-4.1 | 0.045 | 0.119 | 0.034 | - | 0.010 | 0.007 |
| GPT-5.1 | 0.002 | 0.018 | 0.003 | 0.010 | - | 0.021 |
| GPT-5.2 | 0.008 | 0.018 | 0.004 | 0.007 | 0.021 | - |

Off-diagonal Jaccard: min 0.002, max 0.119, mean 0.025. Model-specific share: 62.6%-94.5%.

### (b) NORMALIZED

| Model | Total | Model-specific | % specific | Shared with >=1 other |
|---|---|---|---|---|
| GPT-4o-mini | 38 | 19 | 50.0% | 19 |
| GPT-4.1-mini | 122 | 75 | 61.5% | 47 |
| GPT-4o | 266 | 243 | 91.4% | 23 |
| GPT-4.1 | 194 | 138 | 71.1% | 56 |
| GPT-5.1 | 508 | 460 | 90.6% | 48 |
| GPT-5.2 | 218 | 180 | 82.6% | 38 |

Pairwise Jaccard:

| | GPT-4o-mini | GPT-4.1-mini | GPT-4o | GPT-4.1 | GPT-5.1 | GPT-5.2 |
|---|---|---|---|---|---|---|
| GPT-4o-mini | - | 0.053 | 0.041 | 0.055 | 0.015 | 0.036 |
| GPT-4.1-mini | 0.053 | - | 0.029 | 0.133 | 0.026 | 0.027 |
| GPT-4o | 0.041 | 0.029 | - | 0.041 | 0.010 | 0.013 |
| GPT-4.1 | 0.055 | 0.133 | 0.041 | - | 0.020 | 0.022 |
| GPT-5.1 | 0.015 | 0.026 | 0.010 | 0.020 | - | 0.046 |
| GPT-5.2 | 0.036 | 0.027 | 0.013 | 0.022 | 0.046 | - |

Off-diagonal Jaccard: min 0.010, max 0.133, mean 0.038. Model-specific share: 50.0%-91.4%.

Totals shrink where several names of one model normalise to the same key (collapsed per model: GPT-4o-mini 0, GPT-4.1-mini 1, GPT-4o 0, GPT-4.1 2, GPT-5.1 1, GPT-5.2 0).

Normalisation rules: split camelCase and on any non-alphanumeric separator; lower-case; drop hedge tokens `approx`, `approximate`, `calc`, `calculated`, `est`, `estimate`, `estimated`, `index`, `pred`, `predicted`, `score`, `value`; rename tokens `num`->`n`, `number`->`n` (`count` kept); join without separators; then apply the synonym map:

| Canonical | Normalised forms mapped to it |
|---|---|
| `logp` | `logp`, `clogp`, `xlogp`, `xlogp3`, `mollogp` |
| `tpsa` | `tpsa`, `topologicalpolarsurfacearea` |
| `mw` | `mw`, `molecularweight`, `molwt` |
| `hbd` | `hbd`, `hbdcount`, `nhbd`, `hbonddonors`, `hbonddonor`, `hbonddonorcount`, `nhbonddonors`, `nhdonors`, `hydrogenbonddonors`, `hydrogenbonddonorcount`, `nhydrogenbonddonors` |
| `hba` | `hba`, `hbacount`, `nhba`, `hbondacceptors`, `hbondacceptor`, `hbondacceptorcount`, `nhbondacceptors`, `nhacceptors`, `hydrogenbondacceptors`, `hydrogenbondacceptorcount`, `nhydrogenbondacceptors` |
| `fsp3` | `fsp3`, `fractioncsp3`, `fractionsp3`, `csp3fraction` |
| `rotb` | `rotatablebonds`, `nrotatablebonds`, `rotatablebondcount`, `nrotatablebond`, `rotbonds`, `nrotbonds` |

Examples (random): `e_state_polarity_index` -> `estatepolarity`; `fraction_rigid_bonds` -> `fractionrigidbonds`; `num_nitro_groups` -> `nnitrogroups`; `num_hydrophilic_groups` -> `nhydrophilicgroups`; `similarity_to_tetracyclines_index` -> `similaritytotetracyclines`; `cytosolic_targeting_likelihood_index` -> `cytosolictargetinglikelihood`; `num_neutral_centers_pH_5_5` -> `nneutralcentersph55`; `num_ring_rigid_features` -> `nringrigidfeatures`; `logS_to_logP_ratio` -> `logstologpratio`; `exact_mass` -> `exactmass`; `hbond_acceptor_sites_ring_n` -> `hbondacceptorsitesringn`; `num_lactone_bonds` -> `nlactonebonds`

### (c) EMBEDDING, cosine >= 0.80

| Model | Total | Model-specific | % specific | Shared with >=1 other |
|---|---|---|---|---|
| GPT-4o-mini | 38 | 13 | 34.2% | 25 |
| GPT-4.1-mini | 123 | 28 | 22.8% | 95 |
| GPT-4o | 266 | 181 | 68.0% | 85 |
| GPT-4.1 | 196 | 93 | 47.4% | 103 |
| GPT-5.1 | 509 | 379 | 74.5% | 130 |
| GPT-5.2 | 218 | 158 | 72.5% | 60 |

Pairwise Jaccard:

| | GPT-4o-mini | GPT-4.1-mini | GPT-4o | GPT-4.1 | GPT-5.1 | GPT-5.2 |
|---|---|---|---|---|---|---|
| GPT-4o-mini | - | 0.095 | 0.048 | 0.083 | 0.017 | 0.036 |
| GPT-4.1-mini | 0.095 | - | 0.096 | 0.227 | 0.060 | 0.049 |
| GPT-4o | 0.048 | 0.096 | - | 0.095 | 0.024 | 0.010 |
| GPT-4.1 | 0.083 | 0.227 | 0.095 | - | 0.054 | 0.032 |
| GPT-5.1 | 0.017 | 0.060 | 0.024 | 0.054 | - | 0.071 |
| GPT-5.2 | 0.036 | 0.049 | 0.010 | 0.032 | 0.071 | - |

Off-diagonal Jaccard: min 0.010, max 0.227, mean 0.067. Model-specific share: 22.8%-74.5%.

### (c) EMBEDDING, cosine >= 0.85

| Model | Total | Model-specific | % specific | Shared with >=1 other |
|---|---|---|---|---|
| GPT-4o-mini | 38 | 14 | 36.8% | 24 |
| GPT-4.1-mini | 123 | 41 | 33.3% | 82 |
| GPT-4o | 266 | 218 | 82.0% | 48 |
| GPT-4.1 | 196 | 109 | 55.6% | 87 |
| GPT-5.1 | 509 | 411 | 80.7% | 98 |
| GPT-5.2 | 218 | 168 | 77.1% | 50 |

Pairwise Jaccard:

| | GPT-4o-mini | GPT-4.1-mini | GPT-4o | GPT-4.1 | GPT-5.1 | GPT-5.2 |
|---|---|---|---|---|---|---|
| GPT-4o-mini | - | 0.088 | 0.048 | 0.083 | 0.011 | 0.028 |
| GPT-4.1-mini | 0.088 | - | 0.049 | 0.190 | 0.039 | 0.046 |
| GPT-4o | 0.048 | 0.049 | - | 0.065 | 0.014 | 0.006 |
| GPT-4.1 | 0.083 | 0.190 | 0.065 | - | 0.032 | 0.020 |
| GPT-5.1 | 0.011 | 0.039 | 0.014 | 0.032 | - | 0.057 |
| GPT-5.2 | 0.028 | 0.046 | 0.006 | 0.020 | 0.057 | - |

Off-diagonal Jaccard: min 0.006, max 0.190, mean 0.052. Model-specific share: 33.3%-82.0%.

### (c) EMBEDDING, cosine >= 0.90

| Model | Total | Model-specific | % specific | Shared with >=1 other |
|---|---|---|---|---|
| GPT-4o-mini | 38 | 18 | 47.4% | 20 |
| GPT-4.1-mini | 123 | 59 | 48.0% | 64 |
| GPT-4o | 266 | 239 | 89.8% | 27 |
| GPT-4.1 | 196 | 133 | 67.9% | 63 |
| GPT-5.1 | 509 | 443 | 87.0% | 66 |
| GPT-5.2 | 218 | 177 | 81.2% | 41 |

Pairwise Jaccard:

| | GPT-4o-mini | GPT-4.1-mini | GPT-4o | GPT-4.1 | GPT-5.1 | GPT-5.2 |
|---|---|---|---|---|---|---|
| GPT-4o-mini | - | 0.066 | 0.041 | 0.049 | 0.009 | 0.024 |
| GPT-4.1-mini | 0.066 | - | 0.029 | 0.139 | 0.028 | 0.040 |
| GPT-4o | 0.041 | 0.029 | - | 0.043 | 0.008 | 0.006 |
| GPT-4.1 | 0.049 | 0.139 | 0.043 | - | 0.019 | 0.010 |
| GPT-5.1 | 0.009 | 0.028 | 0.008 | 0.019 | - | 0.045 |
| GPT-5.2 | 0.024 | 0.040 | 0.006 | 0.010 | 0.045 | - |

Off-diagonal Jaccard: min 0.006, max 0.139, mean 0.037. Model-specific share: 47.4%-89.8%.

20 random matched pairs at cosine >= 0.85 (drawn from the 151 matched pairs whose names are not identical; 126 further matches are identical names):

| Model A | Feature A | Model B | Feature B | Cosine |
|---|---|---|---|---|
| GPT-4o | `num_secondary_nitrogens` | GPT-5.1 | `num_basic_nitrogens` | 0.860 |
| GPT-4.1 | `num_nitrogens` | GPT-5.1 | `num_neutral_nitrogens` | 0.892 |
| GPT-4o-mini | `num_hydrogen_bond_acceptors` | GPT-4o | `num_h_bond_acceptors` | 0.913 |
| GPT-4.1-mini | `num_rotatable_bonds` | GPT-5.1 | `num_rotatable_bonds_relaxed` | 0.912 |
| GPT-5.1 | `aromatic_ring_count` | GPT-5.2 | `ring_count_aromatic` | 0.970 |
| GPT-4.1 | `isoelectric_point` | GPT-5.2 | `isoelectric_point_est` | 0.939 |
| GPT-4o | `num_non_aromatic_bonds` | GPT-5.1 | `num_c_hetero_aromatic_bonds` | 0.867 |
| GPT-4.1-mini | `formal_charge` | GPT-5.2 | `formal_charge_total` | 0.863 |
| GPT-4.1 | `exact_mass` | GPT-5.2 | `exact_mass_est` | 0.887 |
| GPT-4o | `num_primary_nitrogens` | GPT-4.1 | `num_basic_nitrogens` | 0.895 |
| GPT-5.1 | `toxicity_hepatotoxicity_risk_index` | GPT-5.2 | `hepatotoxicity_risk_est` | 0.882 |
| GPT-4.1-mini | `num_fused_heteroaromatic_rings` | GPT-4.1 | `num_ring_fused_heterocycles` | 0.885 |
| GPT-5.1 | `ring_count` | GPT-5.2 | `ring_count_total` | 0.959 |
| GPT-4.1-mini | `num_basic_sulfurs` | GPT-4o | `num_sulfur_atoms` | 0.863 |
| GPT-4.1-mini | `num_nonaromatic_nitrogens` | GPT-5.1 | `num_basic_nitrogens` | 0.868 |
| GPT-4o-mini | `num_nitrogens` | GPT-5.1 | `num_basic_nitrogens` | 0.935 |
| GPT-4.1-mini | `num_pi_electrons` | GPT-5.1 | `num_electrons` | 0.884 |
| GPT-5.1 | `exact_mass` | GPT-5.2 | `exact_mass_est` | 0.887 |
| GPT-4.1-mini | `num_nonring_nitrogens` | GPT-4o | `num_primary_nitrogens` | 0.854 |
| GPT-4.1-mini | `aromatic_ring_count` | GPT-5.2 | `ring_count_aromatic` | 0.970 |

## Task 3. External MIC check

Observations rebuilt exactly as in the submitted analysis: ST10 rows predicted by MolE, one observation per reported gram-positive / gram-negative MIC, MIC <= 128, inner-joined to the 29 compounds with GPT-5.2 features -> **38 observations from 29 compounds** (25 gram-positive, 13 gram-negative). `data/mic_points.csv` holds every observation with all scores.

**Which model produced the submitted rho = -0.480?** An older one: the single-run MolE+GPT-5.2 model in `results/freeform/gpt-5.2/` (the pre-revision protocol), applied with MolE's own MedChemExpress embeddings. Re-scoring with that model reproduces the stored scores exactly (max |diff| = 5.4e-07 on the natural-log scale it used). It is **not** one of the 5-trial models.

Inputs for the 29 compounds: MolE embeddings and RDKit descriptors from MolE's repository (`data/04.new_predictions/*_medchemexpress.tsv.gz`), GPT-5.2 features from this project. MolE's descriptor file lacks Tannic acid; its 98 descriptors were computed here with RDKit from the same SMILES. Scores: G = (1/40) sum log2 p; G+ / G- over the gram-positive / gram-negative strains below.

### 3a/3b. Spearman correlation with log2 MIC (n = 38)

Trial-0 models of the 5-trial protocol, plus the submitted model for reference. 'Matched' uses G+ for observations on gram-positive organisms and G- for gram-negative.

| Model | rho (G) | p (G) | rho (G+/G- matched) | p (matched) | OLS slope of G on log2 MIC (p) |
|---|---|---|---|---|---|
| MolE + GPT-5.2 (submitted model) | -0.480 | 0.0023 | -0.559 | 0.0003 | -0.288 (0.0029) |
| MolE + GPT-5.2 (trial 0) | -0.493 | 0.0017 | -0.575 | 0.0002 | -0.300 (0.0037) |
| MolE (trial 0) | -0.457 | 0.0039 | -0.535 | 0.0005 | -0.376 (0.0032) |
| RDKit descriptors (trial 0) | -0.525 | 0.0007 | -0.587 | 0.0001 | -0.141 (0.0034) |

Across the 5 trial models (mean ± SD of rho; range):

| Representation | rho (G) | range | rho (G+/G- matched) | range | trials with p < 0.05 (G) |
|---|---|---|---|---|---|
| MolE + GPT-5.2 | -0.479 ± 0.054 | [-0.524, -0.386] | -0.555 ± 0.051 | [-0.596, -0.467] | 5/5 |
| MolE | -0.526 ± 0.045 | [-0.570, -0.457] | -0.600 ± 0.054 | [-0.657, -0.535] | 5/5 |
| RDKit descriptors | -0.533 ± 0.043 | [-0.579, -0.468] | -0.597 ± 0.040 | [-0.645, -0.553] | 5/5 |

The submitted rho = -0.480 is reproduced by the submitted model (-0.480). Its OLS slope in log2 units is -0.288; the paper's -0.200 was in natural-log units (-0.200 / ln 2 = -0.289).

Strains counted as gram-positive / gram-negative (recovered from MolE's published G+/G- scores, which are exact linear functions of these subsets; `data/strain_gram.csv`):

- **Gram-positive (22):** Bifidobacterium adolescentis (NT5022); Bifidobacterium longum (NT5028); Blautia obeum (NT5069); Clostridium bolteae (NT5026); Clostridium difficile (NT5083); Clostridium perfringens (NT5032); Clostridium ramosum (NT5006); Clostridium saccharolyticum (NT5037); Collinsella aerofaciens (NT5073); Coprococcus comes (NT5048); Dorea formicigenerans (NT5076); Eggerthella lenta (NT5024); Eubacterium eligens (NT5075); Eubacterium rectale (NT5009); Lactobacillus paracasei (NT5042); Roseburia hominis (NT5079); Roseburia intestinalis (NT5011); Ruminococcus bromii (NT5045); Ruminococcus gnavus (NT5046); Ruminococcus torques (NT5047); Streptococcus parasanguinis (NT5072); Streptococcus salivarius (NT5038)
- **Gram-negative (18):** Akkermansia muciniphila (NT5021); Bacteroides caccae (NT5050); Bacteroides fragilis (ET) (NT5033); Bacteroides fragilis (NT) (NT5003); Bacteroides ovatus (NT5054); Bacteroides thetaiotaomicron (NT5004); Bacteroides uniformis (NT5002); Bacteroides vulgatus (NT5001); Bacteroides xylanisolvens (NT5064); Bilophila wadsworthia (NT5036); Escherichia coli ED1a (NT5078); Escherichia coli IAI1 (NT5077); Fusobacterium nucleatum (NT5025); Odoribacter splanchnicus (NT5081); Parabacteroides distasonis (NT5074); Parabacteroides merdae (NT5071); Prevotella copri (NT5019); Veillonella parvula (NT5017)

Data caveat: 3 of the 38 observations carry an ST10 note that the MIC is not in ug/mL (Everolimus [positive]: 'nM'; Moxidectin [positive]: 'NA, mM'; Moxidectin [negative]: 'NA, mM'). They are kept unconverted, as in the submitted analysis and in MolE's ST10 filtering.

### 3c. Point count

The figure plots 38 observations. Unique (log2 MIC, G) positions: **37**.

Exactly coincident observations (same compound, identical MIC for both gram classes):

- Mequitazine: log2 MIC = 5.966, G = -2.801, grams positive, negative

Visually overlapping pairs (within 2% of the axis range in both x and y; markers in the submitted plot are larger than that): 4

- Diiodohydroxyquinoline (positive) and Fenticonazole (Nitrate) (negative): log2 MIC -3.00 / -3.00, G -2.017 / -2.080
- Doxifluridine (positive) and Nimorazole (negative): log2 MIC 0.00 / 0.07, G -1.895 / -1.801
- Moxidectin (negative) and Flupentixol dihydrochloride (negative): log2 MIC 3.32 / 3.32, G -2.834 / -2.797
- Flupentixol dihydrochloride (positive) and C16-PAF (positive): log2 MIC 2.32 / 2.32, G -2.797 / -2.757

So 38 observations collapse to 37 distinct positions, and 4 further pairs overlap at marker size, which accounts for the 34-35 points a reader can count. No observations were dropped or duplicated.

## Task 4. Feature importance, MolE + GPT-5.2 (5 trial models)

XGBoost gain importance (`importance_type='gain'`, average gain per split, as in the submitted table), mean ± SD over the 5 trial models; a trial that never splits on a feature counts as 0. Top 15 GPT-5.2 features and the 3 highest-gain MolE dimensions, ranked together. `max |rho|` is the largest absolute Spearman correlation with any of the 98 RDKit descriptors over the 1,197 Prestwick compounds. Full list: `data/importance_gpt52.csv`.

| Rank | Feature | Block | Mean gain ± SD | Trials used | Max abs(rho) with RDKit | Best RDKit match |
|---|---|---|---|---|---|---|
| 1 | `mole_228` | MolE | 898.2 ± 196.4 | 5/5 | - | - |
| 2 | `heteroatom_count` | GPT-5.2 | 508.8 ± 241.4 | 5/5 | 0.790 | NumHeteroatoms |
| 3 | `mole_68` | MolE | 470.6 ± 205.6 | 5/5 | - | - |
| 4 | `cyp_inhibition_risk_est` | GPT-5.2 | 387.1 ± 250.2 | 5/5 | 0.808 | MolLogP |
| 5 | `mole_261` | MolE | 282.9 ± 96.2 | 5/5 | - | - |
| 6 | `pka_basic_min_est` | GPT-5.2 | 229.9 ± 28.3 | 5/5 | 0.459 | SMR_VSA6 |
| 7 | `solubility_class_est` | GPT-5.2 | 209.1 ± 129.6 | 5/5 | 0.776 | MolLogP |
| 8 | `logp_xlogp3_est` | GPT-5.2 | 183.5 ± 64.7 | 5/5 | 0.882 | MolLogP |
| 9 | `logd7_4_est` | GPT-5.2 | 142.0 ± 88.3 | 5/5 | 0.843 | MolLogP |
| 10 | `desolvation_penalty_est` | GPT-5.2 | 124.7 ± 89.4 | 5/5 | 0.516 | TPSA |
| 11 | `enzyme_active_site_mimicry_score_est` | GPT-5.2 | 102.4 ± 61.9 | 5/5 | 0.626 | NOCount |
| 12 | `strong_acidic_site_count` | GPT-5.2 | 99.7 ± 72.1 | 5/5 | 0.186 | MaxAbsPartialCharge |
| 13 | `biofilm_penetration_score_est` | GPT-5.2 | 83.5 ± 37.4 | 5/5 | 0.427 | Chi2n |
| 14 | `largest_fused_system_ring_count` | GPT-5.2 | 81.5 ± 161.0 | 5/5 | 0.676 | RingCount |
| 15 | `exact_mass_est` | GPT-5.2 | 74.5 ± 33.5 | 5/5 | 0.834 | Chi2v |
| 16 | `ring_strain_est` | GPT-5.2 | 73.8 ± 34.3 | 5/5 | 0.848 | NumAliphaticRings |
| 17 | `nucleoside_mimic_potential` | GPT-5.2 | 71.5 ± 43.8 | 5/5 | 0.494 | NOCount |
| 18 | `aromatic_ring_count_6_membered` | GPT-5.2 | 67.5 ± 100.4 | 5/5 | 0.830 | NumAromaticCarbocycles |

Share of total gain (`total_gain`, summed over all splits, mean over trials):

| Group | Share of total gain |
|---|---|
| All GPT-5.2 features (213) | 15.4% |
| Top-10 GPT-5.2 features | 7.0% |
| All MolE dimensions incl. missingness indicators (935) | 74.5% |
| Strain one-hot (40) | 10.1% |

Top-10 GPT features vs all MolE dimensions: the top-10 GPT features hold 8.6% of their combined total gain, MolE dimensions 91.4%. Per-split `gain` favours features used in few, high-gain splits; `total_gain` reflects overall reliance, which is dominated by the 935 MolE dimensions and the strain identity.

## Task 5. Fidelity of GPT-5.2 features with an RDKit counterpart

**Reference matters.** The 98-descriptor RDKit file (MolE's chemDesc set, used for the submitted Figure 2) was computed on PubChem's *salted* isomeric SMILES, while GPT-5.2 received the *desalted* SMILES. For 433 of 1197 compounds the two molecules differ in MolWt by more than 1 Da (counter-ions such as HCl, maleate, sodium). The primary comparison below therefore computes every RDKit counterpart on the exact SMILES GPT-5.2 was given (`query_smiles` in its `input` column; 4 of 1,197 do not parse in RDKit and are excluded). Columns marked *98-file* repeat the as-submitted comparison for descriptors in that set.

### 5a. Mapping

35 GPT-5.2 features have a direct RDKit counterpart. Not mapped: features whose definition differs from any RDKit descriptor (e.g. `ring_count_bridgehead` vs `NumBridgeheadAtoms`, `ring_count_spiro` vs `NumSpiroAtoms`, `logd7_4_est`, pKa, charge at pH, ADMET scores). `logp_xlogp3_est` is compared with RDKit's Crippen `MolLogP`, a different logP model than XLogP3.

| GPT-5.2 feature | RDKit counterpart | In 98-descriptor set | Category |
|---|---|---|---|
| `molecular_weight` | MolWt | yes | size/mass |
| `exact_mass_est` | ExactMolWt | no | size/mass |
| `molar_refractivity_est` | MolMR | yes | size/mass |
| `atom_count_heavy` | HeavyAtomCount | yes | counts |
| `atom_count_total` | atoms incl. H | no | counts |
| `atom_count_hydrogen` | H count | no | counts |
| `element_count_c` | C count | no | counts |
| `element_count_n` | N count | no | counts |
| `element_count_o` | O count | no | counts |
| `element_count_s` | S count | no | counts |
| `element_count_f` | F count | no | counts |
| `element_count_cl` | Cl count | no | counts |
| `heteroatom_count` | NumHeteroatoms | yes | counts |
| `valence_electron_count` | NumValenceElectrons | yes | counts |
| `formal_charge_total` | formal charge | no | counts |
| `ring_count_total` | RingCount | yes | counts |
| `ring_count_aromatic` | NumAromaticRings | yes | counts |
| `ring_count_aliphatic` | NumAliphaticRings | yes | counts |
| `aromatic_atom_count` | aromatic atoms | no | counts |
| `bond_count_aromatic` | aromatic bonds | no | counts |
| `ring_atom_count` | ring atoms | no | counts |
| `largest_ring_size` | largest ring size | no | counts |
| `bond_count_amide` | NumAmideBonds | yes | counts |
| `rotatable_bond_count` | NumRotatableBonds | yes | counts |
| `hbond_donor_count` | NumHDonors | yes | counts |
| `hbond_acceptor_count` | NumHAcceptors | yes | counts |
| `lipinski_rule_of_5_violations` | Lipinski violations | no | counts |
| `topological_polar_surface_area` | TPSA | yes | polarity/lipophilicity |
| `logp_xlogp3_est` | MolLogP | yes | polarity/lipophilicity |
| `fraction_csp3` | FractionCSP3 | yes | fractions |
| `kappa1_est` | Kappa1 | yes | topological indices |
| `kappa2_est` | Kappa2 | yes | topological indices |
| `kappa3_est` | Kappa3 | yes | topological indices |
| `balaban_j_est` | BalabanJ | no | topological indices |
| `bertz_ct_est` | BertzCT | yes | topological indices |

### 5b. Agreement with RDKit on GPT's input SMILES

`zeros` = GPT values exactly 0 (RDKit exactly 0 in brackets); *(no 0)* columns drop compounds whose GPT value is exactly 0. `exact` = fraction with GPT equal to RDKit (4 decimals).

| GPT-5.2 feature | n | zeros (RDKit) | rho | MAE | median AE | rho (no 0) | MAE (no 0) | median AE (no 0) | exact | rho / MAE vs 98-file |
|---|---|---|---|---|---|---|---|---|---|---|
| `molecular_weight` | 1193 | 3 (0) | 0.920 | 59.2 | 43.1 | 0.934 | 56 | 43.1 | 1.3% | 0.799 / 93.8 |
| `exact_mass_est` | 1193 | 3 (0) | 0.920 | 59.1 | 43 | 0.934 | 55.9 | 43 | 2.9% | - |
| `molar_refractivity_est` | 1193 | 3 (0) | 0.908 | 16.1 | 12.8 | 0.922 | 15.3 | 12.8 | 0.0% | 0.822 / 23.4 |
| `atom_count_heavy` | 1193 | 3 (0) | 0.949 | 2.86 | 2 | 0.964 | 2.64 | 2 | 9.0% | 0.861 / 4.63 |
| `atom_count_total` | 1193 | 3 (0) | 0.941 | 10.5 | 10 | 0.956 | 10 | 10 | 1.2% | - |
| `atom_count_hydrogen` | 1193 | 3 (0) | 0.823 | 8.44 | 8 | 0.837 | 8.26 | 8 | 2.3% | - |
| `element_count_c` | 1193 | 3 (0) | 0.902 | 3.44 | 3 | 0.916 | 3.29 | 3 | 13.5% | - |
| `element_count_n` | 1193 | 192 (189) | 0.970 | 0.181 | 0 | 0.964 | 0.187 | 0 | 84.8% | - |
| `element_count_o` | 1193 | 151 (148) | 0.958 | 0.632 | 0 | 0.958 | 0.667 | 0 | 74.7% | - |
| `element_count_s` | 1193 | 935 (934) | 0.996 | 0.0134 | 0 | 0.986 | 0.0426 | 0 | 99.0% | - |
| `element_count_f` | 1193 | 1090 (1090) | 1.000 | 0 | 0 | 1.000 | 0 | 0 | 100.0% | - |
| `element_count_cl` | 1193 | 1013 (1012) | 0.996 | 0.00587 | 0 | 0.997 | 0.0278 | 0 | 99.6% | - |
| `heteroatom_count` | 1193 | 3 (0) | 0.932 | 0.949 | 0 | 0.947 | 0.871 | 0 | 56.8% | 0.790 / 1.83 |
| `valence_electron_count` | 1193 | 3 (0) | 0.918 | 18.3 | 14 | 0.932 | 17.1 | 14 | 4.3% | 0.811 / 28.6 |
| `formal_charge_total` | 1193 | 1095 (1095) | 1.000 | 0 | 0 | 1.000 | 0 | 0 | 100.0% | - |
| `ring_count_total` | 1193 | 39 (36) | 0.958 | 0.177 | 0 | 0.968 | 0.16 | 0 | 84.7% | 0.929 / 0.27 |
| `ring_count_aromatic` | 1193 | 215 (218) | 0.931 | 0.139 | 0 | 0.889 | 0.157 | 0 | 87.2% | 0.876 / 0.219 |
| `ring_count_aliphatic` | 1193 | 512 (509) | 0.964 | 0.164 | 0 | 0.918 | 0.251 | 0 | 85.8% | 0.928 / 0.262 |
| `aromatic_atom_count` | 1193 | 215 (218) | 0.946 | 0.718 | 0 | 0.918 | 0.808 | 0 | 76.2% | - |
| `bond_count_aromatic` | 1193 | 223 (218) | 0.878 | 2.24 | 1 | 0.794 | 2.64 | 2 | 34.3% | - |
| `ring_atom_count` | 1193 | 39 (36) | 0.922 | 1.7 | 0 | 0.930 | 1.61 | 0 | 53.1% | - |
| `largest_ring_size` | 1193 | 39 (36) | 0.625 | 0.332 | 0 | 0.537 | 0.292 | 0 | 89.8% | - |
| `bond_count_amide` | 1193 | 848 (859) | 0.954 | 0.111 | 0 | 0.704 | 0.322 | 0 | 92.0% | 0.853 / 0.213 |
| `rotatable_bond_count` | 1193 | 65 (84) | 0.900 | 0.94 | 1 | 0.899 | 0.956 | 1 | 41.0% | 0.774 / 3.42 |
| `hbond_donor_count` | 1193 | 280 (251) | 0.926 | 0.308 | 0 | 0.913 | 0.313 | 0 | 76.7% | 0.810 / 0.634 |
| `hbond_acceptor_count` | 1193 | 11 (7) | 0.846 | 1.07 | 1 | 0.857 | 1.02 | 1 | 43.5% | 0.735 / 1.55 |
| `lipinski_rule_of_5_violations` | 1193 | 943 (990) | 0.621 | 0.192 | 0 | 0.685 | 0.644 | 1 | 82.6% | - |
| `topological_polar_surface_area` | 1193 | 10 (7) | 0.935 | 17.2 | 11.4 | 0.948 | 16.2 | 11.4 | 13.7% | 0.753 / 29.7 |
| `logp_xlogp3_est` | 1193 | 3 (0) | 0.918 | 0.975 | 0.793 | 0.918 | 0.975 | 0.794 | 0.0% | 0.882 / 1.15 |
| `fraction_csp3` | 1193 | 59 (54) | 0.937 | 0.0558 | 0.0427 | 0.936 | 0.0567 | 0.0441 | 14.0% | 0.927 / 0.0597 |
| `kappa1_est` | 1193 | 3 (0) | 0.746 | 8.63 | 7.35 | 0.759 | 8.47 | 7.33 | 0.0% | 0.506 / 3.84 |
| `kappa2_est` | 1193 | 3 (0) | 0.733 | 2.83 | 2.25 | 0.746 | 2.76 | 2.25 | 0.0% | 0.609 / 4.02 |
| `kappa3_est` | 1193 | 3 (0) | 0.512 | 1.86 | 1.21 | 0.523 | 1.82 | 1.21 | 0.0% | 0.418 / 2.85 |
| `balaban_j_est` | 1193 | 3 (0) | 0.018 | 0.493 | 0.403 | 0.011 | 0.491 | 0.402 | 0.1% | - |
| `bertz_ct_est` | 1193 | 3 (0) | 0.821 | 150 | 102 | 0.835 | 142 | 102 | 0.0% | 0.760 / 981 |

Figure 2 descriptors:

| Descriptor | rho, as submitted (98-file) | rho, same input | MAE as submitted | MAE same input | GPT zeros |
|---|---|---|---|---|---|
| MolWt | 0.799 | 0.920 | 93.78 | 59.23 | 3 |
| TPSA | 0.753 | 0.935 | 29.67 | 17.17 | 10 |
| FractionCSP3 | 0.927 | 0.937 | 0.05968 | 0.05579 | 59 |

The as-submitted values reproduce the paper (0.799, 0.753, 0.927; MAE 93.78, 29.67, 0.0597; the caption's 0.928 is a rounding slip, the exact value is 0.9275). Comparing against the molecule GPT actually saw raises molecular weight and TPSA agreement substantially; FractionCSP3 barely changes because counter-ions rarely contain carbon.

**Exact zeros.** `molecular_weight`: 3 GPT zeros (RDKit 0 for 0), rho 0.920 -> 0.934, MAE 59.2 -> 56 without them; `topological_polar_surface_area`: 10 GPT zeros (RDKit 0 for 7), rho 0.935 -> 0.948, MAE 17.2 -> 16.2 without them. The molecular-weight zeros are failed generations: for those 7 compounds every GPT-5.2 feature is exactly 0 (stored as 0 after the retries were exhausted). The remaining TPSA zeros are correct (e.g. quaternary ammonium and polyhalogenated compounds with no polar atoms; RDKit gives 0 too). Zeros move the statistics only slightly.

### 5c. By category (same-input reference, GPT zeros removed)

| Category | Features | Median rho | Range | Median exact-match rate |
|---|---|---|---|---|
| counts | 24 | 0.924 | [0.537, 1.000] | 76.4% |
| fractions | 1 | 0.936 | [0.936, 0.936] | 14.0% |
| polarity/lipophilicity | 2 | 0.933 | [0.918, 0.948] | 6.9% |
| size/mass | 3 | 0.934 | [0.922, 0.934] | 1.3% |
| topological indices | 5 | 0.746 | [0.011, 0.835] | 0.0% |

Best recovered: `element_count_f` (1.00), `formal_charge_total` (1.00), `element_count_cl` (1.00), `element_count_s` (0.99), `ring_count_total` (0.97), `atom_count_heavy` (0.96). Worst: `balaban_j_est` (0.01), `kappa3_est` (0.52), `largest_ring_size` (0.54), `lipinski_rule_of_5_violations` (0.68), `bond_count_amide` (0.70), `kappa2_est` (0.75).

Pattern: GPT-5.2 is reliable for quantities it can read off the SMILES string by counting (element counts, formal charge, ring counts, aromatic atoms), good for additive physicochemical estimates (mass, logP, TPSA, FractionCSP3; rho about 0.9 but with absolute errors far too large to replace a calculator), and poor for graph-theoretic indices that need the full topology (Kappa shape indices, Balaban J; Bertz complexity is intermediate).

### 5d. `exact_mass_est`

`ExactMolWt` is **not** among the 98 RDKit descriptors (the set has `MolWt` and `HeavyAtomMolWt` only), so in the submitted redundancy analysis `exact_mass_est` could only be matched to other descriptors (its best match there is the connectivity index `Chi2v`, see Task 4), and those were computed on the salted structures. Computing `ExactMolWt` on GPT's input SMILES:

- `exact_mass_est` vs RDKit `ExactMolWt`: rho = 0.934, MAE = 55.9 Da, median AE = 43.0 Da (n = 1190; failed generations and unparseable SMILES excluded).
- GPT `molecular_weight` vs RDKit `MolWt`, same compounds: rho = 0.934.
- `exact_mass_est` vs GPT's own `molecular_weight`: rho = 1.000, median |difference| = 0.20 Da (RDKit median |ExactMolWt - MolWt| = 0.25 Da).

So `exact_mass_est` does match `ExactMolWt` once the same molecule and the right descriptor are used; it is essentially GPT's molecular-weight estimate restated with a small monoisotopic adjustment, and it inherits that estimate's error (tens of Da). It looked unalignable only because `ExactMolWt` is absent from the 98-descriptor set and that set describes the salted molecules.

## Task 6. Figures (AAAI two-column)

All figures: no titles, vector PDF plus 300-dpi PNG, font 8 pt (ticks 7 pt), single-column 3.3 in / full-width 7.0 in. Colours blue `#3B6FB6` and vermillion `#D55E00` (pass a colour-vision-deficiency and contrast check) always paired with a marker-shape difference; heatmaps use a single-hue sequential scale, so all figures read in greyscale. Generated by `scripts/t6_figures.py` from the CSVs in `data/`.

| File | Size | Content | Caption notes |
|---|---|---|---|
| `fig1_overlap` | 7.0 x 2.6 in | Left: shared feature-name counts (diagonal = each model's total, grey). Right: pairwise Jaccard, diagonal masked, scale 0-0.15, values to 2 decimals. Order GPT-4o-mini ... GPT-5.2. | Exact name matching, metadata columns removed. |
| `fig1b_overlap_exact_vs_normalized` | 7.0 x 2.6 in | Jaccard, exact (left) vs normalized names (right), same 0-0.15 scale. | Normalization rules in Task 2. |
| `fig2_fidelity` | 7.0 x 2.2 in | GPT-5.2 vs RDKit for MW, TPSA, FractionCSP3; y = x dashed; GPT exact zeros as vermillion crosses with their count. **RDKit computed on the same desalted SMILES GPT received** (Task 5). | rho = 0.92 / 0.93 / 0.94; n zeros = 3 / 10 / 59 (RDKit itself is 0 for 0 / 7 / 54 of these compounds; zero TPSA and FractionCSP3 are often correct). Axes span the full data range. |
| `fig2b_fidelity_as_submitted` | 7.0 x 2.2 in | Same layout against the 98-descriptor file (salted structures), i.e. the submitted comparison. | rho = 0.80 / 0.75 / 0.93; n zeros = 7 / 14 / 63. Use only if the paper keeps the old reference. |
| `fig3_coverage` | 3.3 x 2.8 in | Top-10 GPT-5.2 features by mean gain over the 5 trial models (Task 4); bar = max abs Spearman rho with any of the 98 RDKit descriptors, best match printed at the bar end; dashed line at 0.5. | Correlations over 1,197 compounds against the 98-descriptor set used as the RDKit baseline. |
| `fig4_mic` | 3.3 x 2.6 in | log2 MIC vs G for the trial-0 MolE+GPT-5.2 model; gram-positive blue circles, gram-negative vermillion triangles; horizontal jitter +/-0.12 and transparency so overlapping points show; OLS line, no slope box. | n = 38 observations, 29 compounds; Spearman rho = -0.493 (p = 0.0017) for this model; G uses log2. 3 MICs are not in ug/mL (Task 3). |
| `fig4b_mic_mole_gpt52_vs_mole` | 7.0 x 2.6 in | Same, MolE+GPT-5.2 (left) vs MolE-only (right), shared y axis. | MolE-only trial 0: rho = -0.457. |

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

**Caveats.** GPU `hist` throughout; MolE used CPU `exact`, worth +0.015 on their own
configuration (0.7802 vs 0.7655). Three of the 5 MolE trials selected a subsample = 1.0
configuration, which is deterministic (fit-to-fit SD exactly 0), so the SD in the table is
search variance, not seed variance. For the selected configurations `auc(recall, precision)` and
`average_precision_score` agree to 4e-4.

Single-configuration comparison for reference (`results/mole_method_all_rows/`, FINDINGS 3c):
with MolE's one published configuration applied to every row, MolE + GPT-5.2 scores 0.7331
against MolE's 0.7527, -0.019 with a bootstrap 95% CI of [-0.059, +0.022] - the same small
difference, with the sign flipped by which single configuration is used.

## Draft tables for the revision

`tables.tex`, generated by `scripts/make_tables.py` from the results files, so no number is
typed by hand. It needs `\newcommand{\PM}{\,\pm\,}` (as in the submitted template) and
`booktabs`. Seven tables:

| Label | What it is | Replaces / adds |
|---|---|---|
| `tab:main` | 9 representations, val and test per-strain PR-AUC, mean +/- SD over 5 trials | replaces submitted Table 1 |
| `tab:paired` | paired differences vs MolE and vs RDKit, wins and p | new (answers the single-run objection) |
| `tab:guided` | free-form vs guided GPT features, GPT-only and MolE + GPT | replaces the submitted guided table |
| `tab:protocol` | MolE's original protocol vs ours, both over 5 trials | new (answers the 8.5-point drop) |
| `tab:overlap` | vocabulary Jaccard and model-specific share at 3 strictness levels | replaces exact-match-only overlap |
| `tab:fidelity` | GPT-5.2 vs RDKit on the same desalted input, 12 descriptors | corrects the salted-SMILES reference |
| `tab:mic` | MIC correlation with MolE-only and RDKit-only controls | adds the missing controls |

Notes on choices made in these drafts:

- **One metric per table.** `tab:main`, `tab:paired` and `tab:guided` use the mean per-strain
  PR-AUC in both the validation and test columns. `tab:protocol` uses the pooled PR-AUC in both
  columns, because that is what MolE's code computes. The submitted Table 1 mixed the two
  (section 1), which flattered the GPT rows by 0.003-0.012.
- **Bold.** Best validation and best test value in `tab:main`; significant p-values
  (p < 0.05) in `tab:paired` and `tab:guided`; the best of the three rows in `tab:protocol`.
  Drop the `\textbf` if the venue's style discourages it.
- **The published MolE row stays in `tab:main`** but the caption now says why it is not
  comparable and points at `tab:protocol`, where that protocol is run properly.
- **`tab:paired` reports the paired t-test.** At n = 5 the Wilcoxon signed-rank p cannot go
  below 0.0625, so it can never show significance; both are in `data/paired_tests.csv` and in
  the Task 1 and Task 8 sections.
- **Minus signs** are `$-$`, not hyphens, and p < 0.001 is printed as `$<$0.001`.
- `tab:fidelity` shows 12 of the 35 matched descriptors: the four in Figure 2, the four other
  best-reproduced ones, and the four topological indices where GPT fails. The full set is in
  `data/fidelity_gpt52.csv`.

Not drafted, because they depend on decisions that are yours: the feature-importance table
(Task 4) and any table for the mechanistic-chemistry probe, and the related-work additions
(Qian et al., Zheng et al.) the reviewers asked for.
