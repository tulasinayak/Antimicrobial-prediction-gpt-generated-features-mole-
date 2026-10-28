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
