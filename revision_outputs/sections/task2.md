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
