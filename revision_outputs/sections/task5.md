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
