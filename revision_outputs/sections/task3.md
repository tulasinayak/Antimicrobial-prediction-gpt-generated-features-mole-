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
