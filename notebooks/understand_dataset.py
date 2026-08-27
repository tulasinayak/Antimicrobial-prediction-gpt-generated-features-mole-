"""
understand_dataset.py
---------------------
Computes summary statistics for the Maier et al. antimicrobial screening
dataset and checks internal consistency of compound/strain identifiers.

Loads:
  - chem_library_info_SF1.xlsx   — compound metadata
  - screen_results_info_SF3.xlsx — binary screening labels (compounds × strains)
  - screened_strains_taxids.xlsx — strain taxonomy metadata

Reports: compound count, strain count, total / active / inactive labels,
missing-value count, activity rate, and ID alignment between tables.

Note: This script was originally written for Google Colab; update `data_dir`
in main() to point to your local copy of the data before running.

Usage:
    python understand_dataset.py
"""

import os
import pandas as pd

def load_data(data_dir):
    chem_file = os.path.join(data_dir, "chem_library_info_SF1.xlsx")
    results_file = os.path.join(data_dir, "screen_results_info_SF3.xlsx")
    strains_file = os.path.join(data_dir, "screened_strains_taxids.xlsx")

    df_chem = pd.read_excel(chem_file)
    df_results = pd.read_excel(results_file)
    df_strains = pd.read_excel(strains_file)

    return df_chem, df_results, df_strains

def explore_data(df_chem, df_results, df_strains):
    print("\n🔬 Compound Metadata:")
    print(df_chem.info())
    print(df_chem.head(), "\n")
    
    print("🧫 Screening Results Matrix:")
    print(df_results.shape)
    print(df_results.iloc[:, :5], "\n")

    print("🧬 Strain Metadata:")
    print(df_strains.info())
    print(df_strains.head(), "\n")

def compute_stats(df_chem, df_results, df_strains):
    print("\n📊 Dataset Stats:\n")
    n_compounds = df_chem.shape[0]
    n_strains = df_results.shape[1] - 2  # Exclude compound ID and chemical_name columns
    total_labels = df_results.iloc[:, 3:].count().sum()  # from 4th col onwards are strains
    total_active = df_results.iloc[:, 3:].sum().sum()
    total_inactive = total_labels - total_active
    missing = df_results.iloc[:, 3:].isna().sum().sum()

    print(f"Total compounds: {n_compounds}")
    print(f"Total strains: {n_strains}")
    print(f"Total screening labels: {total_labels}")
    print(f" - Active (1): {total_active}")
    print(f" - Inactive (0): {total_inactive}")
    print(f"Missing entries: {missing}")
    print(f"Label activity rate: {100 * total_active / total_labels:.2f}%\n")

def check_id_alignment(df_chem, df_results, df_strains):
    compound_ids_chem = set(df_chem['prestwick_ID'])
    compound_ids_results = set(df_results['prestwick_ID'])
    missing_in_chem = compound_ids_results - compound_ids_chem

    print("🧩 ID Consistency Check:")
    if not missing_in_chem:
        print("✅ All compound IDs in results are present in chemical metadata.")
    else:
        print(f"❌ {len(missing_in_chem)} compound IDs in results are missing from metadata.")

    strain_cols = df_results.columns[3:]
    known_strains = set(df_strains['NT_code'])
    missing_strains = set(strain_cols) - known_strains

    if not missing_strains:
        print("✅ All strain codes in screening results are present in strain metadata.")
    else:
        print(f"❌ {len(missing_strains)} strain codes in results are missing from strain metadata.")

def main():
    data_dir = "/content/drive/MyDrive/HiWi_antimicrobial/data/raw/maier_microbiome"
    df_chem, df_results, df_strains = load_data(data_dir)

    explore_data(df_chem, df_results, df_strains)
    compute_stats(df_chem, df_results, df_strains)
    check_id_alignment(df_chem, df_results, df_strains)

if __name__ == "__main__":
    main()

