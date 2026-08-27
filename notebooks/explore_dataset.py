"""
explore_dataset.py
------------------
Early exploratory data analysis (EDA) script for the Maier et al. microbiome
screening dataset.

Loads the three raw Excel files from the Maier dataset:
  - chem_library_info_SF1.xlsx   — compound metadata (name, SMILES, MW, …)
  - screen_results_info_SF3.xlsx — binary screening labels (compounds × strains)
  - screened_strains_taxids.xlsx — strain taxonomy metadata

Prints summary statistics and plots histograms of raw screening values
for the first 5 strains.

Note: This script was originally written for Google Colab; update `raw_dir`
to point to your local copy of the data before running.

Usage:
    python explore_dataset.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

raw_dir = '/content/drive/MyDrive/HiWi_antimicrobial/data/raw/maier_microbiome'

# Load files
chem_df = pd.read_excel(os.path.join(raw_dir, 'chem_library_info_SF1.xlsx'))
screen_df = pd.read_excel(os.path.join(raw_dir, 'screen_results_info_SF3.xlsx'))
strain_df = pd.read_excel(os.path.join(raw_dir, 'screened_strains_taxids.xlsx'))

print("\n=== Chemical Metadata Summary ===")
print(chem_df.info())
print(chem_df.describe(include='all').T)

print("\n=== Screening Results Matrix Summary ===")
print(screen_df.info())
print(screen_df.describe().T)

# Check for missing values in screening data
print("\nMissing values in screening matrix:", screen_df.isna().sum().sum())

# Distribution of screening values for first 5 strains (columns)
strain_columns = screen_df.columns[4:]  # Assuming first 4 cols are metadata
print("\nPlotting histograms of screening values for first 5 strains...")
for strain in strain_columns[:5]:
    plt.figure(figsize=(6,4))
    plt.hist(screen_df[strain], bins=50, alpha=0.7)
    plt.title(f'Screening Value Distribution: {strain}')
    plt.xlabel('Screening Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

print("\n=== Strain Metadata Summary ===")
print(strain_df.info())
print(strain_df['Species'].value_counts())

