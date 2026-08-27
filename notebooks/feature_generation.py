"""
feature_generation.py
---------------------
Calls the OpenAI API to generate a flat numeric molecular feature vector
for every compound in the Prestwick library.

For each compound SMILES it sends a structured JSON prompt to the chosen
GPT model and parses the response into a flat CSV row.  Two SMILES
resolution strategies are tried in order:
  1. Heuristic desalting (largest fragment)
  2. PubChem lookup by compound name

Usage:
    export OPENAI_API_KEY="sk-..."
    python feature_generation.py --model_name gpt-4.1
    python D:\Hiwi\GPT+Mole\GPT+Mole\notebooks\feature_generation.py --mode freeform --model_name gpt-5.1


Output:
    data/gpt_feature_generation/gpt_features_5_1_<model>_unified.csv
"""

import os
import time
import json
import argparse
from pathlib import Path

import pandas as pd
import requests
from openai import OpenAI
from tqdm import tqdm

# ── Paths (relative to this script — works on any machine) ───────────────────
_HERE       = Path(__file__).resolve().parent
DATA_DIR    = _HERE.parent / "data"
INPUT_FILE  = DATA_DIR / "prestwick_library_screened.tsv.gz"
OUTPUT_FILE = DATA_DIR / "gpt_generation_without_guidance" / "mechchem_gpt"

# ── Config ────────────────────────────────────────────────────────────────────
SMILES_COLUMN = "rdkit_no_salt"
ID_COLUMN     = "prestwick_ID"
BATCH_DELAY   = 0.3
MAX_RETRIES   = 3
DEFAULT_MODEL = "gpt-4.1-mini"

MODEL_TOKEN_LIMITS = {
    "gpt-5.2":      8000,
    "gpt-5.1":      7000,
    "gpt-4o":       5000,
    "gpt-4o-mini":  5000,
    "gpt-4.1":      6000,
    "gpt-4.1-mini": 5000,
}

# ── API key — set via environment variable, never hardcode ───────────────────
# export OPENAI_API_KEY="sk-..."   (Linux/Mac)
# $env:OPENAI_API_KEY="sk-..."     (Windows PowerShell)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY is not set.\n"
        "  Linux/Mac : export OPENAI_API_KEY='sk-...'\n"
        "  Windows   : $env:OPENAI_API_KEY='sk-...'"
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ── CLI args (parse_known_args avoids errors when run inside Jupyter) ─────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate GPT molecular features for the Prestwick library."
    )
    parser.add_argument(
        "--model_name", type=str, default=DEFAULT_MODEL,
        help="OpenAI model to use (e.g. gpt-4.1, gpt-4.1-mini, gpt-5.1)"
    )
    parser.add_argument(
        "--mode", type=str, default="structured", choices=["structured", "freeform"],
        help="structured: fixed schema; freeform: GPT chooses features for antimicrobial prediction"
    )
    parser.add_argument(
        "--input_file", type=str, default=None,
        help="Path to input TSV/CSV with compounds (default: Prestwick library). "
             "Must contain --id_column and rdkit_no_salt columns."
    )
    parser.add_argument(
        "--id_column", type=str, default=None,
        help="Name of the compound ID column (default: prestwick_ID)."
    )
    parser.add_argument(
        "--lock_schema_from", type=str, default=None,
        help="Path to an existing freeform CSV whose feature columns define the locked schema. "
             "Skips the discovery phase — all compounds use FREEFORM_LOCKED_PROMPT from the start."
    )
    args, _ = parser.parse_known_args()
    return args


PROMPT_TEMPLATE = """You are a senior cheminformatics and molecular descriptor expert.

Your task is to generate a COMPLETE molecular feature vector from a given SMILES string.

The output must EXACTLY match the schema below and include ALL fields.
Do NOT omit any field.
Do NOT include any explanation text.
Output ONLY a single compact JSON object (no newlines, no comments).

IMPORTANT:
- Base all reasoning strictly on the SMILES string.
- Provide chemically consistent estimates using your knowledge of molecular structure and physicochemistry.
- Numeric values must be realistic and internally coherent.
- Boolean features must be 0 or 1.
- Counts must be integers.
- If a feature is not present, return 0 (not null).
- Ensure molecular formula, atom counts, and molecular weight are consistent.

Input SMILES: "{smiles}"

JSON SCHEMA (FLAT STRUCTURE — ALL INDIVIDUAL COLUMNS):

{
  "atom_count_total": 0,
  "heavy_atom_count": 0,
  "hetero_atom_count": 0,
  "formal_charge": 0,
  "ring_count": 0,
  "aromatic_ring_count": 0,
  "aliphatic_ring_count": 0,
  "saturated_ring_count": 0,
  "heterocycle_count": 0,
  "fused_system_count": 0,
  "spiro_center_count": 0,
  "NHOHCount": 0,
  "NOCount": 0,
  "MolWt": 0.0,
  "ExactMolWt": 0.0,
  "HeavyAtomMolWt": 0.0,
  "NumValenceElectrons": 0,
  "NumRadicalElectrons": 0,
  "NumHDonors": 0,
  "NumHAcceptors": 0,
  "NumRotatableBonds": 0,
  "TPSA": 0.0,
  "MolLogP": 0.0,
  "MolMR": 0.0,
  "FractionCSP3": 0.0,
  "NumAliphaticRings": 0,
  "NumAromaticRings": 0,
  "NumSaturatedRings": 0,
  "NumHeterocycles": 0,
  "NumAromaticHeterocycles": 0,
  "NumSaturatedHeterocycles": 0,
  "NumAliphaticCarbocycles": 0,
  "NumAromaticCarbocycles": 0,
  "NumSaturatedCarbocycles": 0,
  "MaxAbsPartialCharge": 0.0,
  "MaxPartialCharge": 0.0,
  "MinAbsPartialCharge": 0.0,
  "MinPartialCharge": 0.0,
  "LabuteASA": 0.0,
  "pyLabuteASA": 0.0,
  "NumAmideBonds": 0,
  "NumBridgeheadAtoms": 0,
  "NumHBA": 0,
  "NumHBD": 0,
  "NumHeteroatoms": 0,
  "NumLipinskiHBA": 0,
  "NumLipinskiHBD": 0,
  "NumRings": 0,
  "NumSpiroAtoms": 0,
  "PBF": 0.0,
  "PMI1": 0.0,
  "PMI2": 0.0,
  "PMI3": 0.0,
  "Asphericity": 0.0,
  "Eccentricity": 0.0,
  "InertialShapeFactor": 0.0,
  "RadiusOfGyration": 0.0,
  "SpherocityIndex": 0.0,
  "Chi0": 0.0,
  "Chi1": 0.0,
  "Chi2n": 0.0,
  "Chi2v": 0.0,
  "Chi3n": 0.0,
  "Chi3v": 0.0,
  "Chi4n": 0.0,
  "Chi4v": 0.0,
  "Kappa1": 0.0,
  "Kappa2": 0.0,
  "Kappa3": 0.0,
  "HallKierAlpha": 0.0,
  "BalabanJ": 0.0,
  "BertzCT": 0.0,
  "Ipc": 0.0,
  "PEOE_VSA1": 0.0,
  "PEOE_VSA2": 0.0,
  "PEOE_VSA3": 0.0,
  "PEOE_VSA4": 0.0,
  "PEOE_VSA5": 0.0,
  "PEOE_VSA6": 0.0,
  "PEOE_VSA7": 0.0,
  "PEOE_VSA8": 0.0,
  "PEOE_VSA9": 0.0,
  "PEOE_VSA10": 0.0,
  "PEOE_VSA11": 0.0,
  "PEOE_VSA12": 0.0,
  "PEOE_VSA13": 0.0,
  "PEOE_VSA14": 0.0,
  "SMR_VSA1": 0.0,
  "SMR_VSA2": 0.0,
  "SMR_VSA3": 0.0,
  "SMR_VSA4": 0.0,
  "SMR_VSA5": 0.0,
  "SMR_VSA6": 0.0,
  "SMR_VSA7": 0.0,
  "SMR_VSA8": 0.0,
  "SMR_VSA9": 0.0,
  "SMR_VSA10": 0.0,
  "SlogP_VSA1": 0.0,
  "SlogP_VSA2": 0.0,
  "SlogP_VSA3": 0.0,
  "SlogP_VSA4": 0.0,
  "SlogP_VSA5": 0.0,
  "SlogP_VSA6": 0.0,
  "SlogP_VSA7": 0.0,
  "SlogP_VSA8": 0.0,
  "SlogP_VSA9": 0.0,
  "SlogP_VSA10": 0.0,
  "SlogP_VSA11": 0.0,
  "SlogP_VSA12": 0.0,
  "fr_Al_OH": 0,
  "fr_Ar_OH": 0,
  "fr_Ar_N": 0,
  "fr_Ar_NH": 0,
  "fr_NH0": 0,
  "fr_NH1": 0,
  "fr_NH2": 0,
  "fr_amide": 0,
  "fr_ester": 0,
  "fr_ketone": 0,
  "fr_ether": 0,
  "fr_phenol": 0,
  "fr_nitrile": 0,
  "fr_halogen": 0,
  "fr_sulfonamide": 0,
  "fr_guanidine": 0,
  "fr_quatN": 0,
  "fr_imidazole": 0,
  "fr_thiophene": 0,
  "fr_pyridine": 0,
  "fr_benzene": 0,
  "net_charge_ph7_est": 0,
  "cationic_centers": 0,
  "anionic_centers": 0,
  "amphiphilicity_index": 0.0,
  "polar_surface_fraction": 0.0,
  "hydrophobic_surface_fraction": 0.0,
  "membrane_interaction_potential": 0.0,
  "peptidomimetic_likeness": 0.0,
  "aromaticity_fraction": 0.0,
  "rigidity_index": 0.0,
  "lipinski_violations": 0,
  "synthetic_accessibility_score_est": 0.0,
  "pka_acidic_est": 0.0,
  "pka_basic_est": 0.0
}
Return ONLY the JSON object.
""".strip()


# ── Freeform prompt — GPT decides which features matter ───────────────────────
FREEFORM_DISCOVERY_PROMPT = """You are a senior cheminformatics and molecular descriptor expert specializing in antimicrobial drug discovery and bacterial resistance.

Given the SMILES string below, generate the most comprehensive molecular feature vector you can — drawing on everything you know about the molecule's chemistry, its potential to inhibit bacteria, and factors that drive or limit antimicrobial activity. Think broadly and deeply. Do not stop until you have exhausted all aspects of the molecule and its features related to antimicrobial activity.

Generate as many features as possible and think what is important for the given task for antimicrobial prediciton. 

Rules:
- Output ONLY a single compact JSON object (no newlines, no markdown, no comments).
- Keys must be snake_case, descriptive, and unambiguous.
- All values must be numeric (int or float). Never null — use 0 if a feature is absent or inapplicable.
- Provide chemically reasonable estimates for anything not directly countable.
- Base ALL values strictly on the SMILES structure.
- DO NOT truncate — output every feature you generate.

Input SMILES: "{smiles}"

Return ONLY the JSON object.""".strip()

# Used for all compounds after the first — locks in the GPT-discovered key names
# while still asking GPT to reason from scratch for each SMILES.
FREEFORM_LOCKED_PROMPT = """You are a senior cheminformatics and molecular descriptor expert specializing in antimicrobial drug discovery.

Given the SMILES string below, compute values for the following molecular features using your expert knowledge of antimicrobial activity.

Output ONLY a single compact JSON object using EXACTLY these keys (no additions, no omissions):
{schema_keys}

Rules:
- All values must be numeric (int or float). Never null — use 0 if a feature is absent or inapplicable.
- Base every value strictly on the SMILES structure.
- No explanations, no markdown, no comments.

Input SMILES: "{smiles}"

Return ONLY the JSON object.""".strip()


def clean_smiles(smiles: str) -> str:
    return (smiles or "").strip().replace("\n", "").replace("\r", "")


def largest_fragment_no_rdkit(smiles: str) -> str:
    """Heuristic desalting: keep the longest fragment."""
    if not smiles:
        return None
    frags = [f.strip() for f in smiles.split(".") if f.strip()]
    return max(frags, key=len) if frags else None


PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"

def pubchem_smiles_from_name(name: str):
    if not name:
        return None
    try:
        r = requests.get(
            PUBCHEM_URL.format(name=requests.utils.quote(name)),
            timeout=15
        )
        if r.status_code != 200:
            return None
        props = r.json().get("PropertyTable", {}).get("Properties", [])
        return props[0].get("CanonicalSMILES") if props else None
    except Exception:
        return None


def _build_prompt(smiles: str, mode: str, locked_schema: list) -> str:
    if mode == "freeform":
        if locked_schema:
            keys_json = json.dumps(locked_schema)
            return FREEFORM_LOCKED_PROMPT.replace("{smiles}", smiles).replace("{schema_keys}", keys_json)
        return FREEFORM_DISCOVERY_PROMPT.replace("{smiles}", smiles)
    return PROMPT_TEMPLATE.replace("{smiles}", smiles)


def gpt_extract_features(smiles: str, prestwick_id: str, model_name: str,
                          max_tokens: int, use_max_completion_tokens: bool,
                          mode: str = "structured", locked_schema: list = None) -> dict:
    locked_schema = locked_schema or []
    prompt = _build_prompt(smiles, mode, locked_schema)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            kwargs = dict(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": "You output ONLY strict JSON. No prose, no explanations, no markdown."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            if use_max_completion_tokens:
                kwargs["max_completion_tokens"] = max_tokens
            else:
                kwargs["max_tokens"] = max_tokens

            resp = client.chat.completions.create(**kwargs)
            data = json.loads(resp.choices[0].message.content)
            data.setdefault("input", {})
            data["input"]["query_smiles"] = smiles
            data["input"]["parsed"] = True
            return data

        except Exception as e:
            print(f"[WARN] GPT failed on SMILES ({prestwick_id}): {e}")
            time.sleep(0.8 * attempt)

    return {"error": f"Failed to parse SMILES: {smiles}", ID_COLUMN: prestwick_id}


def main():
    args = parse_args()
    model_name = args.model_name
    mode       = args.mode
    max_tokens = MODEL_TOKEN_LIMITS.get(model_name, 3000)
    use_max_completion_tokens = model_name.startswith("gpt-5")

    # ── Override input/ID from CLI if provided ────────────────────────────────
    input_path = Path(args.input_file) if args.input_file else INPUT_FILE
    id_col     = args.id_column        if args.id_column     else ID_COLUMN

    print(f"[INFO] Model       : {model_name}")
    print(f"[INFO] Mode        : {mode}")
    print(f"[INFO] Input file  : {input_path}")
    print(f"[INFO] ID column   : {id_col}")
    print(f"[INFO] Output base : {OUTPUT_FILE}")

    sep = "\t" if str(input_path).endswith((".tsv", ".tsv.gz", ".gz")) else ","
    df  = pd.read_csv(input_path, sep=sep)

    if SMILES_COLUMN not in df.columns:
        raise ValueError(f"Column '{SMILES_COLUMN}' not found in dataset.")
    if id_col not in df.columns:
        raise ValueError(f"Column '{id_col}' not found in dataset.")

    NAME_COLUMNS = ["compound_name", "chemical name", "drug_name", "name", "ProductName", "Synonyms"]

    results, errors = [], []

    # ── Pre-load locked schema if requested ───────────────────────────────────
    META_COLS = {ID_COLUMN, "prestwick_ID", "ProductName", "smiles_resolution", "input"}
    if args.lock_schema_from:
        schema_df = pd.read_csv(args.lock_schema_from)
        locked_schema = [c for c in schema_df.columns if c not in META_COLS]
        print(f"[INFO] Schema locked from {args.lock_schema_from}: {len(locked_schema)} features")
    else:
        locked_schema = []  # populated after first successful freeform response

    print(f"[INFO] Generating features for {len(df)} molecules...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        prestwick_id     = str(row[id_col])
        raw_smiles       = clean_smiles(str(row[SMILES_COLUMN]))
        resolution_method = "fast"

        # 1) Heuristic desalting (largest fragment)
        cleaned_smiles = largest_fragment_no_rdkit(raw_smiles)

        # 2) PubChem fallback by compound name
        if not cleaned_smiles:
            for col in NAME_COLUMNS:
                if col in df.columns:
                    pubchem_smiles = pubchem_smiles_from_name(str(row[col]))
                    cleaned_smiles = largest_fragment_no_rdkit(pubchem_smiles)
                    if cleaned_smiles:
                        resolution_method = "pubchem"
                        break

        if not cleaned_smiles:
            errors.append({id_col: prestwick_id,
                            "error": "Could not resolve valid SMILES",
                            "raw_smiles": raw_smiles})
            continue

        features = gpt_extract_features(
            cleaned_smiles, prestwick_id, model_name,
            max_tokens, use_max_completion_tokens,
            mode=mode, locked_schema=locked_schema,
        )

        if features and "error" not in features:
            features[id_col] = prestwick_id
            features["smiles_resolution"] = resolution_method
            results.append(features)

            # Lock in the schema from the first successful freeform response
            if mode == "freeform" and not locked_schema:
                locked_schema = [k for k in features
                                 if k not in (id_col, "smiles_resolution", "input")]
                print(f"[INFO] Freeform schema locked: {len(locked_schema)} features")
                print(f"[INFO] Features: {locked_schema}")
        else:
            errors.append({id_col: prestwick_id,
                            "error": features.get("error", "Unknown"),
                            "raw_smiles": raw_smiles})

        time.sleep(BATCH_DELAY)

    suffix = "freeform" if mode == "freeform" else "unified"
    if args.input_file:
        # Save alongside the input file with a descriptive name
        input_stem = Path(args.input_file).stem
        output_file = Path(args.input_file).parent / f"gpt_features_{input_stem}_{model_name}_{suffix}.csv"
    else:
        output_file = OUTPUT_FILE.with_name(
            OUTPUT_FILE.stem + f"_{model_name}_{suffix}.csv"
        )
    print(f"[INFO] Saving {len(results)} results to {output_file}")
    pd.DataFrame(results).to_csv(output_file, index=False)

    if errors:
        error_file = output_file.with_name(output_file.stem + "_errors.csv")
        print(f"[INFO] Saving {len(errors)} errors to {error_file}")
        pd.DataFrame(errors).to_csv(error_file, index=False)

    print("[INFO] Feature generation complete.")


if __name__ == "__main__":
    main()
