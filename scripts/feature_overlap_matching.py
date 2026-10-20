"""Measure cross-model feature-vocabulary divergence under strict and lenient matching.

Figure 1 of the submission counts two feature names as distinct unless they match
exactly, so `logp_est` and `estimated_logP` are treated as different descriptors.
Reviewers wCKY (W4) and 8LiW both asked whether the divergence survives
synonym-aware matching. This reports both bounds so the figure can be published
as a range.

Strict  : exact feature-name matching, as in the submission.
Lenient : case- and punctuation-folded, estimator affixes stripped, and a
          synonym table applied for the common physicochemical descriptors.

Outputs results/feature_overlap/overlap_strict_vs_lenient.csv
    and results/feature_overlap/per_model_uniqueness.csv
"""

from __future__ import annotations

import collections
import glob
import itertools
import os
import re

import pandas as pd

FREEFORM_GLOB = "data/gpt_generation_without_guidance/*.csv"
OUT_DIR = "results/feature_overlap"

# Columns that identify the compound rather than describe it.
META_PREFIXES = ("prestwick_id", "smiles", "rdkit_no_salt", "identifiers",
                 "input", "chemical_name", "name", "unnamed")

# Affixes that mark an estimate rather than a different quantity.
SUFFIX = re.compile(r"(_est|_estimate|_estimated|_pred|_predicted|_approx|_calc"
                    r"|_score|_value|_count|_num|_n)$")
PREFIX = re.compile(r"^(est_|estimated_|predicted_|pred_|approx_|num_|n_|calc_)")
TRAILING = re.compile(r"(est|estimated|predicted|pred|approx)$")

SYNONYMS = {
    # molecular weight
    "mw": "molecularweight", "molwt": "molecularweight",
    "molecularmass": "molecularweight", "exactmass": "molecularweight",
    "monoisotopicmass": "molecularweight", "exactmolwt": "molecularweight",
    # lipophilicity
    "logp": "logp", "clogp": "logp", "alogp": "logp", "mollogp": "logp",
    "octanolwaterpartitioncoefficient": "logp", "lipophilicity": "logp",
    # polar surface area
    "tpsa": "tpsa", "topologicalpolarsurfacearea": "tpsa", "polarsurfacearea": "tpsa",
    # hydrogen bonding
    "hbd": "hbonddonors", "hydrogenbonddonors": "hbonddonors",
    "numhdonors": "hbonddonors", "hbonddonor": "hbonddonors",
    "hba": "hbondacceptors", "hydrogenbondacceptors": "hbondacceptors",
    "numhacceptors": "hbondacceptors", "hbondacceptor": "hbondacceptors",
    # flexibility and rings
    "rotatablebonds": "rotatablebonds", "numrotatablebonds": "rotatablebonds",
    "rotbonds": "rotatablebonds",
    "aromaticrings": "aromaticrings", "numaromaticrings": "aromaticrings",
    "aromaticring": "aromaticrings",
    # composition
    "heavyatoms": "heavyatoms", "numheavyatoms": "heavyatoms",
    "heavyatom": "heavyatoms", "atomheavy": "heavyatoms",
    "heteroatom": "heteroatoms", "numheteroatoms": "heteroatoms",
    "heteroatoms": "heteroatoms",
    # charge and saturation
    "formalcharge": "charge", "netcharge": "charge", "chargeatph74": "charge",
    "charge": "charge",
    "fractioncsp3": "fractioncsp3", "csp3fraction": "fractioncsp3",
    "fractionsp3": "fractioncsp3",
}


def model_name(path: str) -> str:
    base = os.path.basename(path)
    m = re.search(r"(gpt-[\w.\-]+?)_freeform", base)
    return m.group(1) if m else base.replace(".csv", "")


def feature_names(path: str) -> list[str]:
    cols = list(pd.read_csv(path, nrows=1).columns)
    return [c for c in cols
            if not any(str(c).lower().startswith(p) for p in META_PREFIXES)]


def normalize(name: str) -> str:
    s = str(name).split(".")[-1].lower()
    s = SUFFIX.sub("", s)
    s = PREFIX.sub("", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    s = TRAILING.sub("", s)
    return SYNONYMS.get(s, s)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


def main() -> None:
    paths = sorted(glob.glob(FREEFORM_GLOB))
    if not paths:
        print(f"No feature files matched {FREEFORM_GLOB}")
        return

    strict = {model_name(p): set(feature_names(p)) for p in paths}
    lenient = {model_name(p): {normalize(c) for c in feature_names(p)} for p in paths}

    pair_rows = []
    for a, b in itertools.combinations(sorted(strict), 2):
        pair_rows.append({
            "model_a": a, "model_b": b,
            "n_a": len(strict[a]), "n_b": len(strict[b]),
            "jaccard_strict": round(jaccard(strict[a], strict[b]), 4),
            "jaccard_lenient": round(jaccard(lenient[a], lenient[b]), 4),
        })
    pairs = pd.DataFrame(pair_rows)

    count_strict = collections.Counter(f for s in strict.values() for f in s)
    count_lenient = collections.Counter(f for s in lenient.values() for f in s)

    uniq_rows = []
    for m in sorted(strict):
        uniq_rows.append({
            "model": m,
            "n_features_strict": len(strict[m]),
            "n_features_lenient": len(lenient[m]),
            "frac_unique_strict": round(
                sum(1 for f in strict[m] if count_strict[f] == 1) / len(strict[m]), 4),
            "frac_unique_lenient": round(
                sum(1 for f in lenient[m] if count_lenient[f] == 1) / len(lenient[m]), 4),
        })
    uniq = pd.DataFrame(uniq_rows)

    pd.set_option("display.width", 200)
    print(pairs.to_string(index=False))
    print()
    print(uniq.to_string(index=False))
    print()
    print(f"pairwise Jaccard  strict : max {pairs.jaccard_strict.max():.4f}  "
          f"mean {pairs.jaccard_strict.mean():.4f}")
    print(f"                  lenient: max {pairs.jaccard_lenient.max():.4f}  "
          f"mean {pairs.jaccard_lenient.mean():.4f}")
    print(f"model-specific fraction  strict : "
          f"{uniq.frac_unique_strict.min():.3f} - {uniq.frac_unique_strict.max():.3f}")
    print(f"                         lenient: "
          f"{uniq.frac_unique_lenient.min():.3f} - {uniq.frac_unique_lenient.max():.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pairs.to_csv(os.path.join(OUT_DIR, "overlap_strict_vs_lenient.csv"), index=False)
    uniq.to_csv(os.path.join(OUT_DIR, "per_model_uniqueness.csv"), index=False)
    print(f"\nWrote tables to {OUT_DIR}/")


if __name__ == "__main__":
    main()
