"""Task 5: numerical fidelity of every GPT-5.2 feature that has a direct RDKit counterpart.

Primary reference: RDKit computed on the exact SMILES GPT-5.2 was given (the desalted
`query_smiles` in its `input` column). The 98-descriptor file behind the submitted Figure 2
was computed on PubChem's salted SMILES, so for salts it describes a different molecule
(counter-ion included); its statistics are reported alongside as "as submitted".
"""

from __future__ import annotations

import ast

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from scipy import stats

from common import CHEMDESC, DATA, GPT_FILES, ensure_dirs, write_section

RDLogger.DisableLog("rdApp.*")


def _count(symbol):
    return lambda m: sum(a.GetSymbol() == symbol for a in m.GetAtoms())


def _total_h(m):
    return sum(a.GetTotalNumHs() for a in m.GetAtoms())


def _lipinski_violations(m):
    return sum([Descriptors.MolWt(m) > 500, Descriptors.MolLogP(m) > 5,
                Lipinski.NumHDonors(m) > 5, Lipinski.NumHAcceptors(m) > 10])


def _d(name):
    return getattr(Descriptors, name)


# GPT-5.2 feature -> (RDKit counterpart, RDKit function on the input SMILES, category).
# When the counterpart name is a column of the 98-descriptor file, that (salted) value is
# reported as the "as submitted" comparison.
MAPPING = {
    "molecular_weight": ("MolWt", _d("MolWt"), "size/mass"),
    "exact_mass_est": ("ExactMolWt", _d("ExactMolWt"), "size/mass"),
    "molar_refractivity_est": ("MolMR", _d("MolMR"), "size/mass"),
    "atom_count_heavy": ("HeavyAtomCount", _d("HeavyAtomCount"), "counts"),
    "atom_count_total": ("atoms incl. H", lambda m: m.GetNumAtoms() + _total_h(m), "counts"),
    "atom_count_hydrogen": ("H count", _total_h, "counts"),
    "element_count_c": ("C count", _count("C"), "counts"),
    "element_count_n": ("N count", _count("N"), "counts"),
    "element_count_o": ("O count", _count("O"), "counts"),
    "element_count_s": ("S count", _count("S"), "counts"),
    "element_count_f": ("F count", _count("F"), "counts"),
    "element_count_cl": ("Cl count", _count("Cl"), "counts"),
    "heteroatom_count": ("NumHeteroatoms", _d("NumHeteroatoms"), "counts"),
    "valence_electron_count": ("NumValenceElectrons", _d("NumValenceElectrons"), "counts"),
    "formal_charge_total": ("formal charge", Chem.GetFormalCharge, "counts"),
    "ring_count_total": ("RingCount", _d("RingCount"), "counts"),
    "ring_count_aromatic": ("NumAromaticRings", _d("NumAromaticRings"), "counts"),
    "ring_count_aliphatic": ("NumAliphaticRings", _d("NumAliphaticRings"), "counts"),
    "aromatic_atom_count": ("aromatic atoms", lambda m: sum(a.GetIsAromatic() for a in m.GetAtoms()), "counts"),
    "bond_count_aromatic": ("aromatic bonds", lambda m: sum(b.GetIsAromatic() for b in m.GetBonds()), "counts"),
    "ring_atom_count": ("ring atoms", lambda m: sum(a.IsInRing() for a in m.GetAtoms()), "counts"),
    "largest_ring_size": ("largest ring size",
                          lambda m: max((len(r) for r in m.GetRingInfo().AtomRings()), default=0), "counts"),
    "bond_count_amide": ("NumAmideBonds", rdMolDescriptors.CalcNumAmideBonds, "counts"),
    "rotatable_bond_count": ("NumRotatableBonds", _d("NumRotatableBonds"), "counts"),
    "hbond_donor_count": ("NumHDonors", _d("NumHDonors"), "counts"),
    "hbond_acceptor_count": ("NumHAcceptors", _d("NumHAcceptors"), "counts"),
    "lipinski_rule_of_5_violations": ("Lipinski violations", _lipinski_violations, "counts"),
    "topological_polar_surface_area": ("TPSA", _d("TPSA"), "polarity/lipophilicity"),
    "logp_xlogp3_est": ("MolLogP", _d("MolLogP"), "polarity/lipophilicity"),
    "fraction_csp3": ("FractionCSP3", _d("FractionCSP3"), "fractions"),
    "kappa1_est": ("Kappa1", _d("Kappa1"), "topological indices"),
    "kappa2_est": ("Kappa2", _d("Kappa2"), "topological indices"),
    "kappa3_est": ("Kappa3", _d("Kappa3"), "topological indices"),
    "balaban_j_est": ("BalabanJ", _d("BalabanJ"), "topological indices"),
    "bertz_ct_est": ("BertzCT", _d("BertzCT"), "topological indices"),
}
FIG2 = ["molecular_weight", "topological_polar_surface_area", "fraction_csp3"]


def stats_for(gpt: pd.Series, ref: pd.Series) -> dict:
    ok = gpt.notna() & ref.notna()
    g, r = gpt[ok], ref[ok]
    out = {"n": int(ok.sum()), "n_gpt_zero": int((g == 0).sum()), "n_rdkit_zero": int((r == 0).sum())}
    for tag, mask in (("", np.ones(len(g), bool)), ("_nozero", (g != 0).values)):
        gg, rr = g[mask], r[mask]
        out[f"spearman{tag}"] = stats.spearmanr(gg, rr).statistic if len(gg) > 2 else np.nan
        out[f"mae{tag}"] = float((gg - rr).abs().mean())
        out[f"medae{tag}"] = float((gg - rr).abs().median())
    out["frac_exact"] = float((g.round(4) == r.round(4)).mean())
    return out


def main() -> None:
    ensure_dirs()
    gpt = pd.read_csv(GPT_FILES["gpt-5.2"]).set_index("prestwick_ID")
    rd98 = pd.read_csv(CHEMDESC, sep="\t").set_index("prestwick_ID")
    rd98 = rd98[~rd98.index.duplicated()].reindex(gpt.index)
    # `input` records the request, e.g. "{'query_smiles': 'Nc1...', 'parsed': True}".
    query = gpt["input"].map(lambda s: ast.literal_eval(s).get("query_smiles") if isinstance(s, str) else None)
    mols = query.map(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    n_unparsed = int(mols.isna().sum())

    rows, refs = [], {}
    for feat, (label, fn, cat) in MAPPING.items():
        ref = mols.map(lambda m: fn(m) if m is not None else np.nan).astype(float)
        refs[feat] = ref
        g = gpt[feat].astype(float)
        row = {"gpt_feature": feat, "rdkit_counterpart": label, "category": cat, **stats_for(g, ref)}
        if label in rd98.columns:
            sub = stats_for(g, rd98[label])
            row.update({"in_98_set": True, "spearman_98file": sub["spearman"], "mae_98file": sub["mae"],
                        "medae_98file": sub["medae"]})
        else:
            row["in_98_set"] = False
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{DATA}/fidelity_gpt52.csv", index=False)

    salted = (rd98["MolWt"] - refs["molecular_weight"]).abs() > 1
    pd.DataFrame({
        "prestwick_ID": gpt.index, "query_smiles": query.values, "salt_differs": salted.values,
        **{f"gpt_{f}": gpt[f].values for f in FIG2},
        **{f"rdkit_input_{MAPPING[f][0]}": refs[f].values for f in FIG2},
        **{f"rdkit_98file_{MAPPING[f][0]}": rd98[MAPPING[f][0]].values for f in FIG2},
    }).to_csv(f"{DATA}/fidelity_fig2_points.csv", index=False)

    t = tab.set_index("gpt_feature")
    md = ["## Task 5. Fidelity of GPT-5.2 features with an RDKit counterpart", "",
          "**Reference matters.** The 98-descriptor RDKit file (MolE's chemDesc set, used for the submitted "
          "Figure 2) was computed on PubChem's *salted* isomeric SMILES, while GPT-5.2 received the *desalted* "
          f"SMILES. For {int(salted.sum())} of {int(salted.notna().sum())} compounds the two molecules differ in "
          "MolWt by more than 1 Da (counter-ions such as HCl, maleate, sodium). The primary comparison below "
          "therefore computes every RDKit counterpart on the exact SMILES GPT-5.2 was given "
          f"(`query_smiles` in its `input` column; {n_unparsed} of 1,197 do not parse in RDKit and are excluded). "
          "Columns marked *98-file* repeat the as-submitted comparison for descriptors in that set.", "",
          "### 5a. Mapping", "",
          f"{len(MAPPING)} GPT-5.2 features have a direct RDKit counterpart. Not mapped: features whose definition "
          "differs from any RDKit descriptor (e.g. `ring_count_bridgehead` vs `NumBridgeheadAtoms`, "
          "`ring_count_spiro` vs `NumSpiroAtoms`, `logd7_4_est`, pKa, charge at pH, ADMET scores). "
          "`logp_xlogp3_est` is compared with RDKit's Crippen `MolLogP`, a different logP model than XLogP3.", "",
          "| GPT-5.2 feature | RDKit counterpart | In 98-descriptor set | Category |", "|---|---|---|---|"]
    md += [f"| `{r.gpt_feature}` | {r.rdkit_counterpart} | {'yes' if r.in_98_set else 'no'} | {r.category} |"
           for r in tab.itertuples()]
    md += ["", "### 5b. Agreement with RDKit on GPT's input SMILES", "",
           "`zeros` = GPT values exactly 0 (RDKit exactly 0 in brackets); *(no 0)* columns drop compounds whose GPT "
           "value is exactly 0. `exact` = fraction with GPT equal to RDKit (4 decimals).", "",
           "| GPT-5.2 feature | n | zeros (RDKit) | rho | MAE | median AE | rho (no 0) | MAE (no 0) | median AE (no 0) | exact | rho / MAE vs 98-file |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples():
        f98 = f"{r.spearman_98file:.3f} / {r.mae_98file:.3g}" if r.in_98_set else "-"
        md.append(f"| `{r.gpt_feature}` | {r.n} | {r.n_gpt_zero} ({r.n_rdkit_zero}) | {r.spearman:.3f} | "
                  f"{r.mae:.3g} | {r.medae:.3g} | {r.spearman_nozero:.3f} | {r.mae_nozero:.3g} | "
                  f"{r.medae_nozero:.3g} | {r.frac_exact:.1%} | {f98} |")
    md += ["", "Figure 2 descriptors:", "", "| Descriptor | rho, as submitted (98-file) | rho, same input | "
           "MAE as submitted | MAE same input | GPT zeros |", "|---|---|---|---|---|---|"]
    for f in FIG2:
        r = t.loc[f]
        md.append(f"| {MAPPING[f][0]} | {r.spearman_98file:.3f} | {r.spearman:.3f} | {r.mae_98file:.4g} | "
                  f"{r.mae:.4g} | {r.n_gpt_zero} |")
    md += ["", "The as-submitted values reproduce the paper (0.799, 0.753, 0.927; MAE 93.78, 29.67, 0.0597; the "
           "caption's 0.928 is a rounding slip, the exact value is 0.9275). Comparing against the molecule GPT "
           "actually saw raises molecular weight and TPSA agreement substantially; FractionCSP3 barely changes "
           "because counter-ions rarely contain carbon.", "",
           "**Exact zeros.** " + "; ".join(
               f"`{f}`: {t.loc[f].n_gpt_zero} GPT zeros (RDKit 0 for {t.loc[f].n_rdkit_zero}), rho "
               f"{t.loc[f].spearman:.3f} -> {t.loc[f].spearman_nozero:.3f}, MAE {t.loc[f].mae:.3g} -> "
               f"{t.loc[f].mae_nozero:.3g} without them" for f in FIG2[:2])
           + f". The molecular-weight zeros are failed generations: for those {int((gpt['molecular_weight'] == 0).sum())} "
           "compounds every GPT-5.2 feature is exactly 0 (stored as 0 after the retries were exhausted). The "
           "remaining TPSA zeros are correct (e.g. quaternary ammonium and polyhalogenated compounds with no polar "
           "atoms; RDKit gives 0 too). Zeros move the statistics only slightly.", ""]

    cat = tab.groupby("category").agg(n_features=("gpt_feature", "size"), median_rho=("spearman_nozero", "median"),
                                      min_rho=("spearman_nozero", "min"), max_rho=("spearman_nozero", "max"),
                                      median_exact=("frac_exact", "median")).reset_index()
    md += ["### 5c. By category (same-input reference, GPT zeros removed)", "",
           "| Category | Features | Median rho | Range | Median exact-match rate |", "|---|---|---|---|---|"]
    md += [f"| {r.category} | {r.n_features} | {r.median_rho:.3f} | [{r.min_rho:.3f}, {r.max_rho:.3f}] | "
           f"{r.median_exact:.1%} |" for r in cat.itertuples()]
    best = tab.nlargest(6, "spearman_nozero")
    worst = tab.nsmallest(6, "spearman_nozero")
    md += ["", "Best recovered: " + ", ".join(f"`{r.gpt_feature}` ({r.spearman_nozero:.2f})" for r in best.itertuples())
           + ". Worst: " + ", ".join(f"`{r.gpt_feature}` ({r.spearman_nozero:.2f})" for r in worst.itertuples()) + ".",
           "",
           "Pattern: GPT-5.2 is reliable for quantities it can read off the SMILES string by counting "
           "(element counts, formal charge, ring counts, aromatic atoms), good for additive physicochemical "
           "estimates (mass, logP, TPSA, FractionCSP3; rho about 0.9 but with absolute errors far too large to "
           "replace a calculator), and poor for graph-theoretic indices that need the full topology "
           "(Kappa shape indices, Balaban J; Bertz complexity is intermediate).", ""]

    em = gpt["exact_mass_est"].astype(float)
    gmw = gpt["molecular_weight"].astype(float)
    exact = refs["exact_mass_est"]
    mw_in = refs["molecular_weight"]
    ok = (em != 0) & (gmw != 0) & exact.notna()
    rho_exact = stats.spearmanr(em[ok], exact[ok]).statistic
    rho_mw = stats.spearmanr(gmw[ok], mw_in[ok]).statistic
    rho_self = stats.spearmanr(em[ok], gmw[ok]).statistic
    md += ["### 5d. `exact_mass_est`", "",
           "`ExactMolWt` is **not** among the 98 RDKit descriptors (the set has `MolWt` and `HeavyAtomMolWt` "
           "only), so in the submitted redundancy analysis `exact_mass_est` could only be matched to other "
           "descriptors (its best match there is the connectivity index `Chi2v`, see Task 4), and those were "
           "computed on the salted structures. Computing `ExactMolWt` on GPT's input SMILES:", "",
           f"- `exact_mass_est` vs RDKit `ExactMolWt`: rho = {rho_exact:.3f}, MAE = {(em[ok] - exact[ok]).abs().mean():.1f} Da, "
           f"median AE = {(em[ok] - exact[ok]).abs().median():.1f} Da (n = {int(ok.sum())}; failed generations and "
           "unparseable SMILES excluded).",
           f"- GPT `molecular_weight` vs RDKit `MolWt`, same compounds: rho = {rho_mw:.3f}.",
           f"- `exact_mass_est` vs GPT's own `molecular_weight`: rho = {rho_self:.3f}, median |difference| = "
           f"{(em[ok] - gmw[ok]).abs().median():.2f} Da (RDKit median |ExactMolWt - MolWt| = "
           f"{(exact[ok] - mw_in[ok]).abs().median():.2f} Da).", "",
           "So `exact_mass_est` does match `ExactMolWt` once the same molecule and the right descriptor are used; "
           "it is essentially GPT's molecular-weight estimate restated with a small monoisotopic adjustment, and "
           "it inherits that estimate's error (tens of Da). It looked unalignable only because `ExactMolWt` is "
           "absent from the 98-descriptor set and that set describes the salted molecules."]
    write_section("task5", "\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
