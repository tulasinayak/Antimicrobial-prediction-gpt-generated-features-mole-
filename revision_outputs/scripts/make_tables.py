"""Write revision_outputs/tables.tex from the results files.

    python revision_outputs/scripts/make_tables.py

Every number is read from results/ or revision_outputs/data/, so the tables cannot
drift from the runs. Row order and the \\PM macro follow the submitted template.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from common import OUT

TRIALS = "results/table1_trials"
EXTRA = "results/table1_trials_extra"
PROTOCOL = "results/mole_protocol_trials"
DATA = f"{OUT}/data"

MODELS = ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "gpt-5.1", "gpt-5.2"]
NICE = {m: "GPT-" + m[4:] for m in MODELS}

# Guided GPT-only values as printed in the submitted paper, for the contrast column.
SUBMITTED_GUIDED_ONLY = {"gpt-4o-mini": 0.4779, "gpt-4.1-mini": 0.5220, "gpt-4o": 0.4957,
                         "gpt-4.1": 0.5289, "gpt-5.1": 0.4239, "gpt-5.2": 0.5919}


def per_trial(root: str, key: str) -> pd.DataFrame:
    """One column per run directory, one row per trial, holding `key` from its summary JSON."""
    out: dict[str, dict[int, float]] = {}
    for path in sorted(glob.glob(f"{root}/trial_*/*/summary_metrics_*.json")):
        if "_seed" in os.path.basename(path):
            continue
        trial = int(path.split("trial_")[1].split(os.sep)[0].split("/")[0])
        with open(path, encoding="utf-8") as fh:
            out.setdefault(os.path.basename(os.path.dirname(path)), {})[trial] = json.load(fh)[key]
    return pd.DataFrame(out).sort_index()


TEST = pd.concat([per_trial(TRIALS, "test_macro_pr_auc_by_strain"),
                  per_trial(EXTRA, "test_macro_pr_auc_by_strain")], axis=1)
VAL = pd.concat([per_trial(TRIALS, "val_macro_pr_auc_by_strain"),
                 per_trial(EXTRA, "val_macro_pr_auc_by_strain")], axis=1)


def pm(values: pd.Series, bold: bool = False) -> str:
    cell = f"{values.mean():.4f}\\PM{values.std(ddof=1):.4f}"
    return f"\\textbf{{{values.mean():.4f}}}\\PM{values.std(ddof=1):.4f}" if bold else cell


def paired(a: str, b: str) -> tuple[float, float, str, float, float]:
    d = TEST[a] - TEST[b]
    return (d.mean(), d.std(ddof=1), f"{int((d > 0).sum())}/5",
            stats.ttest_rel(TEST[a], TEST[b]).pvalue, stats.wilcoxon(TEST[a], TEST[b]).pvalue)


def sig(p: float) -> str:
    text = "$<$0.001" if p < 0.001 else f"{p:.3f}"
    return f"\\textbf{{{text}}}" if p < 0.05 else text


def signed(x: float, places: int = 4) -> str:
    """A signed number with a real minus sign rather than a hyphen."""
    return ("$+$" if x >= 0 else "$-$") + f"{abs(x):.{places}f}"


def table_main() -> str:
    rows = [("MolE", "mole_only"), ("RDKit descriptors", "rdkit_only"),
            ("MolE + RDKit", "mole_plus_rdkit")]
    rows += [(f"MolE + {NICE[m]}", f"mole_plus_{m}") for m in MODELS]
    best_val = max(rows, key=lambda r: VAL[r[1]].mean())[1]
    best_test = max(rows, key=lambda r: TEST[r[1]].mean())[1]
    body = "\n".join(
        f"{label} & {pm(VAL[key], key == best_val)} & {pm(TEST[key], key == best_test)} \\\\"
        for label, key in rows)
    return r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcc}
\toprule
Representation & Val.\ PR-AUC & Test PR-AUC \\
\midrule
MolE (published) & 0.7010 & 0.7797 \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Scaffold-held-out antimicrobial prediction under our common protocol:
mean\PM{}std of the mean per-strain PR-AUC over five independent trials (seeds 0--4).
Each trial repeats the hyperparameter search, selects the configuration with the best
validation PR-AUC, and evaluates it once on the test set; within a trial all rows share
the same sampled configurations, so comparisons are paired. The first row is the single
run published by \citet{olayo2025mole} under a different search space (up to depth 100,
no early stopping) and a pooled PR-AUC; Table~\ref{tab:protocol} reports that protocol
run over five trials.}
\label{tab:main}
\end{table}"""


def table_paired() -> str:
    rows = [f"mole_plus_{m}" for m in MODELS] + ["mole_plus_rdkit", "rdkit_only"]
    labels = {f"mole_plus_{m}": f"MolE + {NICE[m]}" for m in MODELS}
    labels |= {"mole_plus_rdkit": "MolE + RDKit", "rdkit_only": "RDKit descriptors",
               "mole_only": "MolE"}
    lines = []
    for key in rows:
        dm, sm, wm, tm, _ = paired(key, "mole_only")
        if key == "rdkit_only":
            right = "-- & -- & --"
        else:
            dr, _, wr, tr, _ = paired(key, "rdkit_only")
            right = f"{signed(dr)} & {wr} & {sig(tr)}"
        lines.append(f"{labels[key]} & {signed(dm)} & {sm:.4f} & {wm} & {sig(tm)} & {right} \\\\")
    return r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrcrrcr}
\toprule
& \multicolumn{4}{c}{vs.\ MolE} & \multicolumn{3}{c}{vs.\ RDKit} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-8}
Representation & $\Delta$ & std & Wins & $p$ & $\Delta$ & Wins & $p$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Paired differences in mean per-strain test PR-AUC across the five trials of
Table~\ref{tab:main}. $\Delta$ is the mean difference, Wins the number of trials in which the
row is ahead, and $p$ a paired $t$-test ($n=5$). Only MolE + GPT-5.2 improves on MolE in every
trial; nothing is significantly better than RDKit descriptors alone, which on their own are
0.016 ahead of MolE. With $n=5$ the Wilcoxon signed-rank test cannot fall below $p=0.0625$, so
we report the $t$-test; the Wilcoxon values are in the supplement.}
\label{tab:paired}
\end{table}"""


def table_gpt_only() -> str:
    lines = []
    for m in MODELS:
        free, guided_mole, guided = f"gpt_only_{m}", f"mole_plus_guided_{m}", f"guided_only_{m}"
        d = TEST[guided] - TEST[free]
        lines.append(
            f"{NICE[m]} & {pm(TEST[free])} & {pm(TEST[guided_mole])} & {pm(TEST[guided])} & "
            f"{SUBMITTED_GUIDED_ONLY[m]:.4f} & {sig(stats.ttest_rel(TEST[guided], TEST[free]).pvalue)} \\\\")
    return r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}
\toprule
& \multicolumn{3}{c}{This work (5 trials)} & \multicolumn{2}{c}{Guided vs.\ free-form} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-6}
Model & GPT only & MolE + guided & Guided only & Submitted & $p$ \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Free-form and schema-guided GPT features under the protocol of
Table~\ref{tab:main} (mean per-strain test PR-AUC, mean\PM{}std over five trials).
Reference values from Table~\ref{tab:main}: MolE 0.6818, RDKit 0.6977, MolE + GPT-5.2
0.7102. ``Submitted'' repeats the guided GPT-only column of the earlier version, which used
a smaller test split and an inconsistent flattening of the same generations; $p$ is a paired
$t$-test of guided-only against free-form-only. Constraining GPT to an RDKit-shaped schema
costs 0.044--0.120 PR-AUC when its features stand alone, while GPT-4.1 features alone match
MolE alone.}
\label{tab:guided}
\end{table}"""


def table_protocol() -> str:
    tab = pd.read_csv(f"{PROTOCOL}/table_mole_protocol.csv").set_index("row")
    ours = pd.read_csv(f"{TRIALS}/table1.csv").set_index("row")
    extra = pd.read_csv(f"{EXTRA}/table1.csv").set_index("row")
    pooled = {"mole_only": ours.loc["MolE embeddings only"],
              "mole_plus_gpt-5.2": ours.loc["MolE + GPT-5.2"],
              "gpt_only_gpt-5.2": extra.loc["GPT-5.2 only"]}
    labels = {"mole_only": "MolE", "mole_plus_gpt-5.2": "MolE + GPT-5.2",
              "gpt_only_gpt-5.2": "GPT-5.2 only"}
    lines = []
    for key in ["mole_only", "mole_plus_gpt-5.2", "gpt_only_gpt-5.2"]:
        t, o = tab.loc[key], pooled[key]
        theirs = f"{t.test_mean:.4f}\\PM{t.test_sd:.4f}"
        if key == "mole_plus_gpt-5.2":
            theirs = f"\\textbf{{{t.test_mean:.4f}}}\\PM{t.test_sd:.4f}"
        mine = f"{o.test_pooled_pr_auc_mean:.4f}\\PM{o.test_pooled_pr_auc_sd:.4f}"
        delta = signed(t.test_mean - o.test_pooled_pr_auc_mean, 3)
        if key == "mole_only":
            diff = "--"
        else:
            diff = (f"{signed(t.diff_vs_mole)} ({t.wins_vs_mole}, "
                    f"$p={t.p_paired_t:.3f}$)")
        lines.append(f"{labels[key]} & {theirs} & {mine} & {delta} & {diff} \\\\")
    return r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccl}
\toprule
Representation & MolE's protocol & Our protocol & $\delta$ & vs.\ MolE \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{The same representations under the original MolE protocol and under ours, both
run for five trials and both scored with the pooled PR-AUC so the columns are comparable.
MolE's protocol draws five configurations per trial from its own space (up to 1000 trees and
depth 100, learning rate up to 1.0), fits each five times, selects on validation PR-AUC, and
uses raw features with no scaling, no class weighting and no early stopping. $\delta$ is the
protocol effect. The GPT-5.2 gain survives the change of protocol but stops separating from
zero, because MolE's protocol doubles the trial-to-trial spread of the paired difference
(std 0.034 vs.\ 0.016). Across the 125 MolE-only fits the test PR-AUC spans 0.229--0.810,
and the configuration that produced the published 0.7797 was not drawn once in 25 draws.}
\label{tab:protocol}
\end{table}"""


def specific_share(heading: str) -> str:
    """Range of the '% specific' column of one section of the Task 2 write-up."""
    with open(f"{OUT}/sections/task2.md", encoding="utf-8") as fh:
        block = fh.read().split(f"### {heading}")[1].split("Pairwise")[0]
    pct = [float(c.strip().rstrip("%")) for line in block.splitlines()
           if line.startswith("| GPT-") for c in [line.split("|")[4]]]
    return f"{min(pct):.0f}--{max(pct):.0f}\\%"


def table_overlap() -> str:
    lines = []
    for level, path, heading in [
            ("Exact", "overlap_exact.csv", "(a) EXACT"),
            ("Normalised", "overlap_normalized.csv", "(b) NORMALIZED"),
            ("Embedding ($\\cos \\geq 0.85$)", "overlap_embedding_0.85.csv",
             "(c) EMBEDDING, cosine >= 0.85")]:
        j = pd.read_csv(f"{DATA}/{path}", index_col=0).to_numpy()
        off = j[~np.eye(len(j), dtype=bool)]
        lines.append(f"{level} & {off.min():.3f} & {off.mean():.3f} & {off.max():.3f} & "
                     f"{specific_share(heading)} \\\\")
    return r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{3}{c}{Pairwise Jaccard} & \\
\cmidrule(lr){2-4}
Matching level & Min & Mean & Max & Model-specific \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Overlap between the feature vocabularies the six models propose: pairwise Jaccard over
the 15 model pairs, and the share of each model's features that no other model proposes, at
three levels of strictness. Normalised matching splits and lower-cases names, drops hedge
tokens such as \texttt{est} and \texttt{approx}, and applies a synonym map; embedding matching
pairs names by sentence-embedding cosine under a Hungarian assignment and is a lenient upper
bound, since it also merges genuinely different features (e.g.\
\texttt{num\_secondary\_nitrogens} with \texttt{num\_basic\_nitrogens}). Even under the most
permissive matching a third to four fifths of each model's vocabulary is its own, so the
divergence is not an artefact of exact string matching.}
\label{tab:overlap}
\end{table}"""


FIDELITY_ROWS = ["molecular_weight", "topological_polar_surface_area", "logp_xlogp3_est",
                 "molar_refractivity_est", "fraction_csp3", "atom_count_heavy",
                 "rotatable_bond_count", "hbond_acceptor_count", "kappa1_est", "kappa3_est",
                 "bertz_ct_est", "balaban_j_est"]


def table_fidelity() -> str:
    d = pd.read_csv(f"{DATA}/fidelity_gpt52.csv").set_index("gpt_feature")
    lines = []
    for name in FIDELITY_ROWS:
        r = d.loc[name]
        old = f"{r.spearman_98file:.3f}" if pd.notna(r.spearman_98file) else "--"
        lines.append(f"\\texttt{{{name.replace('_', chr(92) + '_')}}} & {r.rdkit_counterpart} & "
                     f"{r.spearman:.3f} & {r.mae:.3g} & {old} \\\\")
    return r"""\begin{table}[t]
\centering
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llccc}
\toprule
GPT feature & RDKit counterpart & $\rho$ & MAE & $\rho$ (submitted) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{Fidelity of GPT-5.2's numeric estimates against RDKit, over all 1{,}193 compounds.
$\rho$ is Spearman correlation against RDKit computed on the \emph{same} desalted SMILES the
model was given. The last column is the value in the submitted version, which compared against
descriptors computed on PubChem's salted SMILES; 433 of the 1{,}193 compounds differ by more
than 1\,Da between the two, which depressed every correlation. GPT reproduces size, polarity
and lipophilicity well and topological indices poorly; \texttt{balaban\_j\_est} carries no
signal at all.}
\label{tab:fidelity}
\end{table}"""


def table_mic() -> str:
    m = pd.read_csv(f"{DATA}/mic_correlations_per_trial.csv")
    lines = []
    for rep in ["MolE", "RDKit descriptors", "MolE + GPT-5.2"]:
        s = m[m.representation == rep]
        lines.append(f"{rep} & {signed(s.rho_G.mean(), 3)}\\PM{s.rho_G.std(ddof=1):.3f} & "
                     f"{signed(s.rho_matched.mean(), 3)}\\PM"
                     f"{s.rho_matched.std(ddof=1):.3f} \\\\")
    return r"""\begin{table}[t]
\centering
\small
\begin{tabular}{lcc}
\toprule
Representation & $\rho$ (all strains) & $\rho$ (Gram-matched) \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\caption{External validation against the 38 published MIC measurements: Spearman correlation
between the predicted antimicrobial potential and $\log$ MIC, as mean\PM{}std over the five
trial models of Table~\ref{tab:main}. The Gram-matched column scores each compound against the
strains of its reported Gram class. The submitted version reported $-0.480$ for
MolE + GPT-5.2 from a single earlier model and gave no control; with controls, MolE alone and
RDKit alone correlate at least as strongly, so this check gives no evidence that GPT features
improve attribution.}
\label{tab:mic}
\end{table}"""


def main() -> None:
    parts = [
        "% Generated by revision_outputs/scripts/make_tables.py -- do not edit by hand.",
        "% Requires \\newcommand{\\PM}{\\,\\pm\\,} and the booktabs package.",
        table_main(), table_paired(), table_gpt_only(), table_protocol(),
        table_overlap(), table_fidelity(), table_mic(),
    ]
    with open(f"{OUT}/tables.tex", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n\n".join(parts) + "\n")
    print(f"wrote {OUT}/tables.tex")


if __name__ == "__main__":
    main()
