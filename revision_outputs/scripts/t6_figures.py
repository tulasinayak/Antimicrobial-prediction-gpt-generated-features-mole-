"""Task 6: figures for the AAAI two-column format (vector PDF + 300 dpi PNG, no titles).

Reads the CSVs written by t2-t5 from revision_outputs/data/ (run those first).
Colours: blue #3B6FB6 / vermillion #D55E00 (passes the CVD and contrast checks of the
dataviz validator), always paired with a marker-shape difference so the figures also read
in greyscale; heatmaps use a single-hue sequential map (monotone lightness).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from scipy import stats  # noqa: E402

from common import DATA, FIGS, GPT_LABEL, GPT_MODELS, ensure_dirs  # noqa: E402

BLUE, VERMILLION, GREY, INK, MUTED = "#3B6FB6", "#D55E00", "#BDBDBD", "#1a1a1a", "#5a5a5a"
ORDER = [GPT_LABEL[m] for m in GPT_MODELS]
SINGLE, FULL = 3.3, 7.0

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.titlesize": 8, "font.family": "DejaVu Sans",
    "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK,
    "savefig.dpi": 300, "figure.dpi": 150,
})
# Sequential single hue, white -> deep blue.
SEQ = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.04, 0.95, 256)))


def save(fig, name: str) -> None:
    fig.savefig(f"{FIGS}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(f"{FIGS}/{name}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


def heatmap(ax, mat: pd.DataFrame, vmax: float, fmt: str, mask_diag: bool, cbar_label: str):
    data = mat.loc[ORDER, ORDER].to_numpy(float)
    shown = np.ma.masked_where(np.eye(len(data), dtype=bool), data) if mask_diag else data
    cmap = SEQ.copy()
    cmap.set_bad(GREY)
    im = ax.imshow(shown, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    for i in range(len(data)):
        for j in range(len(data)):
            if mask_diag and i == j:
                continue
            v = data[i, j]
            dark = v / vmax > 0.55
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=6.5,
                    color="white" if dark else INK)
    ax.set_xticks(range(len(ORDER)), ORDER, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(len(ORDER)), ORDER)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7, length=2)
    return im


def fig1() -> None:
    shared = pd.read_csv(f"{DATA}/overlap_exact_shared_counts.csv", index_col=0)
    jac = pd.read_csv(f"{DATA}/overlap_exact.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"wspace": 0.55})
    off = shared.where(~np.eye(len(shared), dtype=bool))
    heatmap(axes[0], shared, vmax=float(off.max().max()) * 1.05, fmt=".0f", mask_diag=False,
            cbar_label="Shared feature names")
    # Diagonal of the count panel = each model's total; colour it as grey to keep the
    # off-diagonal scale readable, but print the totals.
    data = shared.loc[ORDER, ORDER].to_numpy()
    for i in range(len(ORDER)):
        axes[0].add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color=GREY, zorder=2))
        axes[0].text(i, i, f"{data[i, i]:.0f}", ha="center", va="center", fontsize=6.5, color=INK, zorder=3)
    heatmap(axes[1], jac, vmax=0.15, fmt=".2f", mask_diag=True, cbar_label="Jaccard similarity")
    save(fig, "fig1_overlap")

    norm = pd.read_csv(f"{DATA}/overlap_normalized.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"wspace": 0.55})
    heatmap(axes[0], jac, vmax=0.15, fmt=".2f", mask_diag=True, cbar_label="Jaccard, exact names")
    heatmap(axes[1], norm, vmax=0.15, fmt=".2f", mask_diag=True, cbar_label="Jaccard, normalized names")
    save(fig, "fig1b_overlap_exact_vs_normalized")


def fig2(reference: str, name: str) -> None:
    pts = pd.read_csv(f"{DATA}/fidelity_fig2_points.csv")
    panels = [
        ("molecular_weight", "MolWt", "molecular weight (g/mol)", "GPT-5.2 estimate (g/mol)"),
        ("topological_polar_surface_area", "TPSA", "TPSA (Å²)", "GPT-5.2 estimate (Å²)"),
        ("fraction_csp3", "FractionCSP3", "FractionCSP3", "GPT-5.2 estimate"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(FULL, 2.2), gridspec_kw={"wspace": 0.38})
    for ax, (gcol, rcol, xlab, ylab) in zip(axes, panels):
        x = pts[f"rdkit_{reference}_{rcol}"]
        y = pts[f"gpt_{gcol}"]
        ok = x.notna() & y.notna()
        x, y = x[ok], y[ok]
        zero = y == 0
        rho = stats.spearmanr(x, y).statistic
        hi = float(np.nanmax(np.r_[x, y])) * 1.03
        lo = -0.03 * hi  # small margin so exact zeros sit inside the axes
        ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color=MUTED, zorder=1)
        ax.scatter(x[~zero], y[~zero], s=6, color=BLUE, alpha=0.45, linewidths=0, zorder=2,
                   label="GPT estimate")
        ax.scatter(x[zero], y[zero], s=14, color=VERMILLION, marker="x", linewidths=0.8, zorder=3,
                   label="GPT value exactly 0")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel(f"RDKit {xlab}")
        ax.set_ylabel(ylab)
        ax.text(0.04, 0.96, f"Spearman ρ = {rho:.2f}\nn zeros = {int(zero.sum())}", transform=ax.transAxes,
                va="top", ha="left", fontsize=7, color=INK, zorder=4,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_aspect("equal", adjustable="box")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.canvas.draw()  # resolve the equal-aspect boxes before reading their top edge
    top = max(ax.get_position().y1 for ax in axes)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, top + 0.01), ncol=2, frameon=False,
               handletextpad=0.2, columnspacing=1.5, markerscale=1.4)
    save(fig, name)


PRETTY = {
    "heteroatom_count": "Heteroatom count", "cyp_inhibition_risk_est": "CYP inhibition risk",
    "pka_basic_min_est": "Min. basic pKa", "solubility_class_est": "Solubility class",
    "logp_xlogp3_est": "logP (XLogP3-like)", "logd7_4_est": "logD at pH 7.4",
    "desolvation_penalty_est": "Desolvation penalty",
    "enzyme_active_site_mimicry_score_est": "Enzyme active-site mimicry",
    "strong_acidic_site_count": "Strong acidic sites", "biofilm_penetration_score_est": "Biofilm penetration",
}


def pretty(feature: str) -> str:
    if feature in PRETTY:
        return PRETTY[feature]
    words = [w for w in feature.split("_") if w not in ("est", "score")]
    return " ".join(words).capitalize()


def fig3() -> None:
    imp = pd.read_csv(f"{DATA}/importance_gpt52.csv")
    top = imp[imp.block == "gpt"].sort_values("mean_gain", ascending=False).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.8))
    y = np.arange(len(top))
    ax.barh(y, top.max_abs_spearman_rdkit, height=0.62, color=BLUE, zorder=2)
    for yi, (v, d) in enumerate(zip(top.max_abs_spearman_rdkit, top.best_rdkit_descriptor)):
        ax.text(v + 0.015, yi, d, va="center", ha="left", fontsize=6.5, color=MUTED, zorder=3,
                bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
    ax.axvline(0.5, ls="--", lw=0.8, color=MUTED, zorder=1)
    labels = [pretty(f) for f in top.feature]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Max |Spearman ρ| with any RDKit descriptor")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "fig3_coverage")


def fig4_axis(ax, pts: pd.DataFrame, col: str, ylabel: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-0.12, 0.12, len(pts))
    for gram, color, marker, label in (("positive", BLUE, "o", "Gram-positive"),
                                       ("negative", VERMILLION, "^", "Gram-negative")):
        sub = pts.gram == gram
        ax.scatter(pts.log2MIC[sub] + jitter[sub], pts[col][sub], s=18, color=color, marker=marker,
                   alpha=0.8, edgecolors="white", linewidths=0.5, zorder=3, label=label)
    ols = stats.linregress(pts.log2MIC, pts[col])
    xs = np.linspace(pts.log2MIC.min() - 0.3, pts.log2MIC.max() + 0.3, 50)
    ax.plot(xs, ols.intercept + ols.slope * xs, color=INK, lw=1.0, zorder=2)
    ax.set_xlabel("log$_2$(MIC, µg/mL)")
    ax.set_ylabel(ylabel)


def fig4() -> None:
    pts = pd.read_csv(f"{DATA}/mic_points.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    fig4_axis(ax, pts, "G_mole_gpt52", "Antimicrobial potential G")
    ax.legend(loc="upper right", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig4_mic")

    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.6), sharey=True, gridspec_kw={"wspace": 0.12})
    fig4_axis(axes[0], pts, "G_mole_gpt52", "Antimicrobial potential G")
    fig4_axis(axes[1], pts, "G_mole", "")
    axes[0].text(0.03, 0.04, "MolE + GPT-5.2", transform=axes[0].transAxes, fontsize=7, color=MUTED)
    axes[1].text(0.03, 0.04, "MolE", transform=axes[1].transAxes, fontsize=7, color=MUTED)
    axes[1].legend(loc="upper right", frameon=False, handletextpad=0.2, borderaxespad=0.2)
    save(fig, "fig4b_mic_mole_gpt52_vs_mole")


def main() -> None:
    ensure_dirs()
    fig1()
    fig2("input", "fig2_fidelity")
    fig2("98file", "fig2b_fidelity_as_submitted")
    fig3()
    fig4()
    print("figures written to", FIGS)


if __name__ == "__main__":
    main()
