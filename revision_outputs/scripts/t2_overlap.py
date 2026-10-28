"""Task 2: cross-model vocabulary overlap at three levels of strictness.

EXACT       feature names compared as written (metadata columns removed).
NORMALIZED  names tokenised (camelCase and separators), lower-cased, hedge tokens dropped,
            num/number -> n, then mapped through SYNONYMS (printed in results.md).
EMBEDDING   names split into words and embedded with sentence-transformers/all-MiniLM-L6-v2
            (mean pooling + L2 normalisation, as sentence-transformers does); two features
            match if cosine similarity >= threshold. Overlap between two models is a maximum
            one-to-one matching, so with exact strings it reduces to set intersection.
"""

from __future__ import annotations

import collections
import itertools
import random
import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from common import (DATA, GPT_FILES, GPT_LABEL, GPT_MODELS, METADATA_COLS, ensure_dirs,
                    write_section)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
THRESHOLDS = [0.80, 0.85, 0.90]

HEDGE = {"est", "estimated", "estimate", "approx", "approximate", "predicted", "pred",
         "calc", "calculated", "score", "index", "value"}
RENAME_TOKENS = {"num": "n", "number": "n"}   # "count" is kept as "count"

# Applied to the joined normalised string. Groups listed in the task plus their count/n
# variants (e.g. hbd_count, num_h_donors) - all are the same RDKit-style quantity.
SYNONYM_GROUPS = {
    "logp": ["logp", "clogp", "xlogp", "xlogp3", "mollogp"],
    "tpsa": ["tpsa", "topologicalpolarsurfacearea"],
    "mw": ["mw", "molecularweight", "molwt"],
    "hbd": ["hbd", "hbdcount", "nhbd", "hbonddonors", "hbonddonor", "hbonddonorcount",
            "nhbonddonors", "nhdonors", "hydrogenbonddonors", "hydrogenbonddonorcount",
            "nhydrogenbonddonors"],
    "hba": ["hba", "hbacount", "nhba", "hbondacceptors", "hbondacceptor", "hbondacceptorcount",
            "nhbondacceptors", "nhacceptors", "hydrogenbondacceptors", "hydrogenbondacceptorcount",
            "nhydrogenbondacceptors"],
    "fsp3": ["fsp3", "fractioncsp3", "fractionsp3", "csp3fraction"],
    "rotb": ["rotatablebonds", "nrotatablebonds", "rotatablebondcount", "nrotatablebond",
             "rotbonds", "nrotbonds"],
}
SYNONYMS = {v: k for k, vs in SYNONYM_GROUPS.items() for v in vs}


def feature_names(model: str) -> list[str]:
    cols = list(pd.read_csv(GPT_FILES[model], nrows=1).columns)
    return [c for c in cols if c not in METADATA_COLS]


def tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))     # camelCase -> camel Case
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", s)          # HTTPServer -> HTTP Server
    return [t.lower() for t in re.split(r"[^A-Za-z0-9]+", s) if t]


def normalize(name: str) -> str:
    toks = [RENAME_TOKENS.get(t, t) for t in tokens(name) if t not in HEDGE]
    key = "".join(toks) or "".join(tokens(name))
    return SYNONYMS.get(key, key)


def overlap_tables(sets: dict[str, set]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shared-count matrix, Jaccard matrix and per-model summary for exact set matching."""
    models = list(sets)
    shared = pd.DataFrame(0, index=models, columns=models)
    jac = pd.DataFrame(1.0, index=models, columns=models)
    for a, b in itertools.product(models, models):
        inter = len(sets[a] & sets[b])
        shared.loc[a, b] = inter
        if a != b:
            jac.loc[a, b] = inter / len(sets[a] | sets[b])
    count = collections.Counter(f for s in sets.values() for f in s)
    summary = pd.DataFrame([{
        "model": m, "total": len(sets[m]),
        "model_specific": sum(count[f] == 1 for f in sets[m]),
    } for m in models])
    return shared, jac, summary


def embed(texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    model = AutoModel.from_pretrained(EMBED_MODEL).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            batch = tok(texts[i:i + 256], padding=True, truncation=True, return_tensors="pt")
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).float()
            vec = (hidden * mask).sum(1) / mask.sum(1)
            out.append(torch.nn.functional.normalize(vec, dim=1).numpy())
    return np.vstack(out)


def max_matching(sim: np.ndarray, thr: float) -> list[tuple[int, int]]:
    """Maximum one-to-one matching among pairs with sim >= thr (ties broken by similarity)."""
    ok = sim >= thr
    if not ok.any():
        return []
    weight = np.where(ok, 1.0 + sim, 0.0)
    r, c = linear_sum_assignment(-weight)
    return [(i, j) for i, j in zip(r, c) if ok[i, j]]


def embedding_level(names: dict[str, list[str]], vecs: dict[str, np.ndarray], thr: float):
    models = list(names)
    shared = pd.DataFrame(0, index=models, columns=models)
    jac = pd.DataFrame(1.0, index=models, columns=models)
    has_match = {m: np.zeros(len(names[m]), bool) for m in models}
    pairs = []
    for a, b in itertools.combinations(models, 2):
        sim = vecs[a] @ vecs[b].T
        match = max_matching(sim, thr)
        k = len(match)
        shared.loc[a, b] = shared.loc[b, a] = k
        jac.loc[a, b] = jac.loc[b, a] = k / (len(names[a]) + len(names[b]) - k)
        # model-specific = no other model has a feature above threshold (not only matched ones)
        has_match[a] |= (sim >= thr).any(1)
        has_match[b] |= (sim >= thr).any(0)
        pairs += [(a, names[a][i], b, names[b][j], float(sim[i, j])) for i, j in match]
    for m in models:
        shared.loc[m, m] = len(names[m])
    summary = pd.DataFrame([{"model": m, "total": len(names[m]),
                             "model_specific": int((~has_match[m]).sum())} for m in models])
    return shared, jac, summary, pairs


def summary_md(summary: pd.DataFrame, jac: pd.DataFrame, title: str) -> list[str]:
    s = summary.copy()
    s["pct"] = 100 * s.model_specific / s.total
    s["shared"] = s.total - s.model_specific
    off = jac.where(~np.eye(len(jac), dtype=bool)).stack()
    lines = [f"### {title}", "",
             "| Model | Total | Model-specific | % specific | Shared with >=1 other |",
             "|---|---|---|---|---|"]
    lines += [f"| {GPT_LABEL[r.model]} | {r.total} | {r.model_specific} | {r.pct:.1f}% | {r.shared} |"
              for r in s.itertuples()]
    lines += ["", "Pairwise Jaccard:", "",
              "| | " + " | ".join(GPT_LABEL[m] for m in jac.columns) + " |",
              "|---|" + "---|" * len(jac.columns)]
    for m in jac.index:
        lines.append(f"| {GPT_LABEL[m]} | " + " | ".join(
            "-" if m == c else f"{jac.loc[m, c]:.3f}" for c in jac.columns) + " |")
    lines += ["", f"Off-diagonal Jaccard: min {off.min():.3f}, max {off.max():.3f}, "
                  f"mean {off.mean():.3f}. Model-specific share: {s.pct.min():.1f}%-{s.pct.max():.1f}%.", ""]
    return lines


def save(shared: pd.DataFrame, jac: pd.DataFrame, level: str) -> None:
    lab = [GPT_LABEL[m] for m in jac.index]
    jac.set_axis(lab, axis=0).set_axis(lab, axis=1).to_csv(f"{DATA}/overlap_{level}.csv")
    shared.set_axis(lab, axis=0).set_axis(lab, axis=1).to_csv(f"{DATA}/overlap_{level}_shared_counts.csv")


def main() -> None:
    ensure_dirs()
    names = {m: feature_names(m) for m in GPT_MODELS}
    for m, ns in names.items():
        with open(f"{DATA}/feature_names_{m}.txt", "w", encoding="utf-8") as fh:
            fh.write("\n".join(ns) + "\n")

    md = ["## Task 2. Vocabulary overlap at three levels of strictness", "",
          "Feature-name lists: `data/feature_names_<model>.txt`. Metadata columns removed: "
          "`prestwick_ID`, `ProductName`, `smiles_resolution`, `input` (`ProductName` is absent "
          "from the free-form files). Matrices: `data/overlap_<level>.csv` (Jaccard) and "
          "`data/overlap_<level>_shared_counts.csv` (shared counts; diagonal = total).", ""]

    # (a) exact
    shared, jac, summ = overlap_tables({m: set(v) for m, v in names.items()})
    save(shared, jac, "exact")
    md += summary_md(summ, jac, "(a) EXACT")

    # (b) normalized
    norm = {m: {normalize(n) for n in v} for m, v in names.items()}
    shared, jac, summ = overlap_tables(norm)
    save(shared, jac, "normalized")
    collapsed = {m: len(names[m]) - len(norm[m]) for m in names}
    md += summary_md(summ, jac, "(b) NORMALIZED")
    md += ["Totals shrink where several names of one model normalise to the same key "
           "(collapsed per model: " + ", ".join(f"{GPT_LABEL[m]} {c}" for m, c in collapsed.items())
           + ").", "",
           "Normalisation rules: split camelCase and on any non-alphanumeric separator; lower-case; "
           "drop hedge tokens " + ", ".join(f"`{h}`" for h in sorted(HEDGE)) + "; rename tokens "
           + ", ".join(f"`{k}`->`{v}`" for k, v in RENAME_TOKENS.items())
           + " (`count` kept); join without separators; then apply the synonym map:", "",
           "| Canonical | Normalised forms mapped to it |", "|---|---|"]
    md += [f"| `{k}` | " + ", ".join(f"`{v}`" for v in vs) + " |" for k, vs in SYNONYM_GROUPS.items()]
    examples = [(m, n, normalize(n)) for m in names for n in names[m]]
    rng = random.Random(0)
    md += ["", "Examples (random): " + "; ".join(f"`{n}` -> `{k}`" for _, n, k in rng.sample(examples, 12)), ""]

    # (c) embedding
    texts = {m: [" ".join(tokens(n)) for n in v] for m, v in names.items()}
    allv = embed([t for m in GPT_MODELS for t in texts[m]])
    vecs, i = {}, 0
    for m in GPT_MODELS:
        vecs[m] = allv[i:i + len(texts[m])]
        i += len(texts[m])
    sample_pairs = None
    for thr in THRESHOLDS:
        shared, jac, summ, pairs = embedding_level(names, vecs, thr)
        level = f"embedding_{thr:.2f}"
        save(shared, jac, level)
        md += summary_md(summ, jac, f"(c) EMBEDDING, cosine >= {thr:.2f}")
        if abs(thr - 0.85) < 1e-9:
            sample_pairs = pairs
    nontrivial = [p for p in sample_pairs if p[1] != p[3]]
    rng = random.Random(0)
    pick = rng.sample(nontrivial, min(20, len(nontrivial)))
    md += [f"20 random matched pairs at cosine >= 0.85 (drawn from the {len(nontrivial)} matched pairs "
           f"whose names are not identical; {len(sample_pairs) - len(nontrivial)} further matches are "
           "identical names):", "", "| Model A | Feature A | Model B | Feature B | Cosine |", "|---|---|---|---|---|"]
    md += [f"| {GPT_LABEL[a]} | `{fa}` | {GPT_LABEL[b]} | `{fb}` | {s:.3f} |" for a, fa, b, fb, s in pick]
    pd.DataFrame(sample_pairs, columns=["model_a", "feature_a", "model_b", "feature_b", "cosine"]) \
        .to_csv(f"{DATA}/embedding_matches_0.85.csv", index=False)

    write_section("task2", "\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
