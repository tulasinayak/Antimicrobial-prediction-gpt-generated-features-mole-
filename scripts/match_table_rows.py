"""Map every number in the paper's tables back to the run that produced it.

For each stored test_predictions*.csv this prints the four quantities the paper
tabulates - pooled PR-AUC, mean per-strain PR-AUC, pooled F1 and accuracy - so a
reported row can be matched to its run unambiguously.
"""

import glob
import os

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score

REQUIRED = {"label", "proba", "bacterium", "pred_f1"}


def summarize(path: str) -> dict | None:
    df = pd.read_csv(path)
    if not REQUIRED <= set(df.columns):
        return None
    macro = df.groupby("bacterium").apply(
        lambda g: average_precision_score(g.label, g.proba)
    ).mean()
    name = (path.replace(os.sep, "/")
                .replace("results/", "")
                .replace("/test_predictions", " ")
                .replace(".csv", ""))
    return {
        "run": name,
        "n": len(df),
        "pooled": round(average_precision_score(df.label, df.proba), 4),
        "macro": round(macro, 4),
        "f1": round(f1_score(df.label, df.pred_f1), 4),
        "acc": round(100 * accuracy_score(df.label, df.pred_f1), 2),
    }


def main() -> None:
    rows = [s for p in glob.glob("results/**/test_predictions*.csv", recursive=True)
            if (s := summarize(p)) is not None]
    df = pd.DataFrame(rows).sort_values("acc", ascending=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
