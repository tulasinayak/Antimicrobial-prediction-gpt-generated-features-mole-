"""Check whether our scaffold split is the one published with the MolE paper.

Downloads the split distributed in rolayoalarcon/mole_antimicrobial_potential and
compares it compound-by-compound against data/raw_data/maier_scaffold_split.tsv.
"""

import gzip
import io
import urllib.request

import pandas as pd

HOST = "raw." + "github" + "usercontent.com"
PATH = "/rolayoalarcon/mole_antimicrobial_potential/main/data/01.prepare_training_data/maier_scaffold_split.tsv.gz"
URL = f"https://{HOST}{PATH}"

LOCAL = "data/raw_data/maier_scaffold_split.tsv"
NORMALIZE = {"validation": "valid", "val": "valid", "dev": "valid"}


def load_remote() -> pd.DataFrame:
    with urllib.request.urlopen(URL, timeout=60) as resp:
        raw = resp.read()
    with gzip.open(io.BytesIO(raw), "rt") as fh:
        return pd.read_csv(fh, sep="\t")


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    id_col = next(c for c in df.columns if "prestw" in c.lower())
    split_col = next(c for c in df.columns
                     if c.lower().strip() in {"split", "set", "partition"})
    out = df[[id_col, split_col]].copy()
    out.columns = ["prestwick_ID", "split"]
    out["prestwick_ID"] = out.prestwick_ID.astype(str).str.strip().str.lower()
    out["split"] = out.split.astype(str).str.strip().str.lower().replace(NORMALIZE)
    return out.sort_values("prestwick_ID").reset_index(drop=True)


def main() -> None:
    remote = normalize(load_remote())
    local = normalize(pd.read_csv(LOCAL, sep="\t"))

    print(f"published : {remote.shape}  {remote.split.value_counts().to_dict()}")
    print(f"ours      : {local.shape}  {local.split.value_counts().to_dict()}")

    if remote.prestwick_ID.tolist() != local.prestwick_ID.tolist():
        only_pub = set(remote.prestwick_ID) - set(local.prestwick_ID)
        only_ours = set(local.prestwick_ID) - set(remote.prestwick_ID)
        print(f"compound sets differ: {len(only_pub)} only published, "
              f"{len(only_ours)} only ours")
        return

    merged = remote.merge(local, on="prestwick_ID", suffixes=("_pub", "_ours"))
    agree = (merged.split_pub == merged.split_ours)
    print(f"identical compound set: yes ({len(merged)} compounds)")
    print(f"split labels agree    : {agree.sum()}/{len(merged)}")
    if not agree.all():
        print(pd.crosstab(merged.split_pub, merged.split_ours))


if __name__ == "__main__":
    main()
