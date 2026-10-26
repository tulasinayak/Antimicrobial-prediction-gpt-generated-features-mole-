"""Build the Kaggle upload bundle for the Table 1 trial runs.

Packs exactly the inputs and scripts that scripts/run_table1.py needs into
kaggle/gpt_mole_bundle.zip, under a top-level gpt_mole/ folder that keeps the
repository layout, so the scripts' relative paths work unchanged once copied.

Upload the zip as a new Kaggle dataset (Kaggle unpacks it), attach that dataset
to kaggle/gpt_mole_trials.ipynb, and run the notebook on a GPU.

Usage:
    python -I kaggle/build_bundle.py
"""

from __future__ import annotations

import hashlib
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import run_table1  # noqa: E402  (the single source of truth for the input paths)

SCRIPTS = [
    "notebooks/build_train.py",
    "scripts/run_table1.py",
    "scripts/aggregate_table1.py",
    "scripts/reproduce_mole_original.py",
    "scripts/run_mole_method.py",
]
OUT = "kaggle/gpt_mole_bundle.zip"
TOP = "gpt_mole"


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def main() -> int:
    inputs = [run_table1.SF3, run_table1.SPLIT, run_table1.MOLE, run_table1.PRESTWICK,
              run_table1.CHEMDESC, *run_table1.FREEFORM.values()]
    files = inputs + SCRIPTS

    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        print("Missing files (run from the repository root):")
        for m in missing:
            print(f"  {m}")
        return 1

    manifest = []
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f"{TOP}/{f}")
            manifest.append(f"{sha256(f)}  {os.path.getsize(f):>10,d}  {f}")
        zf.writestr(f"{TOP}/MANIFEST.txt", "\n".join(manifest) + "\n")

    print("\n".join(manifest))
    print(f"\nWrote {OUT} ({os.path.getsize(OUT) / 1024 ** 2:.1f} MB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
