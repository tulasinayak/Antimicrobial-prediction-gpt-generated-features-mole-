"""Write the two Task 8 Kaggle notebooks, reusing the setup cells of gpt_mole_trials.ipynb.

    python -I kaggle/make_task8_notebooks.py
"""

from __future__ import annotations

import copy
import json

SRC = json.load(open("kaggle/gpt_mole_trials.ipynb", encoding="utf-8"))
# Cells 1 and 2 (GPU check, dependencies) are reused as they are; the setup cell is
# written out below because Task 8 needs a newer bundle than that notebook assumes.
GPU, DEPS = (copy.deepcopy(SRC["cells"][i]) for i in (1, 2))


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def copy_cell(out_root: str, restore_name: str, extra_needed: str = "") -> dict:
    """Setup cell: pick the right attached bundle, copy it, re-gzip inputs, check them.

    Written out in full rather than patched from gpt_mole_trials.ipynb, because Task 8
    needs files only the newer bundle has. Several dataset versions can be attached at
    once and a plain glob may return the stale one, so the bundle is chosen by a marker
    file instead of taking the first match.
    """
    return code('''# 3. Copy the bundle to a writable folder (inputs are read-only) and work from there.
#    Also restores finished runs from a previous version's output, if one is attached.
import glob, gzip, importlib, os, shutil, sys

MARKER = "scripts/run_mole_protocol_search.py"   # present only in the Task 8 bundle
found = [os.path.dirname(os.path.dirname(h))
         for h in glob.glob("/kaggle/input/**/notebooks/build_train.py", recursive=True)]
assert found, "Bundle not found - attach the dataset made from gpt_mole_bundle.zip."
# Several dataset versions can be attached at once, so use one that has the new files
# rather than whichever the glob happens to return first.
usable = [b for b in found if os.path.exists(os.path.join(b, MARKER))]
if not usable:
    listing = "\\n".join(
        f"  {b}  (run_table1.py: {os.path.getsize(os.path.join(b, 'scripts/run_table1.py')):,} bytes)"
        for b in found)
    raise AssertionError(
        "The attached dataset is an OLD version - it does not contain " + MARKER + ".\\n"
        "Attached bundle(s):\\n" + listing + "\\n"
        "Fix: upload the new gpt_mole_bundle.zip to the dataset via New Version, then in this\\n"
        "notebook's Input panel remove the dataset and add it again so the newest version is\\n"
        "used (run_table1.py is then ~12,463 bytes and data/gpt_guided_flat/ exists).")
BUNDLE = usable[0]
WORK = "/kaggle/working/gpt_mole"
# dirs_exist_ok refreshes code and data in place while keeping any results/ already
# computed, so a newer dataset version takes effect without restarting the kernel.
shutil.copytree(BUNDLE, WORK, dirs_exist_ok=True)
os.chdir(WORK)
print("bundle:", BUNDLE, "-> working in", os.getcwd())
if os.path.exists("MANIFEST.txt"):
    print(open("MANIFEST.txt").read())

sys.path.insert(0, "scripts")
import run_table1
importlib.reload(run_table1)   # in case an older copy was imported earlier in this kernel

# Kaggle decompresses .gz files inside an uploaded dataset, so x.tsv.gz arrives as
# x.tsv. Re-compress them to the names the scripts expect.
needed = [run_table1.SF3, run_table1.SPLIT, run_table1.MOLE, run_table1.PRESTWICK,
          run_table1.CHEMDESC, *run_table1.FREEFORM.values()EXTRA]
for path in needed:
    if path.endswith(".gz") and not os.path.exists(path) and os.path.exists(path[:-3]):
        with open(path[:-3], "rb") as src, gzip.open(path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print("re-compressed", path)
missing = [p for p in needed if not os.path.exists(p)]
assert not missing, f"Missing inputs: {missing}"
print(f"all {len(needed)} inputs present")

OUT_ROOT = "OUT_ROOT_VALUE"
previous = [d for d in glob.glob("/kaggle/input/**/RESTORE_NAME", recursive=True)
            if not d.startswith(BUNDLE)]
for d in previous:
    shutil.copytree(d, OUT_ROOT, dirs_exist_ok=True)
    print("restored finished runs from", d)'''
                .replace("EXTRA", extra_needed)
                .replace("OUT_ROOT_VALUE", out_root)
                .replace("RESTORE_NAME", restore_name))


def save(path: str, cells: list[dict]) -> None:
    nb = {"cells": cells, "metadata": SRC["metadata"], "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    for i, c in enumerate(cells):
        if c["cell_type"] == "code":
            compile("".join(c["source"]).replace("!python", "#!python"), f"{path}:{i}", "exec")
    print("wrote", path)


GPU_LOG_CHECK = code('''# 6. Every run must have trained on the GPU.
logs = glob.glob(f"{OUT_ROOT}/trial_*/*/run.log")
cpu_runs = [p for p in logs if "No visible GPU" in open(p, encoding="utf-8", errors="ignore").read()]
print(f"{len(logs)} run logs checked")
assert logs, "No run logs found - the runs above did not start."
assert not cpu_runs, f"{len(cpu_runs)} runs fell back to CPU: {cpu_runs}"
print("all runs on GPU")''')


def task8b() -> None:
    save("kaggle/task8b_gptonly_guided.ipynb", [
        md("""# Task 8b: GPT-only and guided rows under the 5-trial protocol (GPU)

18 extra rows x 5 trials = 90 runs, same protocol as the main table (seed fix, MolE's scaffold split, trial k = search seed and final training seed k):

- `gpt_only_<model>`: free-form GPT features alone (6 rows)
- `mole_plus_guided_<model>`: MolE + guided (schema-constrained) GPT features (6 rows)
- `guided_only_<model>`: guided GPT features alone (6 rows)

Guided inputs are `data/gpt_guided_flat/gpt_features_guided_<model>.csv`, made by `scripts/flatten_guided.py` from the six `gpt_features_4o_<model>_unified.csv` files (one schema, 46 features, the same flattening for both guided rows).

**Setup:** upload the new `gpt_mole_bundle.zip` as a new version of your dataset, attach it, Accelerator = GPU, Internet on. Run cells 1-4 to check, then **Save Version -> Save & Run All**. If the session ends early, attach this notebook's previous output and run again; finished runs are skipped."""),
        GPU, DEPS,
        copy_cell("results/table1_trials", "table1_trials", ", *run_table1.GUIDED.values()"),
        code('''# 4. One quick run first (trial 0, guided GPT-5.2 only) to check the pipeline.
!python -u scripts/run_table1.py --trials 0 --only guided_only_gpt-5.2 --include-guided --only-extra --out-root {OUT_ROOT} --device cuda --resume
assert _exit_code == 0, "run failed - see the output above and its run.log"'''),
        code('''# 5. Full run: 18 rows x 5 trials. Set TRIALS to a subset (e.g. "0,1,2") to split across sessions.
TRIALS = "0,1,2,3,4"
!python -u scripts/run_table1.py --trials {TRIALS} --include-gpt-only --include-guided --only-extra --out-root {OUT_ROOT} --device cuda --resume
assert _exit_code == 0, "some runs failed - see the list above and each run.log"'''),
        GPU_LOG_CHECK,
        code('''# 7. Aggregate: mean +/- sd over trials per row -> table1.csv.
!python scripts/aggregate_table1.py --root {OUT_ROOT}
assert os.path.exists(f"{OUT_ROOT}/table1.csv"), "aggregation produced no table1.csv"'''),
        code('''# 8. Package for download (Output tab -> task8b_results.zip).
archive = shutil.make_archive("/kaggle/working/task8b_results", "zip", OUT_ROOT)
print(archive, f"{os.path.getsize(archive) / 1024 ** 2:.1f} MB")'''),
    ])


def task8a() -> None:
    save("kaggle/task8a_mole_protocol.ipynb", [
        md("""# Task 8a: MolE's full original protocol, 5 trials (GPU)

Rows: MolE, MolE + GPT-5.2, GPT-5.2 only. Per trial k, 5 configurations are drawn (seed k, shared by all rows) from MolE's search space (n_estimators 30-1000, max_depth 5-100, eta 0.05-1, subsample 0.3-1), each fitted 5 times; the configuration with the best mean validation PR-AUC is selected. Raw features, no early stopping, no class weighting, pooled PR-AUC = auc(recall, precision), as in MolE's code. The GPU uses `hist`; MolE used `exact` (MolE alone: 0.7802 exact vs 0.7655 hist on CPU).

Every fit is appended to `fits.csv`, so the run resumes after an interruption: attach this notebook's previous output and run again.

**Setup:** attach the new `gpt_mole_bundle.zip` dataset version, Accelerator = GPU, Internet on, then **Save Version -> Save & Run All**."""),
        GPU, DEPS,
        copy_cell("results/mole_protocol_trials", "mole_protocol_trials"),
        code('''# 4. Run all fits (resumes from fits.csv). Add rdkit_only,mole_plus_rdkit to ROWS if time allows.
ROWS = "mole_only,mole_plus_gpt-5.2,gpt_only_gpt-5.2"
!python -u scripts/run_mole_protocol_search.py --rows {ROWS} --trials 0,1,2,3,4 --device cuda --output_dir {OUT_ROOT}
assert _exit_code == 0, "run failed - see the output above"'''),
        code('''# 5. Table: mean +/- sd over trials and paired difference vs MolE; then the selected configuration per trial.
import pandas as pd
pd.set_option("display.width", 220)
display(pd.read_csv(f"{OUT_ROOT}/table_mole_protocol.csv").round(4))
pd.read_csv(f"{OUT_ROOT}/selected_per_trial.csv").round(4)'''),
        code('''# 6. Package for download (Output tab -> task8a_results.zip).
archive = shutil.make_archive("/kaggle/working/task8a_results", "zip", OUT_ROOT)
print(archive, f"{os.path.getsize(archive) / 1024 ** 2:.1f} MB")'''),
    ])


if __name__ == "__main__":
    task8b()
    task8a()
