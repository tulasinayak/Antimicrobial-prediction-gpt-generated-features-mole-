"""Write the two Task 8 Kaggle notebooks, reusing the setup cells of gpt_mole_trials.ipynb.

    python -I kaggle/make_task8_notebooks.py
"""

from __future__ import annotations

import copy
import json

SRC = json.load(open("kaggle/gpt_mole_trials.ipynb", encoding="utf-8"))
GPU, DEPS, COPY = (copy.deepcopy(SRC["cells"][i]) for i in (1, 2, 3))
COPY_SRC = "".join(COPY["source"])


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def copy_cell(out_root: str, restore_name: str, extra_needed: str = "") -> dict:
    s = COPY_SRC.replace('OUT_ROOT = "results/table1_trials"', f'OUT_ROOT = "{out_root}"')
    s = s.replace('glob.glob("/kaggle/input/**/table1_trials", recursive=True)',
                  f'glob.glob("/kaggle/input/**/{restore_name}", recursive=True)')
    s = s.replace("run_table1.CHEMDESC, *run_table1.FREEFORM.values()]",
                  "run_table1.CHEMDESC, *run_table1.FREEFORM.values()" + extra_needed + "]")
    assert out_root in s and restore_name in s and (not extra_needed or extra_needed in s)
    cell = copy.deepcopy(COPY)
    cell["source"] = s.splitlines(keepends=True)
    return cell


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
