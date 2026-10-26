"""Regenerate every row of Table 1 under one protocol, over several training seeds.

Each row is produced by the same build_train.py invocation, with the same exact
scaffold split (MolE's published one) and the same metric, so the resulting table
cannot mix pooled and per-strain PR-AUC the way the submitted Table 1 does.

Rows produced:
  mole_only            MolE embeddings alone
  rdkit_only           98-descriptor chemical-descriptor set alone
  mole_plus_rdkit      MolE + descriptors   <- missing from the submission entirely
  mole_plus_<model>    MolE + each of the six free-form GPT feature sets
  gpt_only_<model>     each GPT feature set alone (with --include-gpt-only)

Usage:
    python scripts/run_table1.py --dry-run
    python scripts/run_table1.py --seeds 0,1,2,3,4
    python scripts/run_table1.py --only mole_only,rdkit_only
    python scripts/run_table1.py --trials 0,1,2,3,4 --out-root results/table1_trials --resume
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

RAW = "data/raw_data"
SF3 = f"{RAW}/screen_results_info_SF3.xlsx"
SPLIT = f"{RAW}/maier_scaffold_split.tsv"
MOLE = f"{RAW}/maier_mole_representation_with_id.tsv.gz"
PRESTWICK = f"{RAW}/prestwick_library_screened.tsv.gz"
CHEMDESC = f"{RAW}/maier_chemdesc_representation.tsv"

FREEFORM = {
    "gpt-4o":        "data/gpt_generation_without_guidance/gpt_features4o_gpt-4o_freeform.csv",
    "gpt-4o-mini":   "data/gpt_generation_without_guidance/gpt_features4omini_gpt-4o-mini_freeform.csv",
    "gpt-4.1":       "data/gpt_generation_without_guidance/gpt_features_4_gpt-4.1_freeform.csv",
    "gpt-4.1-mini":  "data/gpt_generation_without_guidance/gpt_features_4_gpt-4.1-mini_freeform.csv",
    "gpt-5.1":       "data/gpt_generation_without_guidance/gpt_features5_gpt-5.1_freeform.csv",
    "gpt-5.2":       "data/gpt_generation_without_guidance/gpt_features5_gpt-5.2_freeform.csv",
}

OUT_ROOT = "results/table1"
DEVICE = "cpu"  # set from --device in main()
# Each run keeps a model plus per-seed predictions; budget generously.
MIN_FREE_GB = 5.0


def build_rows(include_gpt_only: bool) -> list[tuple[str, list[str]]]:
    """Return (row_name, extra build_train.py arguments) for every Table 1 row."""
    rows: list[tuple[str, list[str]]] = [
        ("mole_only", [
            "--feature_mode", "mole_only",
            "--mole_mode", "load", "--mole_tsv", MOLE,
            "--prestwick_screened", PRESTWICK,
        ]),
        ("rdkit_only", [
            "--feature_mode", "chemdesc_only",
            "--chemdesc_features", CHEMDESC,
        ]),
        ("mole_plus_rdkit", [
            "--feature_mode", "mole_plus_chemdesc",
            "--mole_mode", "load", "--mole_tsv", MOLE,
            "--prestwick_screened", PRESTWICK,
            "--chemdesc_features", CHEMDESC,
        ]),
    ]

    for model, csv in FREEFORM.items():
        rows.append((f"mole_plus_{model}", [
            "--feature_mode", "mole_plus_gpt",
            "--gpt_features", csv,
            "--mole_mode", "load", "--mole_tsv", MOLE,
            "--prestwick_screened", PRESTWICK,
        ]))

    if include_gpt_only:
        for model, csv in FREEFORM.items():
            rows.append((f"gpt_only_{model}", [
                "--feature_mode", "gpt_only",
                "--gpt_features", csv,
                "--prestwick_screened", PRESTWICK,
            ]))

    return rows


def command(row: str, extra: list[str], seeds: str,
            out_root: str, fixed_params: str | None,
            random_state: int | None = None) -> list[str]:
    cmd = [
        # -u keeps the child's stdout unbuffered so run.log streams while training.
        sys.executable, "-I", "-u", "notebooks/build_train.py",
        "--sf3", SF3,
        "--exact_split", SPLIT,
        "--output_dir", os.path.join(out_root, row),
        "--test_seeds", seeds,
        "--skip_intermediate_dumps",
    ]
    if random_state is not None:
        cmd += ["--random_state", str(random_state)]
    if DEVICE != "cpu":
        cmd += ["--device", DEVICE]
    if fixed_params:
        cmd += ["--fixed_params", fixed_params]
    return cmd + extra


def jobs(rows, args) -> list[tuple[str, str, list[str]]]:
    """Return (label, out_dir, command) for every child run.

    Without --trials this is one run per row, as before. With --trials each trial k
    is a complete, independent repeat of the pipeline: build_train.py is called with
    --random_state k, which seeds the hyperparameter sampler (so every row in trial k
    searches over the same k-th draw of configurations, keeping rows paired), and the
    selected configuration is refit and tested once with training seed k. Trials are
    the outer loop so an interrupted run leaves whole trials, not whole rows.
    """
    if not args.trials:
        return [(row, os.path.join(args.out_root, row),
                 command(row, extra, args.seeds, args.out_root, args.fixed_params))
                for row, extra in rows]

    out = []
    for k in (int(s) for s in args.trials.split(",") if s.strip()):
        trial_root = os.path.join(args.out_root, f"trial_{k}")
        for row, extra in rows:
            out.append((f"trial_{k}/{row}", os.path.join(trial_root, row),
                        command(row, extra, str(k), trial_root, args.fixed_params,
                                random_state=k)))
    return out


def is_complete(out_dir: str) -> bool:
    """A run is complete once build_train.py has written its summary JSON."""
    return os.path.isdir(out_dir) and any(
        f.startswith("summary_metrics_") and "_seed" not in f and f.endswith(".json")
        for f in os.listdir(out_dir))


def check_inputs() -> list[str]:
    needed = [SF3, SPLIT, MOLE, PRESTWICK, CHEMDESC, *FREEFORM.values()]
    return [p for p in needed if not os.path.exists(p)]


def free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 1024 ** 3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="0,1,2,3,4",
                    help="Training seeds for the final refits (default 0,1,2,3,4).")
    ap.add_argument("--only", default=None,
                    help="Comma-separated subset of row names to run.")
    ap.add_argument("--include-gpt-only", action="store_true",
                    help="Also run the GPT-only ablations (six extra rows).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the commands without running anything.")
    ap.add_argument("--fixed-params", default=None,
                    help="Pass through to build_train.py: 'shared' trains one configuration "
                         "across every row instead of a per-row search, cutting each row from "
                         "20 fits to len(seeds).")
    ap.add_argument("--trials", default=None,
                    help="Comma-separated trial ids, e.g. 0,1,2,3,4. Each trial repeats the full "
                         "pipeline (hyperparameter search, selection, test) with --random_state "
                         "and training seed set to the trial id, writing to "
                         "<out-root>/trial_<k>/<row>. Overrides --seeds.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="Pass through to build_train.py; 'cuda' trains on the GPU.")
    ap.add_argument("--resume", action="store_true",
                    help="Skip runs whose output directory already holds a summary JSON.")
    ap.add_argument("--out-root", default=OUT_ROOT,
                    help=f"Directory for the run outputs (default: {OUT_ROOT}).")
    ap.add_argument("--tmp-dir", default="results/table1/_tmp",
                    help="Scratch directory for child processes, kept off the system drive "
                         "(default: results/table1/_tmp, i.e. beside the outputs on D:).")
    args = ap.parse_args()
    global DEVICE
    DEVICE = args.device

    missing = check_inputs()
    if missing:
        print("Missing required input files:")
        for m in missing:
            print(f"  {m}")
        return 1

    rows = build_rows(args.include_gpt_only)
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        unknown = wanted - {r for r, _ in rows}
        if unknown:
            print(f"Unknown row name(s): {', '.join(sorted(unknown))}")
            print(f"Available: {', '.join(r for r, _ in rows)}")
            return 1
        rows = [(r, e) for r, e in rows if r in wanted]

    todo = jobs(rows, args)

    if args.dry_run:
        for label, out_dir, cmd in todo:
            done = " (complete, would skip with --resume)" if is_complete(out_dir) else ""
            print(f"# {label}{done}")
            print("  " + " ".join(cmd))
        return 0

    if free_gb() < MIN_FREE_GB:
        print(f"Only {free_gb():.1f} GB free here; need at least {MIN_FREE_GB} GB. Aborting.")
        return 1

    os.makedirs(args.out_root, exist_ok=True)

    # Keep every child process off the system drive: scratch, joblib spills and
    # pandas temporaries all land next to the outputs instead.
    tmp_dir = os.path.abspath(args.tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)
    env = {**os.environ,
           "PYTHONIOENCODING": "utf-8",
           "TMPDIR": tmp_dir, "TEMP": tmp_dir, "TMP": tmp_dir,
           "JOBLIB_TEMP_FOLDER": tmp_dir}
    print(f"child process temp: {tmp_dir} ({free_gb(tmp_dir):.1f} GB free)")

    failures = []
    for i, (label, out_dir, cmd) in enumerate(todo, 1):
        if args.resume and is_complete(out_dir):
            print(f"[{i}/{len(todo)}] {label}: already complete, skipping", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")

        print(f"[{i}/{len(todo)}] {label} -> {out_dir}", flush=True)
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - started

        status = "ok" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"      {status} in {elapsed/60:.1f} min  (log: {log_path})", flush=True)
        if proc.returncode != 0:
            failures.append(label)

    if failures:
        print(f"\n{len(failures)} run(s) failed: {', '.join(failures)}")
        return 1

    print(f"\nAll {len(todo)} runs complete. Aggregate with scripts/aggregate_table1.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
