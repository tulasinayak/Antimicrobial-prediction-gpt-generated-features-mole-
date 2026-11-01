"""Assemble revision_outputs/results.md from the section files and zip the folder."""

from __future__ import annotations

import os
import shutil

from common import OUT, SECTIONS

ORDER = ["summary", "task1", "task2", "task3", "task4", "task5", "task6", "task7", "task8",
         "tables"]


def main() -> None:
    parts = []
    for name in ORDER:
        with open(f"{SECTIONS}/{name}.md", encoding="utf-8") as fh:
            parts.append(fh.read().rstrip())
    with open(f"{OUT}/results.md", "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(parts) + "\n")

    shutil.rmtree(f"{OUT}/scripts/__pycache__", ignore_errors=True)
    base = os.path.abspath(OUT)
    archive = shutil.make_archive(base, "zip", root_dir=os.path.dirname(base), base_dir=os.path.basename(base))
    print(f"wrote {OUT}/results.md and {archive}")


if __name__ == "__main__":
    main()
