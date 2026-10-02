#!/usr/bin/env python3
"""
sync_from_working_tree.py — publish-mirror updater for the TESS binary modelling pipeline.

Copies the CODE + DOCS of the working tree (the numbered scripts, the audit
scripts, the notes, and the sample-definition CSVs) into this repository's
published `src/`, `audits/`, `docs/`, and `data/targets/` layout, so a maintainer
can then review and `git add / commit / push`.

It DELIBERATELY never copies the private or heavy artefacts — the running
Pipeline_Log.txt, per-target results_*.csv, run logs, PHOEBE bundles (*.phoebe),
or FITS light curves (*.fits). Those stay on the working machine.

Run it from anywhere; all paths are resolved relative to this file:

    python tools/sync_from_working_tree.py

Layout assumption (post-2026-09 reorg — every script lives once in Code/):
    <CODE>/neglected-eclipsing-binaries/tools/sync_from_working_tree.py   (this file)
    <CODE>/paths.py, 01TargetSelection.py ... 06Visualization.py,
           GenerateThesisTables.py, SchematicRocheGeometry.py
    <CODE>/Audits/*.py, Notes/*.txt, Targets/neglected_targets*.csv
    <CODE>/data/  (products, typed; never published)
"""
from __future__ import annotations

import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent   # neglected-eclipsing-binaries/
CODE = REPO.parent                               # Code/
# After the 2026-09 folder reorg, every script (and the shared paths.py) lives once in
# Code/; products moved to Code/data/<type>/ and are never published (guards below).

# Working scripts -> published names under src/. Numbered stages drop their number;
# paths.py keeps its name (every stage does `import paths`).
SCRIPTS: list[tuple[Path, Path]] = [
    (CODE / "paths.py",                REPO / "src" / "paths.py"),
    # Run timing / provenance / statistics; every stage and audit does `import runstats`.
    (CODE / "runstats.py",             REPO / "src" / "runstats.py"),
    (CODE / "01TargetSelection.py",    REPO / "src" / "TargetSelection.py"),
    (CODE / "02DataAcquisition.py",    REPO / "src" / "DataAcquisition.py"),
    (CODE / "03OCAnalysis.py",         REPO / "src" / "OCAnalysis.py"),
    (CODE / "04Modeling.py",           REPO / "src" / "Modeling.py"),
    (CODE / "05Optimization.py",       REPO / "src" / "Optimization.py"),
    (CODE / "06Visualization.py",      REPO / "src" / "Visualization.py"),
    (CODE / "GenerateThesisTables.py", REPO / "src" / "GenerateThesisTables.py"),
    # Standalone illustration utility, not a numbered pipeline stage.
    (CODE / "SchematicRocheGeometry.py", REPO / "src" / "SchematicRocheGeometry.py"),
]

# (source_dir, glob_pattern, destination_dir)
GLOBS: list[tuple[Path, str, Path]] = [
    (CODE / "Audits",  "*.py",                    REPO / "audits"),
    (CODE / "Notes",   "*.txt",                   REPO / "docs"),
    (CODE / "Targets", "neglected_targets*.csv",  REPO / "data" / "targets"),
    (CODE / "Targets", "triage_*.csv",            REPO / "data" / "targets"),
]

# Hard guard: never publish these, whatever a pattern might match.
_FORBIDDEN_NAMES = {"Pipeline_Log.txt"}
_FORBIDDEN_SUFFIXES = {".phoebe", ".fits"}
_FORBIDDEN_PREFIXES = ("results_",)


def _is_private(p: Path) -> bool:
    return (
        p.name in _FORBIDDEN_NAMES
        or p.suffix in _FORBIDDEN_SUFFIXES
        or p.name.startswith(_FORBIDDEN_PREFIXES)
    )


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  SKIP (missing): {src}")
        return
    if _is_private(src):
        print(f"  REFUSED (private): {src.name}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  {src.name}  ->  {dst.relative_to(REPO)}")


def main() -> int:
    print(f"Syncing working tree -> repo\n  CODE: {CODE}\n  REPO: {REPO}\n")
    print("Scripts:")
    for src, dst in SCRIPTS:
        _copy(src, dst)
    for src_dir, pattern, dst_dir in GLOBS:
        print(f"\n{src_dir.name}/{pattern}:")
        for src in sorted(src_dir.glob(pattern)):
            _copy(src, dst_dir / src.name)
    print("\nDone. Review `git status` / `git diff`, then commit & push.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
