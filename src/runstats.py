"""
Run timing and statistics for every pipeline script.

A RunTracker records wall-clock time for the whole script and, for scripts that loop
over stars, for each star. Products go to Runs/run_stats/ (inside the project tree, so
they are durable):

    Runs/run_stats/runs.csv             one row per script invocation: run_id, code
                                        fingerprint (code_sha) and settings. Written
                                        when the run STARTS, so even a killed run has one.
    Runs/run_stats/<script>_stars.csv   one row per star per run (appended immediately,
                                        so a crash mid-batch loses nothing already written)
    Runs/run_stats/<script>_chunks.csv  optional finer rows within a star (05: one per
                                        MCMC checkpoint chunk), see chunk_row()
    Runs/run_stats/run_summary.txt      one block per script invocation

Every star and chunk row carries the run_id of its runs.csv row, so "which code and
settings produced this star" is a join, not a Pipeline_Log search.

CSV rather than XLSX on purpose: a CSV row can be appended the moment a star finishes and
opens directly in Excel; an XLSX would have to be rewritten whole after every star, and a
crash during that write corrupts the entire file.

The end-of-run summary is written by an atexit hook, so it also appears after Ctrl+C,
sys.exit() and uncaught exceptions. It does NOT appear after SIGKILL (or an mpirun
SIGTERM that kills the interpreter outright) — the per-star CSV rows are the record then.

Importing this module has no side effects. Standard library only (numpy is imported
lazily by chain_acceptance, the one helper that needs it).

    tracker = runstats.RunTracker("02DataAcquisition", catalogue=sys.argv[1:],
                                  columns=["sectors", "n_points"])
    for target in target_list:
        tracker.star_start(name)
        ...
        tracker.star_end(name, "skipped", reason="no TESS data"); continue
        ...
        tracker.star_end(name, "done", sectors=3, n_points=51234)
"""
import atexit
import csv
import hashlib
import json
import os
import platform
import socket
import statistics
import sys
import time
from datetime import datetime
from typing import Optional

# Anchored to Code/ (this file's folder), not the CWD: audits are sometimes launched
# from Audits/ or a data folder, which would otherwise scatter copies of the folder.
RUN_STATS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "Runs", "run_stats")

_BASE_COLUMNS  = ["run_id", "run_started", "star", "status", "reason",
                  "star_started", "star_ended", "wall_seconds", "wall_hms"]
_CHUNK_BASE    = ["run_id", "star", "ended"]
_RUNS_COLUMNS  = ["run_id", "script", "started", "catalogue", "host", "python",
                  "code_sha", "code_files", "command", "settings_json"]


def format_hms(seconds: float) -> str:
    """Seconds → 'Dd HHh MMm', 'Hh MMm SSs' or 'MMm SSs' ('?' if not finite)."""
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "?"
    seconds = int(round(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours:02d}h {minutes:02d}m"
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    return f"{minutes}m {secs:02d}s"


def _stamp(epoch: float) -> str:
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M:%S")


def code_fingerprint(files) -> str:
    """First 12 hex digits of sha256 over the contents of `files` (sorted by path).

    A missing or unreadable file contributes its name and the word MISSING, so the
    fingerprint still changes if a dependency disappears."""
    digest = hashlib.sha256()
    for path in sorted(os.path.abspath(f) for f in files):
        digest.update(os.path.basename(path).encode())
        try:
            with open(path, "rb") as handle:
                digest.update(handle.read())
        except OSError:
            digest.update(b"MISSING")
    return digest.hexdigest()[:12]


# ── MCMC helpers (pure; testable without PHOEBE) ──────────────────────────────
def chain_acceptance(samples, start: int, stop: int, n_walkers: int):
    """(mean, min) over walkers of the fraction of steps start..stop-1 in which the
    walker MOVED (any parameter changed from the previous step).

    Same definition as Audits/audit_mcmc_chains.py. Computed from the chain because
    PHOEBE's own `acceptance_fractions` is wrong after continue_from (it seeds emcee's
    accepted COUNTS with the previous fractions). Accepts steps-first (emcee-native,
    PHOEBE's layout) or walkers-first arrays. Returns (nan, nan) if there is no
    transition to measure."""
    import numpy as np
    chain = np.asarray(samples)
    if chain.ndim != 3:
        return float("nan"), float("nan")
    if chain.shape[0] == n_walkers and chain.shape[1] != n_walkers:
        chain = np.swapaxes(chain, 0, 1)            # → (steps, walkers, params)
    start = max(1, int(start))                      # step 0 has no predecessor
    stop  = min(int(stop), chain.shape[0])
    if stop <= start:
        return float("nan"), float("nan")
    moved = np.any(chain[start:stop] != chain[start - 1:stop - 1], axis=2)
    per_walker = moved.mean(axis=0)
    return float(per_walker.mean()), float(per_walker.min())


def count_failed(failed_samples) -> tuple:
    """(total, most common message ≤120 chars) from PHOEBE's `failed_samples` dict
    (message → list of samples whose forward model failed, lnprob = −inf)."""
    per_msg = {msg: len(samples) for msg, samples in (failed_samples or {}).items()}
    total = sum(per_msg.values())
    top   = max(per_msg, key=per_msg.get)[:120] if per_msg else ""
    return total, top


def eta_seconds(s_per_step: float, steps_left: int) -> float:
    """Remaining time at a constant rate; nan if the rate is unknown."""
    if not s_per_step or s_per_step != s_per_step or steps_left <= 0:
        return 0.0 if steps_left <= 0 else float("nan")
    return float(s_per_step) * int(steps_left)


class RunTracker:
    """Wall-clock and per-star statistics for one script invocation."""

    def __init__(self, script: str, catalogue=None, columns: Optional[list] = None,
                 per_star: bool = True, enabled: bool = True,
                 out_dir: str = RUN_STATS_DIR, chunk_columns: Optional[list] = None,
                 settings: Optional[dict] = None, code_files: Optional[list] = None,
                 record_provenance: bool = True) -> None:
        self.script        = script
        self.catalogue     = " ".join(catalogue) if isinstance(catalogue, (list, tuple)) \
                             else (catalogue or "")
        self.columns       = list(columns or [])
        self.chunk_columns = list(chunk_columns or [])
        self.per_star      = per_star
        self.enabled       = enabled
        self.out_dir       = out_dir
        self.t0            = time.time()
        self.run_id        = (datetime.fromtimestamp(self.t0).strftime("%Y%m%d-%H%M%S")
                              + f"-p{os.getpid()}")
        self.rows: list = []
        self.errors: dict = {}
        self.notes: list = []
        self.settings: dict = {}
        self.code_sha = ""
        self._open_star: Optional[tuple] = None
        self._exit_cause = "normal exit"
        self._summary_written = False
        self._provenance_written = False
        self.csv_path: Optional[str] = None
        self.chunk_path: Optional[str] = None
        if not enabled:
            return
        os.makedirs(out_dir, exist_ok=True)
        if per_star:
            self.csv_path = self._resolve_csv_path("stars", _BASE_COLUMNS + self.columns)
        if self.chunk_columns:
            self.chunk_path = self._resolve_csv_path("chunks",
                                                     _CHUNK_BASE + self.chunk_columns)
        if record_provenance:
            self.record_provenance(settings, code_files)
        # Record the cause of an abnormal exit, then defer to the original hook.
        previous_hook = sys.excepthook

        def _hook(exc_type, exc, tb):
            self._exit_cause = ("interrupted (KeyboardInterrupt)"
                                if issubclass(exc_type, KeyboardInterrupt)
                                else f"crashed ({exc_type.__name__}: {exc})")
            previous_hook(exc_type, exc, tb)
        sys.excepthook = _hook
        atexit.register(self.write_summary)

    # ── provenance ───────────────────────────────────────────────────────────
    def record_provenance(self, settings: Optional[dict] = None,
                          code_files: Optional[list] = None) -> None:
        """Append this run's row to runs.csv (once). Scripts whose settings are only
        known later (05: constants defined after the tracker) pass
        record_provenance=False to the constructor and call this before the loop."""
        if not self.enabled or self._provenance_written:
            return
        self._provenance_written = True
        files = list(code_files or [sys.argv[0]])
        self.settings = dict(settings or {})
        self.code_sha = code_fingerprint(files)
        row = {"run_id": self.run_id, "script": self.script, "started": _stamp(self.t0),
               "catalogue": self.catalogue, "host": socket.gethostname(),
               "python": platform.python_version(), "code_sha": self.code_sha,
               "code_files": ";".join(os.path.basename(f) for f in files),
               "command": " ".join(sys.argv),
               "settings_json": json.dumps(self.settings, sort_keys=True, default=str)}
        path = os.path.join(self.out_dir, "runs.csv")
        try:
            new_file = not os.path.exists(path)
            with open(path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_RUNS_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as error:                  # never let bookkeeping kill a run
            print(f"  [runstats] could not write runs.csv: {error}", flush=True)

    # ── per-star ─────────────────────────────────────────────────────────────
    def _resolve_csv_path(self, kind: str, header: list) -> str:
        """<script>_<kind>.csv, or a _vN sibling if the existing header differs."""
        version = 1
        while True:
            suffix = "" if version == 1 else f"_v{version}"
            path   = os.path.join(self.out_dir, f"{self.script}_{kind}{suffix}.csv")
            if not os.path.exists(path):
                return path
            with open(path, newline="", encoding="utf-8") as handle:
                existing = next(csv.reader(handle), None)
            if existing == header:
                return path
            version += 1

    @staticmethod
    def _clean(value):
        return round(value, 4) if isinstance(value, float) else value

    def star_start(self, star: str) -> None:
        if self._open_star is not None:
            # Previous star never reported — a code path forgot star_end.
            self.star_end(self._open_star[0], "unreported",
                          reason="no star_end before next star")
        self._open_star = (star, time.time())

    def star_end(self, star: str, status: str, reason: str = "", **extra) -> float:
        """Close the open star, append its CSV row, return its wall time (s)."""
        started = (self._open_star[1] if self._open_star and self._open_star[0] == star
                   else time.time())
        self._open_star = None
        ended   = time.time()
        wall    = ended - started
        if status == "failed":
            self.count_error(reason or "failed")
        unknown = set(extra) - set(self.columns)
        if unknown:
            reason = (reason + "; " if reason else "") + \
                     f"[runstats: undeclared columns dropped: {sorted(unknown)}]"
        row = {"run_id": self.run_id, "run_started": _stamp(self.t0), "star": star,
               "status": status, "reason": reason, "star_started": _stamp(started),
               "star_ended": _stamp(ended), "wall_seconds": round(wall, 1),
               "wall_hms": format_hms(wall)}
        for column in self.columns:
            row[column] = self._clean(extra.get(column, ""))
        self.rows.append(row)
        if self.enabled and self.per_star:
            new_file = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_BASE_COLUMNS + self.columns)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
            print(f"  [runstats] {star}: {status} in {format_hms(wall)}"
                  + (f" - {reason}" if reason else ""), flush=True)
        return wall

    def done_star_seconds(self) -> list:
        """Wall times of stars completed (status 'done') so far in this run."""
        return [row["wall_seconds"] for row in self.rows if row["status"] == "done"]

    # ── per-chunk ────────────────────────────────────────────────────────────
    def chunk_row(self, star: str, **fields) -> None:
        """Append one row to <script>_chunks.csv. NEVER raises: a logging fault must
        not be able to end a multi-week batch, so any error becomes a printed warning."""
        if not (self.enabled and self.chunk_path):
            return
        try:
            row = {"run_id": self.run_id, "star": star, "ended": _stamp(time.time())}
            for column in self.chunk_columns:
                row[column] = self._clean(fields.get(column, ""))
            new_file = not os.path.exists(self.chunk_path)
            with open(self.chunk_path, "a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=_CHUNK_BASE + self.chunk_columns)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as error:
            print(f"  [runstats] chunk row not written: {error}", flush=True)

    def count_error(self, kind: str) -> None:
        self.errors[kind] = self.errors.get(kind, 0) + 1

    def note(self, text: str) -> None:
        """Free-text line added to the end-of-run summary."""
        self.notes.append(text)

    # ── end of run ───────────────────────────────────────────────────────────
    def write_summary(self) -> None:
        if self._summary_written or not self.enabled:
            return
        self._summary_written = True
        # A script that deferred provenance but died before recording it still gets
        # a runs.csv row (code fingerprint, no settings).
        self.record_provenance()
        if self._open_star is not None:
            self.star_end(self._open_star[0], "interrupted",
                          reason=f"script ended mid-star ({self._exit_cause})")
        t1      = time.time()
        lines   = [
            "=" * 72,
            f"{self.script}  |  {self._exit_cause}",
            f"  run_id    : {self.run_id}  (code_sha {self.code_sha})",
            f"  catalogue : {self.catalogue or '-'}",
            f"  started   : {_stamp(self.t0)}",
            f"  ended     : {_stamp(t1)}",
            f"  elapsed   : {format_hms(t1 - self.t0)}  ({t1 - self.t0:.0f} s)",
        ]
        if self.settings:
            lines.append("  settings  : " + ", ".join(f"{k}={v}" for k, v
                                                      in sorted(self.settings.items())))
        if self.per_star:
            counts = {}
            for row in self.rows:
                counts[row["status"]] = counts.get(row["status"], 0) + 1
            done = self.done_star_seconds()
            lines.append("  stars     : " + (", ".join(f"{n} {s}" for s, n in counts.items())
                                             or "none processed"))
            if done:
                lines.append(f"  per star  : mean {format_hms(statistics.mean(done))}, "
                             f"median {format_hms(statistics.median(done))}, "
                             f"max {format_hms(max(done))}  (completed stars only)")
            for row in self.rows:
                if row["status"] not in ("done",):
                    lines.append(f"    - {row['star']}: {row['status']}"
                                 + (f" - {row['reason']}" if row["reason"] else ""))
            if self.csv_path:
                lines.append(f"  rows      : {self.csv_path}")
            if self.chunk_path:
                lines.append(f"  chunks    : {self.chunk_path}")
        if self.errors:
            lines.append(f"  errors    : {sum(self.errors.values())}  "
                         + "; ".join(f"{k} ×{v}" for k, v in self.errors.items()))
        else:
            lines.append("  errors    : 0")
        lines.extend(f"  note      : {text}" for text in self.notes)
        block = "\n".join(lines)
        print("\n" + block, flush=True)
        with open(os.path.join(self.out_dir, "run_summary.txt"), "a",
                  encoding="utf-8") as handle:
            handle.write(block + "\n")
