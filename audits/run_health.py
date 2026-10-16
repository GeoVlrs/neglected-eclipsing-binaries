"""
================================================================================
RUN HEALTH BOARD — WHERE IS EVERY QUEUE STAR, AND DOES ANYTHING NEED ACTION?
================================================================================
Summary:
    One row per star of a Stage-05 catalogue (Pipeline_Log Entry 85):

      * STATE: done / kept / running(A|B) / paused / parked / pending /
        pending-refit (results from an older bundle).
      * STEPS, s/it, acceptance and failed-model fraction of its latest chunk,
        with the most common failure message (Runs/run_stats/05Optimization_chunks.csv).
      * NOTE: the park reason and its fix (data/results/parked_<star>.txt), or
        the latest early-warning codes (Runs/run_stats/05Optimization_health.csv).

    Read-only and light: parses CSV, text and log files; loads no bundle, runs
    no model, so it is safe beside a live MCMC.

Usage (from Code/, Windows or WSL python, seconds):
    python Audits/run_health.py Targets/phaseE_queue.csv
    python Audits/run_health.py Targets/phaseE_queue.csv --all-warnings

Exit code: 1 if any star is parked (action needed), else 0.
================================================================================
"""
import csv
import glob
import hashlib
import os
import re
import sys
import time

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR  = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
sys.path.insert(0, AUDIT_DIR)
import paths                                              # noqa: E402
import earlywarn                                          # noqa: E402
from morphology import resolve_morphology_kind            # noqa: E402
from audit_results_ledger import keep_list_from_05        # noqa: E402

STATS_DIR   = os.path.join(CODE_DIR, "Runs", "run_stats")
CHUNKS_CSV  = os.path.join(STATS_DIR, "05Optimization_chunks.csv")
HEALTH_CSV  = os.path.join(STATS_DIR, "05Optimization_health.csv")
QUEUE_LOGS  = sorted(glob.glob(os.path.join(CODE_DIR, "Runs", "phaseE_queue*.log")))
FRESH_S     = 6 * 3600        # a queue log written within this is treated as live


def sha12(path: str) -> str:
    """First 12 hex digits of sha256 (Script 05's file_sha)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:12]
    except OSError:
        return ""


def results_sha(path: str) -> str:
    """input_bundle_sha provenance row of a results CSV ('' if absent)."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row.get("Symbol") == "input_bundle_sha":
                    return row.get("Value", "")
    except OSError:
        pass
    return ""


def read_rows(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def live_stars() -> dict:
    """{star: queue label} for queue logs written recently whose last marker has
    not been followed by a wrapper exit note."""
    out = {}
    for log in QUEUE_LOGS:
        if time.time() - os.path.getmtime(log) > FRESH_S:
            continue
        with open(log, encoding="utf-8", errors="replace") as handle:
            text = handle.read()[-2_000_000:]
        stars = re.findall(r"OPTIMISATION: (.+?)  \(", text)
        tail  = text[text.rfind("OPTIMISATION: "):] if stars else ""
        if stars and "not restarting" not in tail and "giving up" not in tail:
            label = re.sub(r"^phaseE_queue|\.log$", "", os.path.basename(log)) or "?"
            out[stars[-1]] = label
    return out


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    catalogue  = argv[1]
    all_warn   = "--all-warnings" in argv
    targets    = read_rows(catalogue)
    keep       = keep_list_from_05()
    chunks     = read_rows(CHUNKS_CSV)
    health     = read_rows(HEALTH_CSV)
    live       = live_stars()

    rows, n_parked = [], 0
    for i, target in enumerate(targets, 1):
        star, safe = target["name"], target["name"].replace(" ", "_")
        kind    = resolve_morphology_kind(star, target.get("type", ""))
        bundle  = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe}.phoebe")
        b_sha   = sha12(bundle)
        res     = os.path.join(paths.RESULTS_DIR, f"results_{safe}.csv")
        ckpt    = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe}_mcmc.phoebe")
        park    = earlywarn.read_parked(earlywarn.parked_path(paths.RESULTS_DIR, safe))
        mine    = [c for c in chunks if c.get("star") == star]
        last    = mine[-1] if mine else {}
        note    = ""

        if os.path.exists(res) and star in keep:
            state = "kept"
        elif os.path.exists(res) and results_sha(res) == b_sha and b_sha:
            state = "done"
        elif park and park.get("input_bundle_sha") == b_sha:
            state, n_parked = "PARKED", n_parked + 1
            note = f"{park.get('stage', '?')}: {park.get('reason', '?')} — see parked_{safe}.txt"
        elif star in live:
            state = f"running({live[star]})"
        elif os.path.exists(ckpt):
            try:
                started = open(ckpt + ".input_sha", encoding="utf-8").read().strip()
            except OSError:
                started = ""
            state = "paused" if started == b_sha else "pending(stale ckpt)"
        elif os.path.exists(res):
            state = "pending-refit"
        else:
            state = "pending"

        events = [h for h in health if h.get("star") == star]
        if events and not note:
            recent = events if all_warn else [e for e in events
                                               if e.get("run_id") == events[-1].get("run_id")]
            codes = sorted({f"{e['level']}:{e['code']}" for e in recent})
            note = ", ".join(codes)

        # Chain statistics only for a chain that is current (running / paused / parked).
        current = state.startswith(("running", "paused", "PARKED"))
        steps = last.get("step_to", "") if current else ""
        spi   = f"{float(last['s_per_iter']):.0f}" if steps and last.get("s_per_iter") else ""
        acc   = f"{float(last['accept_mean']):.2f}" if steps and last.get("accept_mean") else ""
        fail  = (f"{float(last['failed_frac_chunk']):.2f} {last.get('failed_top', '')[:18]}"
                 if steps and last.get("failed_frac_chunk") else "")
        model = {"contact": "contact", "detached": "detached",
                 "semidetached": "semi-det"}.get(kind, kind)
        rows.append((i, star, model, state, steps, spi, acc, fail, note))

    head = ("#", "star", "model", "state", "steps", "s/it", "acc", "failed (top)", "note")
    widths = [max(len(str(r[k])) for r in rows + [head]) for k in range(len(head))]
    widths[-1] = min(widths[-1], 70)
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*head))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt.format(*[str(x)[:widths[k]] for k, x in enumerate(r)]))
    counts = {}
    for r in rows:
        key = re.sub(r"\(.*\)", "", r[3])
        counts[key] = counts.get(key, 0) + 1
    print("\n" + " · ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    if n_parked:
        print(f"{n_parked} PARKED — action needed: read data/results/parked_<star>.txt "
              "(reason + fix), rebuild with Script 04 or set IGNORE_PARK.")
    return 1 if n_parked else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
