#!/usr/bin/env bash
# run_05_autorestart.sh — run Stage 05 under mpirun and, if it dies, restart it from
# its last checkpoint. Written for the libphoebe radiosity segfault (Pipeline_Log
# Entry 75), which kills the whole MPI job with no Python traceback.
#
# Usage (WSL, from anywhere):
#   bash run_05_autorestart.sh Targets/neglected_targets_batch6.csv Runs/batch6_production_05.log \
#        "BATCH 6 (first detached batch; ROCHE_LOBE_CAP=1.0)"
#
# Paths are relative to Code/ (the script cd's there). Output is appended to the log
# (tee -a) after a marker line per attempt, as the CLAUDE.md launch-hygiene rule asks,
# so no separate printf marker is needed. The optional third argument is the label:
#   ########## <label> — <date> ##########                          (launch)
#   ########## <label> [auto-restart n/N] — <date> ##########      (each restart)
# Without a label the marker reads "RUN 05 (<catalogue>)".
#
# Restarting is safe because 05 is idempotent: a finished star is skipped (its results
# file exists) and an MCMC star resumes from its last 100-step checkpoint. A restart
# loses at most the chunk in progress. Every attempt is a new run_id in runs.csv.
#
# It stops, without restarting, when:
#   * 05 exits 0;
#   * you press Ctrl-C, or send SIGTERM to this script;
#   * MAX_RESTARTS restarts have been used (default 3).
# A deterministic Python error would also be retried up to MAX_RESTARTS times, then stop.
#
# Settings (environment variables): MAX_RESTARTS=3, PAUSE_S=60 (wait before a restart,
# so MPI can clean up and you can cancel), NPROCS=8 (Entry 80; was 14).
# RUN_CMD replaces the mpirun command; it exists only to test this wrapper.
#
# Dynamic -n mode (Pipeline_Log Entry 78). One star can use at most 8 processes
# (emcee evaluates 7 of the 14 walkers at a time), so NPROCS=8 is the efficient size.
# Two queues may run at once — even from the SAME catalogue — because Script 05 takes
# a per-star lock (data/bundles/bundle_<star>.lock) and each queue skips stars another
# live queue holds. Starting a second queue is deliberate, so it must be opted into:
#   NPROCS=8 bash run_05_autorestart.sh <cat> Runs/<log_A>.log "<label A>"
#   NPROCS=8 ALLOW_CONCURRENT=1 bash run_05_autorestart.sh <cat> Runs/<log_B>.log "<label B>"
# Use a separate log per queue. Stop a queue with Ctrl-C; its star resumes later from
# its checkpoint (the lock of a dead queue is taken over automatically).
# Never run a pre-Entry-78 05 (no locks) alongside a queue: it would not see them.
set -u

CATALOGUE=${1:?usage: run_05_autorestart.sh <catalogue.csv> <log> [label]}
LOG=${2:?usage: run_05_autorestart.sh <catalogue.csv> <log> [label]}
LABEL=${3:-RUN 05 ($CATALOGUE)}
MAX_RESTARTS=${MAX_RESTARTS:-3}
PAUSE_S=${PAUSE_S:-60}
NPROCS=${NPROCS:-8}   # Entry 80: -n 8 measured ~1.5x faster per star than -n 14

cd "$(dirname "$0")" || exit 2

if pgrep -f "Optimization.py" >/dev/null && [ "${ALLOW_CONCURRENT:-0}" != "1" ]; then
    echo "Optimization.py is already running; not starting a second copy." >&2
    echo "(For a deliberate second queue, set ALLOW_CONCURRENT=1 — see the header.)" >&2
    exit 2
fi
# One wrapper per log: an exclusive flock on <log>.lock, held for the wrapper's whole
# life and released by the kernel when it exits (so it can never go stale).
exec 9>"${LOG}.lock"
if ! flock -n 9; then
    echo "Another wrapper is already writing to ${LOG}; use a separate log per queue." >&2
    exit 2
fi

stop=0
trap 'stop=1' INT TERM

note() {
    printf '[autorestart %s] %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"
}

run_05() {
    if [ -n "${RUN_CMD:-}" ]; then
        bash -c "$RUN_CMD"
    else
        mpirun --use-hwthread-cpus -n "$NPROCS" python3 -u Optimization.py "$CATALOGUE"
    fi
}

attempt=0
while :; do
    if [ "$attempt" -eq 0 ]; then tag=""; else tag=" [auto-restart $attempt/$MAX_RESTARTS]"; fi
    printf '\n\n########## %s%s — %s ##########\n' \
        "$LABEL" "$tag" "$(date '+%F %T')" >> "$LOG"

    run_05 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}

    if [ "$rc" -eq 0 ]; then
        note "05 finished (exit 0)."
        exit 0
    fi
    if [ "$stop" -eq 1 ]; then
        note "05 exited with code $rc after Ctrl-C/SIGTERM; not restarting."
        exit "$rc"
    fi
    if [ "$attempt" -ge "$MAX_RESTARTS" ]; then
        note "05 exited with code $rc; restart limit ($MAX_RESTARTS) reached; giving up."
        exit "$rc"
    fi

    attempt=$((attempt + 1))
    note "05 exited with code $rc; restarting from the last checkpoint in ${PAUSE_S}s (restart $attempt/$MAX_RESTARTS). Ctrl-C to cancel."
    sleep "$PAUSE_S" &
    wait $!   # unlike a foreground sleep, wait returns as soon as Ctrl-C/SIGTERM arrives
    if [ "$stop" -eq 1 ]; then
        kill $! 2>/dev/null
        note "Cancelled during the pause; not restarting."
        exit "$rc"
    fi
done
