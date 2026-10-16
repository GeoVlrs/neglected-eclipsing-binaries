#!/usr/bin/env bash
# smoke_05.sh — short end-to-end run of Script 05 on a SCRATCH copy of a bundle,
# required after any change to Optimization.py / earlywarn.py / morphology.py
# before a production queue is relaunched (Pipeline_Log Entry 85; HOW_TO §10).
#
#   bash smoke_05.sh <bundle.phoebe> <gcvs_type> [label] [scratch name]
#   e.g. bash smoke_05.sh Runs/otcep_sd/bundle_OT_Cep_detached_20261009.phoebe EB
#        bash smoke_05.sh data/bundles/bundle_V358_Cep.phoebe EB/SD
#        bash smoke_05.sh data/bundles/bundle_RZ_Dra.phoebe EB/SD rzdra "ZZ Smoke R"
#
# The bundle is copied to a scratch star (default "ZZ Smoke"; the name must start
# with "ZZ "); the GCVS type decides its routing (EW → contact, EB → detached,
# EB/SD → semi-detached). A tiny chain runs and the log's multi-start and
# early-warning lines are printed. Nothing of a real star is opened for writing.
# Several smoke runs can go in parallel if each has its own scratch name.
#
# COST: ~20-40 min at NPROCS=8 beside a live queue. Ask before starting it.
#
# Settings (environment; defaults in brackets):
#   NPROCS [8]
#   MCMC_NITERS [30], MCMC_CHECKPOINT [15], MCMC_BURNIN [10]
#   NM_MAXITER [20], NM_SCOUT_MAXITER [script default, capped at NM_MAXITER]
#   SIGMA_PRESCALE [script default 1; 0 = sample against the raw uncertainties]
#   IGNORE_PARK [the scratch name, so the sampler path is always exercised;
#                set IGNORE_PARK= to test parking]
#   SCRIPT [Optimization.py]  — a staged copy can be tested before it is swapped in
#   Q_FROM [unset] — a contact star's name: its data/qsearch/q_adopted_<star>.txt is
#                copied to the scratch star, which Script 05 requires for contact stars
set -u
cd "$(dirname "$0")" || exit 2
SRC=${1:?usage: smoke_05.sh <bundle.phoebe> <gcvs_type> [label] [scratch name]}
TYPE=${2:?usage: smoke_05.sh <bundle.phoebe> <gcvs_type> [label] [scratch name]}
LABEL=${3:-smoke}
NAME=${4:-ZZ Smoke}
SCRIPT=${SCRIPT:-Optimization.py}
[ -f "$SRC" ] || { echo "no such bundle: $SRC"; exit 2; }
[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT"; exit 2; }
case "$NAME" in "ZZ "*) ;; *) echo "scratch name must start with 'ZZ ': $NAME"; exit 2;; esac

SAFE=${NAME// /_}
DIR=Runs/smoke
STAMP=$(date '+%Y%m%d-%H%M%S')
TAG=${LABEL//[^A-Za-z0-9_-]/_}
LOG="$DIR/smoke_${STAMP}_${TAG}.log"
CSV="$DIR/smoke_${TAG}.csv"
mkdir -p "$DIR"

# Clear ONLY this scratch star's previous products (moved aside, never deleted).
for f in data/bundles/bundle_${SAFE}.phoebe data/bundles/bundle_${SAFE}_mcmc.phoebe \
         data/bundles/bundle_${SAFE}_mcmc.phoebe.input_sha data/bundles/bundle_${SAFE}_mcmc.phoebe.nm_start \
         data/bundles/bundle_${SAFE}_final.phoebe data/bundles/bundle_${SAFE}.lock \
         data/results/results_${SAFE}.csv data/results/parked_${SAFE}.txt; do
    [ -e "$f" ] && mv "$f" "$DIR/$(basename "$f").prev_${STAMP}"
done
cp "$SRC" "data/bundles/bundle_${SAFE}.phoebe"
QFILE="data/qsearch/q_adopted_${SAFE}.txt"
[ -e "$QFILE" ] && mv "$QFILE" "$DIR/q_adopted_${SAFE}.txt.prev_${STAMP}"
if [ -n "${Q_FROM:-}" ]; then
    QSRC="data/qsearch/q_adopted_${Q_FROM// /_}.txt"
    [ -f "$QSRC" ] || { echo "no adopted-q file: $QSRC"; exit 2; }
    cp "$QSRC" "$QFILE"
fi
printf 'name,type,period,sectors,tic\n%s,%s,0.5,0,0\n' "$NAME" "$TYPE" > "$CSV"

export MCMC_NITERS=${MCMC_NITERS:-30} MCMC_CHECKPOINT=${MCMC_CHECKPOINT:-15}
export MCMC_BURNIN=${MCMC_BURNIN:-10} NM_MAXITER=${NM_MAXITER:-20}
export IGNORE_PARK=${IGNORE_PARK-$NAME}
X="-x MCMC_NITERS -x MCMC_CHECKPOINT -x MCMC_BURNIN -x NM_MAXITER -x IGNORE_PARK"
[ -n "${NM_SCOUT_MAXITER:-}" ] && { export NM_SCOUT_MAXITER; X="$X -x NM_SCOUT_MAXITER"; }
[ -n "${SIGMA_PRESCALE:-}" ] && { export SIGMA_PRESCALE; X="$X -x SIGMA_PRESCALE"; }
echo "[smoke] $SRC as $NAME ($TYPE) with $SCRIPT, NM_MAXITER=$NM_MAXITER, MCMC_NITERS=$MCMC_NITERS, IGNORE_PARK='${IGNORE_PARK}', log $LOG"
mpirun --use-hwthread-cpus -n "${NPROCS:-8}" $X python3 -u "$SCRIPT" "$CSV" > "$LOG" 2>&1
rc=$?
echo "[smoke] exit $rc"
grep -aE "\[(preflight|post-NM|health|parked|morphology|multi-start|prior)|Nelder-Mead|PARK|OVERRIDDEN|Traceback|Error:|chunk statistics unavailable|check unavailable|MCMC sampling complete|Results saved|Final reduced|Reduced χ² at" "$LOG" \
    | grep -av "WARNING" | cut -c1-220
exit $rc
