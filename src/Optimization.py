"""
================================================================================
SCRIPT 05 — MCMC POSTERIOR OPTIMISATION (PHOEBE 2 + emcee)
================================================================================
Summary:
    Solves the astrophysical inverse problem by mapping the joint posterior
    probability distribution over the free geometric and thermodynamic
    parameters of each binary system. The optimisation proceeds in two stages:

        Stage 1 — Nelder-Mead pre-wash:
            A deterministic simplex optimiser performs ~200 iterations to locate
            the χ² valley before MCMC begins. This prevents walkers from spending
            burn-in cycles in unphysical parameter regions and reduces the risk of
            early Roche-lobe overflow crashes.

        Stage 2 — emcee MCMC sampling:
            The affine-invariant ensemble sampler distributes MCMC_NWALKERS
            parallel Markov chains across the parameter space. Because walkers
            share position information through stretch moves, the sampler is
            efficient across correlated, elongated posterior distributions — a
            critical property for binary star inverse problems where inclination,
            radius sum, and mass ratio are strongly degenerate in single-band
            photometry.

    Convergence is assessed via the integrated autocorrelation time τ and the
    Gelman-Rubin R̂ statistic. A chain is declared converged when its length
    exceeds 50τ and R̂ < 1.1 for all parameters.

    The final best-fit model is computed with Wilson irradiation for contact
    (EW) systems, capturing the strong mutual irradiation between overcontact
    components that drives out-of-eclipse flux curvature.

    MPI parallelisation distributes each per-walker PHOEBE 2 forward model
    (the computational bottleneck) across available CPU threads. All file I/O
    is performed exclusively by MPI rank 0 to prevent write collisions.

Inputs:
    neglected_targets.csv     — Target list from Script 01.
    bundle_<star>.phoebe      — Initialised bundles from Script 04.

Outputs:
    results_<star>.csv            — Derived physical parameters with uncertainties.
    bundle_<star>_final.phoebe    — Bundle with converged posterior distributions.
    plots/phoebe_final_<star>.png — Best-fit model vs. TESS data plot.

Dependencies:
    pip install phoebe emcee numpy matplotlib mpi4py

Run with MPI:
    mpirun --use-hwthread-cpus -n 14 python3 05Optimization.py

References:
    Foreman-Mackey et al. (2013), PASP 125, 306 — emcee MCMC sampler.
    Conroy et al. (2020), ApJS 250, 34 — PHOEBE 2 inverse-problem interface.
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import os
import sys
import warnings
from typing import Optional

# Force PHOEBE to use only locally-installed passbands. On an offline/firewalled
# host (e.g. WSL) PHOEBE otherwise tries to reach tables.phoebe-project.org for
# passband update checks on essentially every forward model, and each attempt
# blocks ~30–90 s × 3 retries — dominating the MCMC wall-clock (observed ~250 s
# per step, almost all network timeout). TESS:T is already installed locally, so
# disabling the online checks is safe. MUST be set before `import phoebe`.
# (Override with PHOEBE_ENABLE_ONLINE_PASSBANDS=TRUE if a new passband is needed.)
os.environ.setdefault("PHOEBE_ENABLE_ONLINE_PASSBANDS", "FALSE")

# Semi-detached stars (Entry 77): PHOEBE's 'semidetached' constraint sets the
# secondary's volume EXACTLY to its critical (Roche-lobe) value; floating-point
# rounding then lands it ~1e-16 above, and backend/universe.py clamps it back with a
# WARNING on its "UNIVERSE" logger — on every mesh of every model, from every MPI
# rank (~100 lines/min, tens of millions per star). PHOEBE only clamps when the
# excess is < 1e-10 relative and raises an error otherwise, so the message carries no
# information. Show it once per process, then drop it. MUST be installed before
# `import phoebe`: worker ranks stay inside that import and never run later code.
import logging


class _OnceNumericsVolumeWarning(logging.Filter):
    """Pass the 'target_volume ... slightly over critical value' warning once."""
    _seen = False

    def filter(self, record: logging.LogRecord) -> bool:
        if "slightly over critical value, likely due to numerics" not in record.getMessage():
            return True
        if _OnceNumericsVolumeWarning._seen:
            return False
        _OnceNumericsVolumeWarning._seen = True
        record.msg = (str(record.msg) + "  [further identical clamps suppressed "
                      "in this process — semi-detached lobe constraint, Entry 80]")
        return True


logging.getLogger("UNIVERSE").addFilter(_OnceNumericsVolumeWarning())

# ── Scientific libraries ──────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phoebe
import emcee

# ── MPI parallelisation ───────────────────────────────────────────────────────
# Each PHOEBE 2 forward model (one per MCMC walker per step) is dispatched to
# a separate MPI worker. The master rank (0) orchestrates the sampler while
# worker ranks execute run_compute() calls in parallel.
try:
    from mpi4py import MPI
    mpi_comm      = MPI.COMM_WORLD
    mpi_rank      = mpi_comm.Get_rank()
    mpi_n_procs   = mpi_comm.Get_size()
except ImportError:
    mpi_rank    = 0
    mpi_n_procs = 1


def master_print(*args, **kwargs) -> None:
    """Print only from MPI rank 0 to avoid garbled interleaved output."""
    if mpi_rank == 0:
        print(*args, **kwargs)


if mpi_rank == 0:
    master_print(f"\n[MPI] Cluster initialised: {mpi_n_procs} processor core(s).")

warnings.filterwarnings("ignore")
phoebe.logger(clevel="WARNING")

import paths
paths.ensure_dirs()

# Run timing / statistics (Runs/run_stats/). Created AFTER `import phoebe`: worker ranks
# are held inside that import, so only rank 0 ever reaches this line; `enabled` guards it
# anyway. Columns are declared up front so every row of the CSV lines up.
import statistics
import time
import runstats
run_tracker = runstats.RunTracker(
    "05Optimization", catalogue=sys.argv[1:], enabled=(mpi_rank == 0),
    columns=["setup_seconds", "nm_seconds", "resumed_from", "steps_this_session",
             "final_steps", "n_chunks", "mcmc_seconds", "s_per_iter", "chunk_first_s",
             "chunk_min_s", "chunk_max_s", "chunk_last_s", "stop_reason", "chi2_red",
             "err_scale", "median_accept", "min_accept", "failed_models", "failed_frac",
             "failed_top_reason", "verdicts", "morphology_flag"],
    # One row per checkpoint chunk (5F) → Runs/run_stats/05Optimization_chunks.csv.
    chunk_columns=["step_from", "step_to", "steps", "chunk_seconds", "s_per_iter",
                   "accept_mean", "accept_min", "failed_new", "failed_cum",
                   "failed_frac_chunk", "failed_top", "eta_star_s", "eta_batch_s"],
    # Settings are module constants defined further down; provenance is recorded
    # just before the star loop (run_tracker.record_provenance).
    record_provenance=False,
)
# Set by prepare_bundle_for_mcmc (5E) so the per-star row can report the NM duration.
last_nm_seconds: float = float("nan")

# Per-walker chain diagnostics (Section 5I-b). Audits/ has no __init__.py, so it
# is imported by path rather than as a package. Safe to import here: its
# matplotlib.use("Agg") is a no-op (already set above, before phoebe), and its
# `import phoebe` hits the module cache, so no MPI re-detection. Never import in
# the other direction — this script has no __main__ guard and would re-run.
# Degrades to no diagnostics rather than aborting a batch if it cannot be loaded.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audits"))
try:
    import audit_mcmc_chains as chain_audit
except Exception as chain_audit_error:      # pragma: no cover — defensive
    chain_audit = None
    master_print(f"[5I-b] chain diagnostics unavailable: {chain_audit_error}")

# Temperature boundary for von Zeipel vs. Lucy bolometric coefficients.
RADIATIVE_CONVECTIVE_BOUNDARY_K: float = 8_000.0


# =============================================================================
# SECTION 1 — CONFIGURATION
# =============================================================================

# Optional positional argument selects the target catalogue, so a single batch
# can be optimised in isolation, e.g.:
#   mpirun --use-hwthread-cpus -n 14 python3 -u 05Optimization.py \
#       Targets/neglected_targets_batch1.csv 2>&1 | tee optimization_log.txt
INPUT_CATALOG: str = sys.argv[1] if len(sys.argv) > 1 else "neglected_targets.csv"

# MCMC sampler settings — production defaults, overridable via environment
# variables for fast trial/smoke runs without editing this file, e.g.:
#   MCMC_NITERS=45 MCMC_CHECKPOINT=15 MCMC_BURNIN=10 NM_MAXITER=20 \
#       python3 05Optimization.py trial_05.csv
# Under mpirun, export the vars and forward them: `export MCMC_NITERS=45; ...;
#   mpirun -x MCMC_NITERS -x MCMC_CHECKPOINT -x MCMC_BURNIN -x NM_MAXITER ...`.
MCMC_N_WALKERS: int = int(os.environ.get("MCMC_NWALKERS", 14))
MCMC_N_ITERATIONS: int = int(os.environ.get("MCMC_NITERS", 2850))   # fixed-mode target / fallback
MCMC_BURN_IN_STEPS: int = int(os.environ.get("MCMC_BURNIN", 350))   # steps discarded before posterior

# Nelder-Mead pre-wash iteration cap (Stage 1). Lower it for trial runs.
NM_MAXITER: int = int(os.environ.get("NM_MAXITER", 200))

# Checkpoint cadence: the MCMC runs in chunks of this many iterations, saving the
# bundle (chains included) to disk after each chunk. This lets a long run be
# stopped — even hard-killed — and resumed mid-chain from the last checkpoint,
# which is essential for sharing a home desktop with the optimisation.
MCMC_CHECKPOINT_NITERS: int = int(os.environ.get("MCMC_CHECKPOINT", 100))

# Convergence criterion: chain must be longer than this multiple of τ.
CONVERGENCE_MULTIPLIER: int = 50

# Gelman-Rubin R̂ threshold below which a parameter is considered converged.
RHAT_CONVERGENCE_THRESHOLD: float = 1.1

# --- Adaptive convergence gate (v1) -----------------------------------------
# Instead of a fixed step budget, the sampler runs in MCMC_CHECKPOINT_NITERS-sized
# chunks between a floor and a ceiling and stops as soon as EVERY sampled parameter
# satisfies both the 50τ autocorrelation criterion (Sokal 1997; Goodman & Weare
# 2010; Foreman-Mackey et al. 2013) and R̂ < 1.1 (Gelman & Rubin 1992; Brooks &
# Gelman 1998). The floor keeps the τ/R̂ estimates stable before the first test;
# the ceiling bounds the worst case (multimodal / prior-railed fill-out, which no
# number of steps resolves). Setting MCMC_NITERS explicitly forces the legacy
# fixed-length run (MIN = MAX = MCMC_NITERS) for trials and reproductions.
MCMC_FIXED_MODE: bool = "MCMC_NITERS" in os.environ
MCMC_MIN_NITERS: int = (MCMC_N_ITERATIONS if MCMC_FIXED_MODE
                        else int(os.environ.get("MCMC_MIN_NITERS", 1500)))
MCMC_MAX_NITERS: int = (MCMC_N_ITERATIONS if MCMC_FIXED_MODE
                        else int(os.environ.get("MCMC_MAX_NITERS", 6000)))

# Lower bound on the contact fill-out factor f. The overcontact neck pinches toward the
# L1 point as f → 0, where PHOEBE's mesh destabilises (walkers proposing low/negative f
# gave "fillout_factor not within limits" + "Projections failing / Mesh failed",
# lnprob=-inf). A positive floor keeps walkers in valid contact space. Value pinned by a
# self-consistent forward-model probe at the production mesh resolution (ntriangles=2000):
# f ≤ 0.02 FAILS, f = 0.03 is stable. (ntriangles=5000 would lower the stable floor to
# 0.02 but costs ~6–7× per model — not worth it for MCMC.) 0.03 stays below the lowest
# genuine observed fill-out (TZ Boo f=0.039), so no real W UMa system is truncated.
# Empirical, hence overridable.
FILLOUT_FLOOR: float = float(os.environ.get("FILLOUT_FLOOR", 0.03))

# Upper bound on a DETACHED component's radius, as a fraction of its Eggleton
# (1983) Roche lobe. Was 0.9 on the assumption that approaching the lobe would
# destabilise the mesh. A probe at production resolution (Pipeline_Log Entry 71)
# showed otherwise: the model computes cleanly at exactly 1.00 x the lobe, and
# only an OVERFLOWING configuration (1.02) fails — and it fails on PHOEBE's own
# geometry check, not on the mesh. PHOEBE therefore enforces the physical limit
# itself, so the prior does not need to and 0.9 only excluded real geometry.
# This matters because four remaining targets (RZ Dra, V1034 Cyg, GW Gem,
# V404 Lyr) are classified semi-detached, where a component FILLS its lobe by
# definition — unreachable under any cap below 1.0.
ROCHE_LOBE_CAP: float = float(os.environ.get("ROCHE_LOBE_CAP", 1.0))


# =============================================================================
# SECTION 2 — FREE PARAMETER DEFINITIONS WITH PRIOR BOUNDS
# =============================================================================
# Format: "phoebe_twig": (lower_bound, upper_bound, "human description")
#
# The mass ratio q is held fixed for contact (EW) systems because it is
# notoriously degenerate with inclination and fillout factor in single-band
# photometry. This must be stated explicitly in the thesis methodology.
#
# For EB (detached) systems the inclination floor is set to 60° and radii
# are capped at 1.0 to prevent ghost-star collapse (requiv → 0 or → overflow).

FREE_PARAMS_CONTACT: dict[str, tuple[float, float, str]] = {
    "teff@secondary":                   (3000, 10000, "Secondary effective temperature (K)"),
    "incl@binary":                      (40.0,  90.0, "Orbital inclination (degrees)"),
    # incl floor lowered 50 -> 40: FI Boo (shallow eclipses) railed exactly at 50°, and
    # contact systems can still eclipse at moderate inclinations. Widening only helps
    # low-i systems; every other completed star sits at i >= 54°, so none are affected.
    "fillout_factor@contact_envelope":  (FILLOUT_FLOOR, 0.50, "Contact fill-out factor f"),
    # Bounds tightened at BOTH ends to keep walkers out of the mesh-collapse regime
    # (extreme overcontact and the f→0 neck pinch both gave "Mesh failed / Projections
    # failing", lnprob=-inf). Upper 0.70 -> 0.50 covers essentially all W UMa systems;
    # lower 0.0 -> FILLOUT_FLOOR (0.03, mesh-probe-verified at ntriangles=2000) stops the
    # low-/negative-f proposals that failed the mesh (see FILLOUT_FLOOR).
}

FREE_PARAMS_DETACHED: dict[str, tuple[float, float, str]] = {
    "teff@secondary":   (3000, 10000, "Secondary effective temperature (K)"),
    "incl@binary":      (60.0,  90.0, "Orbital inclination (degrees)"),
    "q@binary":         (0.1,   1.0,  "Mass ratio m₂/m₁"),
    # NOTE: requiv is in SOLAR RADII; these placeholder bounds are REPLACED at
    # runtime by sma-scaled Roche-lobe bounds in prepare_bundle_for_mcmc (the fit
    # effectively constrains the fractional radius r = requiv / sma).
    "requiv@primary":   (0.1,   1.0,  "Primary equivalent radius (R_sun; sma-scaled at runtime)"),
    "requiv@secondary": (0.1,   1.0,  "Secondary equivalent radius (R_sun; sma-scaled at runtime)"),
}

# Semi-detached: Script 04 adds PHOEBE's 'semidetached' constraint, so the
# secondary's radius equals its Roche lobe at every sampled q and is NOT free.
FREE_PARAMS_SEMIDETACHED: dict[str, tuple[float, float, str]] = {
    k: v for k, v in FREE_PARAMS_DETACHED.items() if k != "requiv@secondary"
}


# =============================================================================
# SECTION 3 — HELPER FUNCTIONS
# =============================================================================

# Morphology (contact / detached / semi-detached) comes from the shared module
# morphology.py, imported by Scripts 04, 05 and 06 alike, so a bundle is always
# optimised under the geometry it was built with. It replaces the local copies of
# MORPHOLOGY_OVERRIDE / is_contact_morphology / resolve_morphology (Entry 76).
from morphology import (                                    # noqa: E402
    CONTACT, DETACHED, SEMIDETACHED, MORPHOLOGY_OVERRIDE,
    is_contact_morphology, resolve_morphology, resolve_morphology_kind, morphology_source,
)


def free_params_for(kind: str) -> dict:
    """Copy of the free-parameter set for a morphology kind."""
    return {CONTACT: FREE_PARAMS_CONTACT, DETACHED: FREE_PARAMS_DETACHED,
            SEMIDETACHED: FREE_PARAMS_SEMIDETACHED}[kind].copy()


def adaptive_burn_in(chains_steps_first: np.ndarray) -> int:
    """
    Burn-in = max(MCMC_BURN_IN_STEPS, 2·τ_max), with τ measured on the chain
    AFTER the fixed floor (an untrimmed chain carries the initial transient into
    τ). Capped at half the chain. A fixed 350 steps was ~1.5 τ_max for V358 Cep
    (τ_incl ≈ 232); 2 τ is the usual minimum (Entry 76).
    """
    n_steps = chains_steps_first.shape[0]
    floor = min(MCMC_BURN_IN_STEPS, max(0, n_steps - 2))
    tail = chains_steps_first[floor:]
    burn = floor
    if tail.shape[0] > 50:
        tau = np.atleast_1d(assess_autocorrelation_convergence(tail))
        if np.all(np.isfinite(tau)):
            burn = max(floor, int(np.ceil(2.0 * float(np.max(tau)))))
    return int(min(burn, n_steps // 2)) if n_steps > 2 else 0


def steps_first(chains: np.ndarray) -> np.ndarray:
    """Orient an emcee chain as (n_steps, n_walkers, n_params)."""
    chains = np.asarray(chains)
    if chains.ndim == 3 and chains.shape[0] == MCMC_N_WALKERS and chains.shape[1] != MCMC_N_WALKERS:
        chains = np.swapaxes(chains, 0, 1)
    return chains


def prior_bounds(bundle: object, twig: str, fallback: tuple) -> tuple[float, float]:
    """(low, high) of the uniform prior actually stored in the bundle for `twig`."""
    try:
        qualifier, _, rest = twig.partition("@")
        kwargs = {"qualifier": qualifier, "context": "distribution", "distribution": "my_priors"}
        comp = rest.split("@")[0] if rest else ""
        if comp in ("primary", "secondary", "binary", "contact_envelope"):
            kwargs["component"] = comp
        dist = bundle.get_parameter(**kwargs).get_value()
        return float(dist.low), float(dist.high)
    except Exception:
        return float(fallback[0]), float(fallback[1])


def file_sha(path: str) -> str:
    """sha256 of a file's bytes, first 12 hex digits ('' if unreadable)."""
    try:
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()[:12]
    except OSError:
        return ""


def read_results_meta(results_path: str) -> dict:
    """Symbol → Value for the provenance rows of an existing results CSV."""
    meta = {}
    try:
        with open(results_path, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                meta[row.get("Symbol", "")] = row.get("Value", "")
    except OSError:
        pass
    return meta


def assess_contact_validity(bundle: object) -> str:
    """
    Flag whether a CONTACT fit's ADOPTED-MEDIAN geometry is actually a valid contact
    configuration. Returns "RECONSIDER_DETACHED" or "CONTACT_OK". Only FLAGS for human
    review (via MORPHOLOGY_OVERRIDE); never switches the model.

    SINGLE, SPECIFIC signal — PHOEBE run_checks on the adopted posterior-median geometry.
    If it reports the requiv-near-critical / "use detached/semidetached model" condition,
    the fitted contact geometry is genuinely invalid → RECONSIDER_DETACHED. Else CONTACT_OK.

    Fill-out railing and high χ² were tried (v2) and REMOVED. A low-fill-out,
    mesh-resolution-limited, O'Connell-inflated CONTACT star produces the same posterior
    signature (fill-out walling, poor χ²) as a detached-wanting one, so those heuristics
    FALSE-POSITIVE on valid contact systems — verified on FI Boo, which is unambiguously
    EW by its light curve yet tripped the v2 wall heuristic. Posterior statistics cannot
    separate the two; only the folded light-curve SHAPE can (continuous EW variation +
    O'Connell ⇒ contact; flat out-of-eclipse ⇒ detached). So the reliable trigger is an
    outright-invalid median geometry, and every flag is a prompt for a human to eyeball
    the light curve before deciding — never an assertion that the star IS detached.
    """
    try:
        report  = bundle.run_checks()
        passed  = getattr(report, "passed", None)
        text    = str(report).lower()
        flagged = (("requiv" in text and ("critical" in text or "requiv_min" in text))
                   or "use detached" in text or "semidetached" in text)
        if (passed is False) or flagged:
            return "RECONSIDER_DETACHED"
    except Exception:
        pass  # unusable check → no clear signal; stay conservative (CONTACT_OK)
    return "CONTACT_OK"


def eggleton_roche_radius(q: float) -> float:
    """
    Eggleton (1983) Roche-lobe radius in units of the semi-major axis.

        r_L / a = 0.49 q^(2/3) / (0.6 q^(2/3) + ln(1 + q^(1/3)))

    where q = M_star / M_companion for the star whose lobe is wanted. Used to cap
    the detached requiv priors below Roche-lobe filling so the MCMC cannot stray
    into L1-overflow geometries (the "potential > L1 critical value" condition seen
    for TZ Lyr during Script 04 bundle construction).
    """
    q13 = q ** (1.0 / 3.0)
    q23 = q13 * q13
    return 0.49 * q23 / (0.6 * q23 + np.log(1.0 + q13))


def compute_reduced_chi2(
    bundle: object,
    n_free_params: int,
    dataset: str = "lc01",
) -> tuple[float, float, int]:
    """
    Compute the reduced chi-squared goodness-of-fit statistic.

    χ²_red = Σ[(O_i − C_i)² / σ_i²] / (N − n_free)

    This metric reflects the exact residuals the MCMC sampler minimises
    and is computed without any post-hoc flux rescaling.

    Parameters
    ----------
    bundle:
        PHOEBE 2 bundle with a computed solution model.
    n_free_params:
        Number of free parameters (degrees-of-freedom denominator term).
    dataset:
        PHOEBE 2 dataset label (default "lc01").

    Returns
    -------
    chi2:
        Raw chi-squared sum.
    chi2_reduced:
        Reduced chi-squared (chi2 / dof).
    n_observations:
        Number of observed data points.
    """
    try:
        obs_flux  = bundle.get_value("fluxes", dataset=dataset, context="dataset")
        obs_sigma = bundle.get_value("sigmas", dataset=dataset, context="dataset")
        syn_flux  = bundle.get_value("fluxes", dataset=dataset, model="solution_model")

        n_obs       = len(obs_flux)
        dof         = max(1, n_obs - n_free_params)
        chi2        = float(np.sum(((obs_flux - syn_flux) / obs_sigma) ** 2))
        return chi2, chi2 / dof, n_obs
    except Exception:
        return np.inf, np.inf, 0


def assess_autocorrelation_convergence(mcmc_chains: np.ndarray) -> np.ndarray:
    """
    Compute the integrated autocorrelation time τ for each free parameter.

    A chain is converged when its length N satisfies N > CONVERGENCE_MULTIPLIER × τ.
    The emcee autocorr module uses the Goodman & Weare (2010) windowing function
    and returns the median τ across all walkers for each parameter dimension.

    Parameters
    ----------
    mcmc_chains:
        MCMC chain array with shape (n_steps, n_walkers, n_params).

    Returns
    -------
    tau_values:
        Array of integrated autocorrelation times, one per free parameter.
        Returns np.inf for each parameter if computation fails.
    """
    try:
        # tol=0 prevents the function from raising an error when the chain
        # is not yet long enough to satisfy the 50τ convergence criterion.
        return emcee.autocorr.integrated_time(mcmc_chains, tol=0)
    except Exception:
        return np.full(mcmc_chains.shape[-1], np.inf)


def gelman_rubin_rhat(mcmc_chains: np.ndarray) -> np.ndarray:
    """
    Gelman-Rubin potential scale reduction factor R̂ per parameter (Gelman &
    Rubin 1992; Brooks & Gelman 1998). R̂ → 1 as the between-walker variance
    vanishes, i.e. all walkers sample the same stationary distribution.

    Accepts emcee-native (n_steps, n_walkers, n_params) or the swapped
    (n_walkers, n_steps, n_params); returns np.inf per parameter on failure.
    """
    try:
        chains = np.asarray(mcmc_chains)
        if chains.ndim == 3 and chains.shape[1] == MCMC_N_WALKERS:
            chains = np.swapaxes(chains, 0, 1)          # -> (n_walkers, n_steps, n_params)
        n_w, n_s, n_p = chains.shape
        rhats = np.full(n_p, np.inf)
        for i in range(n_p):
            pc      = chains[:, :, i]                    # (n_walkers, n_steps)
            within  = np.mean(np.var(pc, axis=1))        # W
            between = n_s * np.var(np.mean(pc, axis=1))  # B
            if within > 0:
                var_plus = (1 - 1 / n_s) * within + (1 / n_s) * between
                rhats[i] = np.sqrt(var_plus / within)
        return rhats
    except Exception:
        return np.full(np.asarray(mcmc_chains).shape[-1], np.inf)


def chains_meet_convergence(bundle: object, done: int,
                            solution: str = "best_solution") -> bool:
    """
    Stop condition for the adaptive gate: True only when EVERY sampled parameter
    simultaneously clears both standard convergence tests —
      • chain length  done > CONVERGENCE_MULTIPLIER × τ   (Sokal 1997), and
      • R̂ < RHAT_CONVERGENCE_THRESHOLD                    (Gelman & Rubin 1992).
    A non-finite τ or R̂ (too-short or degenerate chain) counts as NOT converged,
    so the run continues toward the ceiling rather than stopping on a bad estimate.

    R̂ is evaluated on the POST-BURN-IN chain, matching the samples the posterior
    is actually drawn from. Evaluating it on the untrimmed chain — as this gate
    did through batch 4 — inflates the within-chain variance W with burn-in
    scatter, and since R̂ = √(V̂/W) that biases the test toward declaring
    convergence. The effect is negligible on a well-mixed star (ratio 1.00-1.01)
    but large on a poorly-mixed one: GQ Cnc's inclination reads R̂ = 1.41
    untrimmed and 2.04 trimmed. See Audits/audit_mcmc_chains.py, which reports
    both variants per star, and Pipeline_Log Entry 47.
    """
    try:
        chains = np.asarray(bundle.get_value("samples", solution=solution))
    except Exception:
        return False
    # τ estimator wants time on axis 0 (emcee-native); guard a transposed layout.
    tau_chains = steps_first(chains)
    # Adaptive burn-in (2·τ_max, floor MCMC_BURN_IN_STEPS); τ and R̂ are both
    # measured on the POST-burn-in chain the posterior is drawn from.
    trim = adaptive_burn_in(tau_chains)
    tau  = np.atleast_1d(assess_autocorrelation_convergence(tau_chains[trim:]))
    rhat = np.atleast_1d(gelman_rubin_rhat(tau_chains[trim:]))
    if not (np.all(np.isfinite(tau)) and np.all(np.isfinite(rhat))):
        return False
    tau_ok  = bool(np.all(done > CONVERGENCE_MULTIPLIER * tau))
    rhat_ok = bool(np.all(rhat < RHAT_CONVERGENCE_THRESHOLD))
    return tau_ok and rhat_ok


def compute_posterior_summary(
    bundle: object,
    solution: str = "best_solution",
    burnin: int = 0,
) -> dict[str, tuple[float, float, float]]:
    """
    Summarise the MCMC posterior for each fitted parameter as (median, −σ, +σ).

    Uncertainties are taken as the 16th/50th/84th percentiles of the
    post-burn-in, walker-flattened chain — the standard asymmetric 1σ credible
    interval for emcee posteriors (Foreman-Mackey et al. 2013). This replaces
    the earlier placeholder that reported NaN for every uncertainty, which left
    the results tables without error bars.

    Parameters
    ----------
    bundle:
        Bundle holding the completed MCMC solution.
    solution:
        PHOEBE solution label (default "best_solution").
    burnin:
        Number of initial steps discarded from each walker before summarising.

    Returns
    -------
    A dict keyed by a normalised PHOEBE qualifier (e.g. "teff@secondary",
    "incl", "q", "fillout_factor", "requiv@primary") mapping to
    (median, sigma_minus, sigma_plus). Empty if the chain is unavailable.
    """
    summary: dict[str, tuple[float, float, float]] = {}
    try:
        samples      = np.asarray(bundle.get_value("samples", solution=solution))
        fitted_twigs = list(bundle.get_value("fitted_twigs", solution=solution))
    except Exception:
        return summary

    # emcee stores chains in native order (n_steps, n_walkers, n_params) — time is
    # axis 0. Discard the burn-in STEPS along axis 0, then flatten across the
    # remaining steps and all walkers. (Slicing axis 1 here would wrongly discard
    # walkers rather than early steps.)
    if samples.ndim != 3:
        return summary
    # Orient so that axis 0 is steps even if a future PHOEBE version transposes.
    if samples.shape[0] == MCMC_N_WALKERS and samples.shape[1] != MCMC_N_WALKERS:
        samples = np.swapaxes(samples, 0, 1)
    n_steps = samples.shape[0]
    cut     = min(max(0, burnin), max(0, n_steps - 1))
    flat    = samples[cut:, :, :].reshape(-1, samples.shape[2])

    p16, p50, p84 = np.percentile(flat, [16, 50, 84], axis=0)
    for i, twig in enumerate(fitted_twigs):
        qualifier = twig.split("@")[0]
        component = twig.split("@")[1] if "@" in twig else ""
        # Retain the component tag only when it distinguishes the two stars,
        # so the extractor can match e.g. requiv@primary vs requiv@secondary.
        key = (f"{qualifier}@{component}"
               if component in ("primary", "secondary") else qualifier)
        summary[key] = (float(p50[i]), float(p50[i] - p16[i]), float(p84[i] - p50[i]))
    return summary


def extract_best_fit_parameters(
    bundle: object,
    free_params: dict[str, tuple],
    posterior: dict[str, tuple[float, float, float]],
) -> dict[str, tuple[float, float, float]]:
    """
    Combine adopted posterior-median point values with MCMC uncertainties.

    Point values are read from the bundle (already set to the posterior median by
    adopt_solution); the asymmetric ±σ come from `posterior`, the output of
    compute_posterior_summary(). Parameters that were held fixed (T1, and q for
    contact systems) carry zero / NaN uncertainty as appropriate.

    Parameters
    ----------
    bundle:
        Bundle after adopt_solution() has been called.
    free_params:
        The free parameter dictionary, used to detect contact vs. detached morphology.
    posterior:
        Per-qualifier (median, −σ, +σ) summary from compute_posterior_summary().

    Returns
    -------
    A dictionary mapping descriptive parameter keys to (value, −σ, +σ) tuples.
    """
    def unc(*keys: str) -> tuple[float, float]:
        """Return (−σ, +σ) for the first matching posterior key, else (NaN, NaN)."""
        for k in keys:
            if k in posterior:
                return posterior[k][1], posterior[k][2]
        return np.nan, np.nan

    t1   = bundle.get_value("teff", component="primary",   context="component")
    t2   = bundle.get_value("teff", component="secondary", context="component")
    incl = bundle.get_value("incl", component="binary",    context="component")
    q    = bundle.get_value("q",    component="binary",    context="component")

    t2_m,   t2_p   = unc("teff@secondary")
    incl_m, incl_p = unc("incl", "incl@binary")
    q_m,    q_p    = unc("q", "q@binary")

    # T1 is held fixed, so the T₂/T₁ ratio inherits the secondary's error scaled by 1/T1.
    ratio = t2 / t1
    if np.isfinite(t2_m):
        ratio_m, ratio_p = t2_m / t1, t2_p / t1
    else:
        ratio_m, ratio_p = np.nan, np.nan

    results: dict[str, tuple[float, float, float]] = {
        "T1_K":       (t1, 0.0, 0.0),
        "T2_K":       (t2, t2_m, t2_p),
        "T2_over_T1": (ratio, ratio_m, ratio_p),
        "incl_deg":   (incl, incl_m, incl_p),
        "q":          (q, q_m, q_p),
    }

    if "fillout_factor@contact_envelope" in free_params:
        try:
            f_val      = bundle.get_value("fillout_factor", context="component")
            f_m, f_p   = unc("fillout_factor", "fillout_factor@contact_envelope")
            results["fillout_factor"] = (f_val, f_m, f_p)
        except Exception:
            results["fillout_factor"] = (np.nan, np.nan, np.nan)
    else:
        try:
            # Report FRACTIONAL radii r = requiv / sma. With single-band photometry
            # the absolute scale (sma) is degenerate — the light curve constrains
            # only the relative radii r = R/a, the inclination, and the temperature
            # ratio; absolute radii and masses require spectroscopic radial
            # velocities (Kallrath & Milone 1999; Prša & Zwitter 2005). sma is held
            # fixed, so the requiv uncertainties scale by 1/sma into r.
            sma      = bundle.get_value("sma", component="binary", context="component")
            r1       = bundle.get_value("requiv", component="primary",   context="component")
            r2       = bundle.get_value("requiv", component="secondary", context="component")
            r1_m, r1_p = unc("requiv@primary")
            r2_m, r2_p = unc("requiv@secondary")
            scale = lambda x: x / sma if np.isfinite(x) else np.nan
            results["r1_frac"] = (r1 / sma, scale(r1_m), scale(r1_p))
            results["r2_frac"] = (r2 / sma, scale(r2_m), scale(r2_p))
        except Exception:
            results["r1_frac"] = (np.nan, np.nan, np.nan)
            results["r2_frac"] = (np.nan, np.nan, np.nan)

    return results


# NOTE: there is deliberately NO user-level MPI broadcast helper here. Under
# PHOEBE's MPI model only rank 0 executes this script (rank>0 are held in PHOEBE's
# worker loop from import), so any user comm.bcast() collides with that loop and
# crashes the workers. All MPI coordination is handled internally by run_solver.


def completed_iterations(bundle: object, solution: str = "best_solution") -> int:
    """Return how many MCMC steps the saved chain already contains (0 if none).

    PHOEBE stores the emcee chain in native order (n_steps, n_walkers, n_params),
    so the step count is axis 0 — NOT axis 1 (which is the walker count, and which
    an earlier version mistakenly returned, reporting 14/30 after 10 steps).
    """
    try:
        samples = np.asarray(bundle.get_value("samples", solution=solution))
        if samples.ndim != 3:
            return 0
        # Robust against transposition: the steps axis is the one that is not the
        # walker count (axis 1 in emcee-native order).
        if samples.shape[1] == MCMC_N_WALKERS:
            return int(samples.shape[0])
        if samples.shape[0] == MCMC_N_WALKERS:
            return int(samples.shape[1])
        return int(samples.shape[0])
    except Exception:
        return 0


def read_failed_models(bundle: object, solution: str = "best_solution") -> tuple:
    """(count, most common message) of proposals whose forward model failed
    (lnprob = −inf), from PHOEBE's `failed_samples`. PHOEBE carries that dict across
    continue_from, so the count is cumulative over the whole chain. ("", "unavailable:
    …") if it cannot be read — this feeds run statistics only and must never raise."""
    try:
        return runstats.count_failed(bundle.get_value("failed_samples", solution=solution))
    except Exception as failed_error:
        return "", f"unavailable: {failed_error}"[:120]


def derive_free_params_meta(
    star_name: str,
    variable_type: str,
) -> tuple[dict, bool, int]:
    """
    Reconstruct the free-parameter set, morphology flag, and parameter count
    without touching the bundle.

    Used on the resume path, where the priors and solver already live inside the
    loaded checkpoint and only the membership/count metadata is needed downstream
    (degrees of freedom, contact-vs-detached result extraction).
    """
    kind = resolve_morphology_kind(star_name, variable_type)
    is_contact = kind == CONTACT
    free_params = free_params_for(kind)
    if star_name == "BD And":
        free_params["l3_frac@lc01"] = (0.0, 0.20, "Third-light contamination fraction")
    return free_params, is_contact, len(free_params)


def prepare_bundle_for_mcmc(
    star_name: str,
    variable_type: str,
    bundle_path: str,
) -> Optional[tuple]:
    """
    Load a Script-04 bundle and apply all pre-MCMC configuration (physics,
    sanitisation, limb darkening, priors, Nelder-Mead pre-wash, emcee solver).

    Returns (bundle, free_params, is_contact_binary, n_free_params), or None if
    the input bundle is missing. This is the "fresh start" path; on resume the
    equivalent state is restored from the checkpoint instead.
    """
    if not os.path.exists(bundle_path):
        master_print(f"  WARNING: {bundle_path} not found. Skipping.")
        return None

    bundle = phoebe.load(bundle_path)
    master_print(f"  Bundle loaded: {bundle_path}")

    # ── 5A: RADIATIVE PHYSICS ─────────────────────────────────────────────────
    # Re-apply physics coefficients here in case the bundle was saved with
    # default values or the temperature was adjusted since Script 04.
    for component in ["primary", "secondary"]:
        teff = bundle.get_value(f"teff@{component}")
        if teff >= RADIATIVE_CONVECTIVE_BOUNDARY_K:
            bundle.set_value(f"gravb_bol@{component}", 1.0)
            bundle.set_value(f"irrad_frac_refl_bol@{component}", 1.0)
            master_print(f"  [physics] {component}: radiative  β₁=1.0, A=1.0  ({teff:.0f} K)")
        else:
            bundle.set_value(f"gravb_bol@{component}", 0.32)
            bundle.set_value(f"irrad_frac_refl_bol@{component}", 0.6)
            master_print(f"  [physics] {component}: convective β₁=0.32, A=0.6  ({teff:.0f} K)")

    # Irradiation method in the COMPUTE OPTIONS, so the sampler and the final
    # compute use the same model (contact = Wilson, otherwise Horvat). Script 04
    # stores it since 2026-10; set here too so an older bundle cannot sample with
    # PHOEBE's default (Horvat) and report with Wilson (Entry 76).
    _kind = resolve_morphology_kind(star_name, variable_type)
    bundle.set_value("irrad_method", compute="phoebe01",
                     value="wilson" if _kind == CONTACT else "horvat")

    # ── 5B: DATA SANITISATION ─────────────────────────────────────────────────
    # Duplicate timestamps cause PHOEBE 2's interpolation to fail. Remove them
    # by rebuilding the dataset with only the unique-time subset.
    try:
        obs_times  = bundle.get_value("times",  dataset="lc01", context="dataset")
        obs_fluxes = bundle.get_value("fluxes", dataset="lc01", context="dataset")
        obs_sigmas = bundle.get_value("sigmas", dataset="lc01", context="dataset")
        passband   = bundle.get_value("passband",   dataset="lc01", context="dataset")
        pblum_mode = bundle.get_value("pblum_mode", dataset="lc01")
        # The dataset is rebuilt below; exptime (set by Script 04 for long-cadence
        # data, with fti_method="oversample" in the compute options) would be lost.
        exptime    = float(bundle.get_value("exptime", dataset="lc01", context="dataset",
                                            check_visible=False))
        # fti_method / fti_oversample are PER-DATASET compute options
        # (fti_method@lc01@phoebe01): re-adding lc01 resets them to "none" / default,
        # which silently switched exposure integration off (Phase D, Entry 79).
        fti_method = bundle.get_value("fti_method", compute="phoebe01", dataset="lc01",
                                      check_visible=False)
        fti_over   = bundle.get_value("fti_oversample", compute="phoebe01", dataset="lc01",
                                      check_visible=False)

        unique_mask     = np.unique(obs_times, return_index=True)[1]
        n_duplicates    = len(obs_times) - len(unique_mask)

        bundle.remove_dataset("lc01")
        bundle.add_dataset(
            "lc",
            times=obs_times[unique_mask],
            fluxes=obs_fluxes[unique_mask],
            sigmas=obs_sigmas[unique_mask],
            passband=passband,
            dataset="lc01",
        )
        if exptime > 0:
            bundle.set_value("exptime", dataset="lc01", context="dataset", value=exptime)
            bundle.set_value("fti_method", compute="phoebe01", dataset="lc01", value=fti_method)
            if fti_method != "none":
                bundle.set_value("fti_oversample", compute="phoebe01", dataset="lc01",
                                 value=int(fti_over))
            _fti_now = bundle.get_value("fti_method", compute="phoebe01", dataset="lc01",
                                        check_visible=False)
            _ovs_now = bundle.get_value("fti_oversample", compute="phoebe01", dataset="lc01",
                                        check_visible=False)
            master_print(f"  [sanitise] exptime {exptime:.0f} s restored "
                         f"(fti_method = {_fti_now}, fti_oversample = {_ovs_now})")
        # 'dataset-scaled' for EVERY morphology: the model is computed with the
        # components' physical (coupled) luminosities and then rescaled as a whole
        # to the data, so the flux level is effectively free at every step.
        # Measured 2026-10-07 (Entry 76) on a detached bundle: dataset-scaled =
        # k × component-coupled to 1e-15 of the amplitude in four configurations
        # (it does NOT silence the secondary, as the old "Bug C" note claimed), and
        # k ranged 0.37-0.83 with T2 and the radii — which is exactly the flux-level
        # drift that the former component-coupled mode, rescaled once at the seed,
        # forced the fit to absorb into r1, r2 and T2.
        bundle.set_value("pblum_mode", dataset="lc01", value="dataset-scaled")
        master_print(f"  [sanitise] Removed {n_duplicates} duplicate timestamps.")
    except Exception as sanitise_error:
        master_print(f"  [sanitise warning] {sanitise_error}")

    # ── 5C: LIMB DARKENING ────────────────────────────────────────────────────
    # A blackbody is not a uniform disk: the Eddington-Barbier relation
    # predicts that limb-regions appear cooler due to shallower optical-depth
    # sampling. u ≈ 0.6 is a standard linear LD coefficient for solar-type
    # stars in the TESS bandpass and prevents the uniform-disk approximation
    # from biasing the fitted inclination and radii.
    bundle.set_value("ld_mode_bol@primary",   "manual")
    bundle.set_value("ld_mode_bol@secondary", "manual")
    for comp in ["primary", "secondary"]:
        try:
            bundle.set_value(f"ld_mode@{comp}@lc01",   "manual")
            bundle.set_value(f"ld_func@{comp}@lc01",   "linear")
            bundle.set_value(f"ld_coeffs@{comp}@lc01", [0.6])
        except Exception as ld_error:
            master_print(f"  [LD warning] {comp}: {ld_error}")

    # ── 5D: FREE PARAMETERS & PRIORS ─────────────────────────────────────────
    morphology_kind   = resolve_morphology_kind(star_name, variable_type)
    is_contact_binary = morphology_kind == CONTACT

    if is_contact_binary:
        try:
            bundle.flip_constraint("fillout_factor", solve_for="pot")
        except Exception:
            pass  # constraint may already be inverted from Script 04

    free_params = free_params_for(morphology_kind)
    master_print(f"  [morphology] {morphology_kind} ({morphology_source(star_name)})")

    # Apply a temperature prior anchored to T1 to prevent the secondary from
    # wandering into physically implausible regions during burn-in.
    t1_current = bundle.get_value("teff@primary", context="component")
    if is_contact_binary:
        # W-type W UMa systems can have T2 > T1; allow up to 25% above T1.
        t2_lower, t2_upper = 0.80 * t1_current, 1.25 * t1_current
    else:
        t2_lower, t2_upper = 0.70 * t1_current, 1.10 * t1_current

    # --- β₁ regime lock (fixes the frozen-gravity-darkening bug) -------------
    # gravb_bol / irrad_frac_refl_bol are set once (5A) from the current
    # secondary temperature and are NOT re-evaluated inside PHOEBE's emcee solver.
    # If the T₂ prior straddled the 8000 K convective/radiative boundary, walkers
    # crossing it would keep the stale coefficient — physically inconsistent.
    # We therefore clamp the prior to the regime the secondary currently sits in,
    # so the coefficient assigned in 5A is valid for the entire chain. The hard
    # discontinuity at 8000 K (Lucy 1967 ↔ von Zeipel 1924) makes sampling across
    # it ill-defined anyway; locking to one regime is the self-consistent choice.
    t2_current = bundle.get_value("teff@secondary", context="component")
    boundary   = RADIATIVE_CONVECTIVE_BOUNDARY_K
    if t2_current >= boundary and t2_lower < boundary < t2_upper:
        t2_lower    = boundary
        regime_note = "locked ≥8000 K (radiative β₁=1.0)"
    elif t2_current < boundary and t2_lower < boundary < t2_upper:
        t2_upper    = boundary
        regime_note = "locked <8000 K (convective β₁=0.32)"
    else:
        regime_note = "single regime — no clamp needed"

    free_params["teff@secondary"] = (t2_lower, t2_upper, "Secondary Teff — thermally bounded prior")
    master_print(f"  [prior] T₂ ∈ [{t2_lower:.0f}, {t2_upper:.0f}] K  ({regime_note})")

    # --- Roche-lobe-scaled requiv priors (detached systems; UNITS FIX) -------
    # CRITICAL: PHOEBE's requiv is in SOLAR RADII, not in units of the semi-major
    # axis. The previous prior [0.1, 1.0] (and an Eggleton cap missing the × sma
    # factor) forced R/a ≈ 0.02–0.19 at PHOEBE's default sma = 5.3 R⊙ — far too
    # small to eclipse, producing a flat model (TZ Lyr modelled depth 0.001 vs data
    # 0.59). We scale both bounds by sma so they bracket physical fractional radii:
    # a small-radius floor up to ROCHE_LOBE_CAP x the Eggleton (1983) Roche lobe.
    # KNOWN LIMITATION: the lobe is evaluated at the SEED q, but q is a free
    # parameter, so the effective cap drifts as q samples — it is a larger
    # fraction of the true lobe at low q and a smaller one at high q. Measured on
    # the ZZ Det scratch star: intended 0.9, actually 0.943 of the true lobe at
    # the fitted q (Entry 71). Raising the cap to 1.0 shrinks that drift's
    # consequences, since an over-permissive bound is caught by PHOEBE's own
    # overflow check while an over-tight one silently truncates the posterior.
    # With single-band photometry the
    # absolute scale (sma) is degenerate, so sma is held fixed and these bounds
    # effectively constrain the fractional radii r = requiv / sma.
    if "requiv@primary" in free_params:
        q_seed       = bundle.get_value("q@binary",   context="component")
        sma          = bundle.get_value("sma@binary", context="component")  # R⊙
        # Caps at the LARGEST lobe the q prior allows (primary: lowest q; secondary:
        # highest q). Before 2026-10 the cap was evaluated at the SEED q, so it was
        # tighter than the true lobe whenever q sampled the other way (Entry 76). An
        # over-permissive cap is harmless: PHOEBE's own exact overflow check rejects
        # requiv > requiv_max at the sampled q.
        q_lo, q_hi   = free_params["q@binary"][0], free_params["q@binary"][1]
        rl_primary   = eggleton_roche_radius(1.0 / q_lo)     # primary lobe fraction r_L/a
        rl_secondary = eggleton_roche_radius(q_hi)           # secondary lobe fraction
        for twig, rl in (("requiv@primary", rl_primary), ("requiv@secondary", rl_secondary)):
            if twig not in free_params:
                continue                  # semi-detached: r2 is set by the lobe constraint
            _, _, desc = free_params[twig]
            # NOTE: `sma` cancels out of the effective constraint — both bounds
            # carry a factor of sma and the physical quantity is r = requiv/sma,
            # so r is bounded by [0.05, ROCHE_LOBE_CAP * rl] whatever sma is.
            # A wrong fixed sma does NOT mis-scale this prior.
            cap_solrad = ROCHE_LOBE_CAP * rl * sma
            lo_solrad  = 0.05 * sma         # fractional-radius floor r ≈ 0.05
            free_params[twig] = (lo_solrad, cap_solrad, desc)
            master_print(f"  [prior] {twig} ∈ [{lo_solrad:.3f}, {cap_solrad:.3f}] R⊙  "
                         f"(R/a ≤ {ROCHE_LOBE_CAP * rl:.3f} = {ROCHE_LOBE_CAP:.2f}×largest lobe "
                         f"over q ∈ [{q_lo}, {q_hi}]; sma={sma:.2f} R⊙)")

        # r₂ > r₁ seed: a deep total primary eclipse requires the secondary to be
        # larger than the primary. A symmetric seed consistently fell into the
        # r₁-at-cap attractor (small secondary transiting giant primary → wrong basin).
        r1_seed = 0.17 * sma
        bundle.set_value("requiv@primary",   r1_seed)
        if "requiv@secondary" in free_params:
            r2_seed = 0.24 * sma
            bundle.set_value("requiv@secondary", r2_seed)
            master_print(f"  [seed] requiv@primary = {r1_seed:.3f} R⊙ (r/a = 0.17), "
                         f"requiv@secondary = {r2_seed:.3f} R⊙ (r/a = 0.24)  "
                         f"[r₂/r₁ = 1.41, total-eclipse NM seed]")
        else:
            r2_lobe = bundle.get_value("requiv", component="secondary", context="component")
            master_print(f"  [seed] requiv@primary = {r1_seed:.3f} R⊙ (r/a = 0.17); "
                         f"requiv@secondary = {r2_lobe:.3f} R⊙ (= Roche lobe, constrained)")

    # Third-light optimisation for BD And — TESS's coarse pixel scale (~21"/px)
    # makes contamination from nearby field stars a realistic systematic for this
    # target. This MUST precede the prior loop below: a twig added to free_params
    # afterwards gets no entry in `my_priors`, and (since the 5D-b split) would
    # land in `my_init` with no prior to bound it. BD And has no bundle yet, so
    # this has never actually run.
    if star_name == "BD And":
        bundle.set_value("l3_mode", dataset="lc01", value="fraction")
        free_params["l3_frac@lc01"] = (0.0, 0.20, "Third-light contamination fraction")
        master_print("  [physics] BD And: third-light fraction l3 enabled (0–20%).")

    n_free_params = len(free_params)

    master_print(f"\n  [5D] Configuring {n_free_params} free parameters and priors...")
    # ONLY the priors are built here. The emcee initialisation ball ("my_init")
    # is built AFTER the Nelder-Mead pre-wash in 5E — see the block there for why
    # the order matters. Nelder-Mead needs "my_priors" to bound its simplex, so
    # the prior half must stay ahead of it.
    prior_distributions = {}

    for twig, (lo, hi, description) in free_params.items():
        prior_distributions[twig] = phoebe.uniform(lo, hi)
        master_print(f"    {twig:<40}  [{lo:.2f}, {hi:.2f}]  — {description}")

    bundle.add_distribution(prior_distributions, distribution="my_priors")

    # (The former "Bug E" pblum@primary rescale at the seed is gone: with
    # dataset-scaled luminosities for every morphology the flux level is fitted at
    # every step — see 5B.)

    # ── 5E: STAGE 1 — NELDER-MEAD PRE-WASH ──────────────────────────────────
    master_print(f"\n  [Stage 1] Nelder-Mead simplex pre-wash (max {NM_MAXITER} iterations)...")
    try:
        bundle.add_solver("optimizer.nelder_mead", solver="nm_solver", overwrite=True)
        # The optimiser MUST be told which parameters to fit via `fit_parameters`;
        # this was previously unset, so the pre-wash optimised nothing (and an
        # earlier attempt to bound it via ParameterSet.set_limits raised and skipped
        # NM entirely). Bound the simplex with the SAME uniform priors as the MCMC
        # (the nelder_mead solver folds the prior into its merit function), so it
        # stays inside the physical ranges — including the Eggleton requiv caps.
        bundle.set_value("fit_parameters", solver="nm_solver",
                         value=list(free_params.keys()))
        bundle.set_value("priors", solver="nm_solver", value=["my_priors"])
        bundle.set_value("maxiter", solver="nm_solver", value=NM_MAXITER)

        global last_nm_seconds
        _nm_t0 = time.time()
        bundle.run_solver("nm_solver", solution="nm_solution", overwrite=True)
        last_nm_seconds = time.time() - _nm_t0
        bundle.adopt_solution("nm_solution", adopt_values=True)
        try:
            _nm_ok  = bool(bundle.get_value("success", solution="nm_solution"))
            _nm_msg = str(bundle.get_value("message", solution="nm_solution"))
        except Exception:
            _nm_ok, _nm_msg = None, "status unavailable"
        _nm_state = {True: "converged", False: "did NOT converge", None: "finished"}[_nm_ok]
        master_print(f"  Nelder-Mead {_nm_state} ({_nm_msg}) — adopted as MCMC starting point.")
    except Exception as nm_error:
        master_print(f"  [NM warning] Pre-wash failed: {nm_error}. Using initial guess.")

    # ── 5D-b: emcee INITIALISATION BALL (must follow the 5E adoption) ────────
    # This block reads `bundle.get_value(twig)`, so it MUST run AFTER
    # adopt_solution("nm_solution") above. It used to live in 5D, i.e. BEFORE the
    # pre-wash — which meant the ball was centred on Script 04's seed values and
    # Nelder-Mead's result was computed, stored in the bundle, and then silently
    # discarded by the sampler (`init_from="my_init"` below). Every contact star
    # therefore entered the MCMC at the identical point i = 82.0°, f = 0.15,
    # q = 0.5, regardless of what NM had found; walkers spent the whole burn-in
    # travelling back to the NM solution, and stars whose acceptance had collapsed
    # never got there at all. See Pipeline_Log Entry 57.
    #
    # Deliberately OUTSIDE the try/except above: if NM raised, the bundle still
    # holds the Script 04 seed and this degrades to the old behaviour rather than
    # failing the star.
    master_print("\n  [5D-b] emcee initialisation ball (centred on the adopted solution):")
    init_distributions = {}
    for twig, (lo, hi, description) in free_params.items():
        # context= is REQUIRED here and was not needed in the old 5D placement:
        # `my_priors` now exists by this point, so a bare twig like
        # "teff@secondary" matches both the component parameter and its prior
        # distribution, and PHOEBE raises "2 results found". Both contexts are
        # listed because BD And's l3_frac@lc01 lives in the dataset context while
        # every other free parameter lives in the component context.
        current_val  = bundle.get_value(twig, context=["component", "dataset"])
        param_range  = hi - lo
        # 2% Gaussian ball: tight enough to keep walkers out of Roche-lobe
        # overflow, wide enough to give emcee's stretch move something to work with.
        init_sigma   = 0.02 * param_range
        clipped_val  = float(np.clip(current_val, lo + init_sigma, hi - init_sigma))
        init_distributions[twig] = phoebe.normal(clipped_val, init_sigma)
        # Printed per parameter because the absence of exactly this line is what
        # let the ordering bug survive 29 stars unnoticed.
        clip_note = "" if np.isclose(clipped_val, current_val) else \
                    f"  [CLIPPED from {current_val:.4f} — solution railed at a bound]"
        master_print(f"    {twig:<40}  N({clipped_val:.4f}, {init_sigma:.4f}){clip_note}")

    bundle.add_distribution(init_distributions, distribution="my_init")

    # ── 5F: ADD emcee SOLVER (execution happens in the chunked loop) ──────────
    _mcmc_budget = (f"{MCMC_N_ITERATIONS} steps" if MCMC_FIXED_MODE
                    else f"{MCMC_MIN_NITERS}–{MCMC_MAX_NITERS} steps, adaptive")
    master_print(f"\n  [Stage 2] Configuring emcee MCMC "
                 f"({MCMC_N_WALKERS} walkers × {_mcmc_budget})...")
    bundle.add_solver(
        "sampler.emcee",
        solver="mcmc_sampler",
        init_from="my_init",
        priors="my_priors",
        nwalkers=MCMC_N_WALKERS,
        niters=MCMC_N_ITERATIONS,
        overwrite=True,
    )

    return bundle, free_params, is_contact_binary, n_free_params


# Stars whose TESS aperture holds a VARIABLE neighbour that PDCSAP's constant
# crowding correction cannot remove (Entry 76: BW Dra and BV Dra, 16.05″ apart,
# each carrying the other's eclipses at its P/2). Kept in the sample by decision
# and FLAGGED in every results file.
VARIABLE_BLENDS: dict[str, str] = {
    "BW Dra": "blended_with_variable_neighbour:BV Dra (16 arcsec)",
    "BV Dra": "blended_with_variable_neighbour:BW Dra (16 arcsec)",
}


# Pre-audit results KEPT by decision (Pipeline_Log Entry 78): fold-smearing cost
# them < 3 % of light-curve amplitude (Audits/kept_results_caveats.csv), so they
# are documented rather than refitted. Without this list the Entry-77 skip rule
# would supersede and refit them, because they carry no input_bundle_sha row.
KEEP_PRE_AUDIT_RESULTS: frozenset = frozenset({
    "BO CVn", "DF Hya", "EF Cep", "EF Dra", "FP Boo", "FU Dra", "GQ Cnc", "HI Pup",
    "II UMa", "NU Cyg", "RT Hyi", "RW Dor", "TY Men", "TY UMa", "V1044 Her",
    "V1055 Her", "V1128 Tau", "V1918 Cyg", "V776 Cas", "V842 Her", "VW Cep", "Y Sex",
})


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def acquire_star_lock(safe_name: str) -> tuple[bool, str]:
    """
    Claim a star for THIS Script-05 process (rank 0) with an exclusive lock file
    data/bundles/bundle_<star>.lock holding "<host>:<pid>". Lets two or more queues
    run from the same catalogue (the "dynamic -n" mode, Entry 78): each queue
    skips stars another live queue holds. A lock left by a dead process on this
    host (crash, segfault, Ctrl-C) is taken over. Returns (acquired, holder).
    """
    import socket
    path = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}.lock")
    me = f"{socket.gethostname()}:{os.getpid()}"
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, me.encode())
            os.close(fd)
            return True, me
        except FileExistsError:
            try:
                holder = open(path, encoding="utf-8").read().strip()
                host, pid = holder.rsplit(":", 1)
                pid = int(pid)
            except (OSError, ValueError):
                holder, host, pid = "unreadable", "", 0
            if host == socket.gethostname() and not _pid_alive(pid):
                try:
                    os.remove(path)          # stale lock from a dead process
                except OSError:
                    pass
                continue
            return False, holder
    return False, "race"


def release_star_lock(safe_name: str) -> None:
    """Remove the star's lock if (and only if) this process holds it."""
    import socket
    path = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}.lock")
    try:
        if open(path, encoding="utf-8").read().strip() == f"{socket.gethostname()}:{os.getpid()}":
            os.remove(path)
    except OSError:
        pass


def phoebe_light_curve_flags(safe_name: str) -> dict:
    """FLUXSRC / CROWDSAP from the Script 02 clean-FITS header, plus data flags."""
    from astropy.io import fits as _fits
    out = {"flags": []}
    star = safe_name.replace("_", " ")
    if star in VARIABLE_BLENDS:
        out["flags"].append(VARIABLE_BLENDS[star])
    path = os.path.join(paths.LIGHTCURVE_DIR, f"lc_{safe_name}_clean.fits")
    with _fits.open(path) as hdul:
        for hdu in hdul[:2]:
            for key in ("FLUXSRC", "CROWDSAP", "SECTOR"):
                if key in hdu.header and key not in out:
                    out[key] = hdu.header[key]
    if out.get("FLUXSRC") == "SAP_CROWDCORR":
        out["flags"].append("SAP_flux_PDC_distorted")
    return out


# =============================================================================
# SECTION 4 — TARGET LIST
# =============================================================================

target_list: list[dict] = []
try:
    with open(INPUT_CATALOG, newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            target_list.append({
                "name":   row["name"],
                "type":   row["type"],
                "period": float(row["period"]),
            })
except FileNotFoundError:
    master_print(f"ERROR: {INPUT_CATALOG} not found. Run Script 01 first.")
    raise SystemExit(1)


# =============================================================================
# SECTION 5 — MAIN OPTIMISATION LOOP
# =============================================================================

PARAM_DISPLAY_LABELS: dict[str, tuple[str, str]] = {
    "T1_K":             ("Primary temperature T₁",    "K"),
    "T2_K":             ("Secondary temperature T₂",  "K"),
    "T2_over_T1":       ("Temperature ratio T₂/T₁",   "—"),
    "incl_deg":         ("Orbital inclination i",      "°"),
    "q":                ("Mass ratio q = m₂/m₁",      "—"),
    "r1_frac":          ("Primary fractional radius r₁ = R₁/a",   "—"),
    "r2_frac":          ("Secondary fractional radius r₂ = R₂/a", "—"),
    "fillout_factor":   ("Contact fill-out factor f",  "—"),
}

# Provenance row in Runs/run_stats/runs.csv: code fingerprint (this script, the chain
# audit whose verdicts the rows report, runstats, paths) and every setting that shapes
# the fit. Recorded here rather than at tracker creation because these constants are
# defined after it.
run_tracker.record_provenance(
    settings={
        "mpi_n_procs": mpi_n_procs, "MCMC_N_WALKERS": MCMC_N_WALKERS,
        "MCMC_CHECKPOINT_NITERS": MCMC_CHECKPOINT_NITERS,
        "MCMC_MIN_NITERS": MCMC_MIN_NITERS, "MCMC_MAX_NITERS": MCMC_MAX_NITERS,
        "MCMC_FIXED_MODE": MCMC_FIXED_MODE, "MCMC_BURN_IN_STEPS": MCMC_BURN_IN_STEPS,
        "NM_MAXITER": NM_MAXITER, "CONVERGENCE_MULTIPLIER": CONVERGENCE_MULTIPLIER,
        "RHAT_CONVERGENCE_THRESHOLD": RHAT_CONVERGENCE_THRESHOLD,
        "FILLOUT_FLOOR": FILLOUT_FLOOR, "ROCHE_LOBE_CAP": ROCHE_LOBE_CAP,
        "phoebe": phoebe.__version__, "emcee": emcee.__version__,
    },
    code_files=[os.path.abspath(__file__), runstats.__file__, paths.__file__,
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "morphology.py"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audits",
                             "audit_mcmc_chains.py")],
)

# For the batch ETA (5F): which targets still lack a results file at launch. Counted
# once; a star finished during this run is always earlier in the list than the one
# being estimated, so it is never double-counted.
def _is_complete(name: str) -> bool:
    """Results exist AND were fitted from the current input bundle (see the skip rule)."""
    safe = name.replace(" ", "_")
    res = os.path.join(paths.RESULTS_DIR, f"results_{safe}.csv")
    if not os.path.exists(res):
        return False
    if name in KEEP_PRE_AUDIT_RESULTS:
        return True
    recorded = read_results_meta(res).get("input_bundle_sha", "")
    return bool(recorded) and recorded == file_sha(os.path.join(paths.BUNDLE_DIR, f"bundle_{safe}.phoebe"))


target_pending = [not _is_complete(t["name"]) for t in target_list]

# Per-star locks (see acquire_star_lock): the lock held for the current star is
# released when the loop moves on, and at exit whatever the exit path.
_held_lock: list = []
if mpi_rank == 0:
    import atexit
    atexit.register(lambda: [release_star_lock(n) for n in list(_held_lock)])

for target_index, target in enumerate(target_list):
    star_name     = target["name"]
    variable_type = target["type"]
    safe_name     = star_name.replace(" ", "_")
    while _held_lock:
        release_star_lock(_held_lock.pop())
    run_tracker.star_start(star_name)
    star_t0         = time.time()
    star_audit      = None
    last_nm_seconds = float("nan")
    setup_seconds   = float("nan")

    master_print("\n" + "=" * 70)
    master_print(f"OPTIMISATION: {star_name}  ({variable_type})  |  Strategy: MCMC (emcee)")
    master_print("=" * 70)

    # ── TARGET PATHS, SKIP & RESUME DETECTION ────────────────────────────────
    bundle_path     = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}.phoebe")
    mcmc_checkpoint = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}_mcmc.phoebe")
    results_path    = os.path.join(paths.RESULTS_DIR, f"results_{safe_name}.csv")

    # Idempotent re-runs: a completed target is skipped, so an interrupted batch can
    # simply be relaunched with the same command and will pick up where it stopped.
    # "Completed" now means the results file exists AND was fitted from the CURRENT
    # input bundle (its input_bundle_sha row matches bundle_<star>.phoebe). A results
    # file from an older bundle, or one without the row (every pre-2026-10 result),
    # is renamed *_superseded_<date>.csv and the star is refitted (Entry 76: before
    # this, a rebuilt bundle was never refitted).
    if star_name in KEEP_PRE_AUDIT_RESULTS and os.path.exists(results_path):
        master_print(f"  [keep] {results_path} is a pre-audit result KEPT by decision "
                     "(Pipeline_Log Entry 78) — not refitted.")
        run_tracker.star_end(star_name, "skipped", reason="kept pre-audit result (Entry 78)")
        continue

    # Claim the star before touching its results or checkpoint, so a second queue
    # running the same catalogue can never resume, supersede or refit a star this
    # queue is fitting (Entry 78, dynamic -n mode).
    if mpi_rank == 0:
        _got, _holder = acquire_star_lock(safe_name)
        if not _got:
            master_print(f"  [skip] {star_name} is being fitted by another queue ({_holder}).")
            run_tracker.star_end(star_name, "skipped", reason=f"locked by {_holder}")
            continue
        _held_lock.append(safe_name)

    input_sha = file_sha(bundle_path)
    if os.path.exists(results_path):
        recorded = read_results_meta(results_path).get("input_bundle_sha", "")
        if recorded and recorded == input_sha:
            master_print(f"  [skip] {results_path} exists and matches the current bundle "
                         f"({input_sha}) — target complete.")
            run_tracker.star_end(star_name, "skipped", reason="results already exist")
            continue
        _stamp = time.strftime('%Y%m%d')
        superseded = results_path.replace(".csv", f"_superseded_{_stamp}.csv")
        # The old FINAL bundle holds the old posterior chain (the expensive part) and
        # would be overwritten by this refit — keep it, and the old final plot, under
        # the same suffix (blast radius; cf. the V899 Her incident, Entry 58).
        _old_final = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}_final.phoebe")
        _old_plot  = os.path.join(paths.PLOT_FINAL_DIR, f"phoebe_final_{safe_name}.png")
        if mpi_rank == 0:
            os.replace(results_path, superseded)
            for _src in (_old_final, _old_plot):
                if os.path.exists(_src):
                    _root, _ext = os.path.splitext(_src)
                    os.replace(_src, f"{_root}_superseded_{_stamp}{_ext}")
        master_print(f"  [refit] {results_path} was fitted from "
                     f"{'bundle ' + recorded if recorded else 'an unrecorded (pre-2026-10) bundle'}; "
                     f"current bundle is {input_sha}. Old results, final bundle and plot kept "
                     f"with suffix _superseded_{_stamp}.")

    # A checkpoint is only resumable if it was started from the current bundle
    # (sidecar <checkpoint>.input_sha written at the fresh start); otherwise it is
    # set aside and the star starts fresh.
    sidecar = mcmc_checkpoint + ".input_sha"
    if os.path.exists(mcmc_checkpoint):
        try:
            started_from = open(sidecar, encoding="utf-8").read().strip()
        except OSError:
            started_from = ""
        if started_from != input_sha:
            stale = mcmc_checkpoint.replace(".phoebe", f"_stale_{time.strftime('%Y%m%d')}.phoebe")
            if mpi_rank == 0:
                os.replace(mcmc_checkpoint, stale)
            master_print(f"  [resume] checkpoint was started from "
                         f"{'bundle ' + started_from if started_from else 'an unrecorded bundle'}, "
                         f"not the current {input_sha} — set aside as {stale}; starting fresh.")

    resuming = os.path.exists(mcmc_checkpoint)
    if resuming:
        # Resume path: the checkpoint already holds the configured bundle, priors,
        # solver and the partial chain. Restore it and rebuild only the cheap
        # morphology-derived metadata; the 5A–5E setup and Nelder-Mead pre-wash
        # are skipped because that state is baked into the checkpoint.
        bundle = phoebe.load(mcmc_checkpoint)
        free_params, is_contact_binary, n_free_params = derive_free_params_meta(
            star_name, variable_type,
        )
        already = completed_iterations(bundle, "best_solution")
        master_print(f"  [resume] Loaded {mcmc_checkpoint} — {already}/{MCMC_MAX_NITERS} "
                     "iterations already complete.")
    else:
        already   = 0
        _setup_t0 = time.time()
        prepared  = prepare_bundle_for_mcmc(star_name, variable_type, bundle_path)
        setup_seconds = time.time() - _setup_t0
        if prepared is None:
            run_tracker.star_end(star_name, "skipped", reason="bundle not found")
            continue
        bundle, free_params, is_contact_binary, n_free_params = prepared
        if mpi_rank == 0:
            with open(sidecar, "w", encoding="utf-8") as fh:
                fh.write(input_sha + "\n")

    # ── 5F: STAGE 2 — CHUNKED MCMC WITH CHECKPOINTING ────────────────────────
    # Run the sampler in MCMC_CHECKPOINT_NITERS-sized chunks, saving the bundle
    # (chains included) after each chunk. `continue_from` extends the same chain
    # across chunks so a stopped run resumes mid-chain instead of restarting.
    #
    # IMPORTANT (PHOEBE continue_from quirk): a continued chunk re-uses the last
    # sample as its starting state, so it appends only (niters − 1) NEW steps, not
    # niters. The shortfall accumulates and, for the final 1-step chunk, a continued
    # chunk would append 0 steps — an infinite loop. We therefore stop as soon as a
    # chunk fails to advance the chain (treating it as effectively complete), rather
    # than demanding an exact step count.
    #
    # `done` is a rank-0-LOCAL counter. Under PHOEBE's MPI model only rank 0 runs
    # this script; rank>0 processes are held in PHOEBE's own worker loop from import
    # time, waiting for command-dict broadcasts. A user-level mpi_comm.bcast() here
    # is therefore FATAL: the workers receive the bare int as their next "packet"
    # and crash on packet.get() ("'int' object has no attribute 'get'"). run_solver
    # coordinates the workers internally, so no user collective call is needed.
    done = completed_iterations(bundle, "best_solution")
    mcmc_failed = False
    converged_early = False

    # Adaptive convergence gate (v1): run chunks up to the ceiling MCMC_MAX_NITERS,
    # but stop as soon as every sampled parameter clears both the 50τ and R̂<1.1
    # tests once past the floor MCMC_MIN_NITERS. In fixed mode MIN=MAX=MCMC_NITERS,
    # so this reduces exactly to the former fixed-length behaviour.
    chunk_seconds: list = []
    chunk_steps: list = []
    mcmc_error_text = ""
    # Failed-model baseline: carried over on a resume; 0 on a fresh star (no solution
    # exists yet, so reading it would return "unavailable").
    failed_before = read_failed_models(bundle)[0] if done else 0
    while done < MCMC_MAX_NITERS:
        prev_done = done
        # +1 compensates for the continue_from overlap (a resumed chunk re-uses the
        # last sample, yielding n−1 net new steps). Adding it to BOTH the cadence cap
        # and `remaining` makes every intermediate chunk land on a round 250-net
        # boundary, not 249 (Entry 24 checkpoint off-by-one fix).
        remaining = MCMC_MAX_NITERS - done
        n_this    = min(MCMC_CHECKPOINT_NITERS + (0 if done == 0 else 1),
                        remaining              + (0 if done == 0 else 1))
        # context="solver" is REQUIRED: once a solution exists, the twigs
        # "niters"/"continue_from" also match parameters copied into the solution,
        # making a bare solver= filter ambiguous (ValueError: 2 results found).
        bundle.set_value("continue_from", solver="mcmc_sampler", context="solver",
                         value="None" if done == 0 else "best_solution")
        bundle.set_value("niters", solver="mcmc_sampler", context="solver", value=n_this)
        try:
            master_print(f"  [MCMC] chunk → +{n_this} iters "
                         f"(from {done}/{MCMC_MAX_NITERS})...")
            _chunk_t0 = time.time()
            bundle.run_solver("mcmc_sampler", solution="best_solution", overwrite=True)
            chunk_seconds.append(time.time() - _chunk_t0)
        except Exception as mcmc_error:
            master_print(f"  MCMC error: {mcmc_error}")
            mcmc_failed = True
            mcmc_error_text = f"MCMC error: {mcmc_error}"
            run_tracker.count_error("MCMC error")
            break

        # Atomic save: write a temporary file and rename it over the checkpoint, so
        # a crash mid-write can never leave a truncated checkpoint behind.
        _tmp_checkpoint = mcmc_checkpoint + ".tmp"
        bundle.save(_tmp_checkpoint)
        os.replace(_tmp_checkpoint, mcmc_checkpoint)
        done = completed_iterations(bundle, "best_solution")
        master_print(f"  [checkpoint] {done}/{MCMC_MAX_NITERS} iters saved → {mcmc_checkpoint}")

        # ── 5F-b: per-chunk statistics + ETA (Runs/run_stats/05Optimization_chunks.csv)
        # Bookkeeping only: wrapped so that no fault here can stop the sampler.
        # Acceptance is measured from the chain (walker moved between steps), NOT from
        # PHOEBE's `acceptance_fractions`, which is wrong after continue_from.
        try:
            chunk_steps.append(done - prev_done)
            _new_steps = chunk_steps[-1]
            _rate_s    = sum(chunk_seconds[-3:]) / max(1, sum(chunk_steps[-3:]))
            _acc_mean, _acc_min = runstats.chain_acceptance(
                bundle.get_value("samples", solution="best_solution"),
                prev_done, done, MCMC_N_WALKERS)
            _failed_cum, _failed_top = read_failed_models(bundle)
            _failed_new = (_failed_cum - failed_before
                           if isinstance(_failed_cum, int) and isinstance(failed_before, int)
                           else "")
            if isinstance(_failed_cum, int):
                failed_before = _failed_cum
            _eta_star  = runstats.eta_seconds(_rate_s, MCMC_MAX_NITERS - done)
            _n_pending = sum(target_pending[target_index + 1:])
            _done_secs = run_tracker.done_star_seconds()
            if _done_secs:
                _per_star, _basis = statistics.median(_done_secs), "median completed star"
            else:
                _per_star = (time.time() - star_t0) + _eta_star
                _basis    = "assuming each is like this star"
            _eta_batch = _eta_star + _n_pending * _per_star
            run_tracker.chunk_row(
                star_name, step_from=prev_done, step_to=done, steps=_new_steps,
                chunk_seconds=chunk_seconds[-1],
                s_per_iter=chunk_seconds[-1] / _new_steps if _new_steps else "",
                accept_mean=_acc_mean, accept_min=_acc_min,
                failed_new=_failed_new, failed_cum=_failed_cum,
                failed_frac_chunk=(_failed_new / (_new_steps * MCMC_N_WALKERS)
                                   if isinstance(_failed_new, int) and _new_steps else ""),
                failed_top=_failed_top, eta_star_s=_eta_star, eta_batch_s=_eta_batch,
            )
            _gate_note = (f"; gate may stop from step {MCMC_MIN_NITERS}"
                          if not MCMC_FIXED_MODE and MCMC_MIN_NITERS < MCMC_MAX_NITERS
                          else "")
            master_print(
                f"  [eta] {done}/{MCMC_MAX_NITERS} steps · last {len(chunk_seconds[-3:])} "
                f"chunk(s) {_rate_s:.1f} s/it · accept {_acc_mean:.2f} · failed models "
                f"+{_failed_new} · star ≤ {runstats.format_hms(_eta_star)} (ceiling"
                f"{_gate_note}) · batch ≈ {runstats.format_hms(_eta_batch)} "
                f"({_n_pending} more star(s) pending, {_basis}; rough)")
        except Exception as stats_error:
            master_print(f"  [eta] chunk statistics unavailable: {stats_error}")

        if done <= prev_done:
            # No forward progress (continue_from overlap at the tail): stop here and
            # treat the chain as complete rather than spinning forever.
            master_print(f"  [checkpoint] chain not advancing past {done}/"
                         f"{MCMC_MAX_NITERS}; treating as complete.")
            break

        # Convergence gate: never test before the floor; stop the moment every
        # sampled parameter is genuinely converged. Multimodal / prior-railed
        # parameters keep R̂ above threshold and simply run to the ceiling.
        if done >= MCMC_MIN_NITERS and chains_meet_convergence(bundle, done):
            master_print(f"  [converge] all sampled parameters meet "
                         f"{CONVERGENCE_MULTIPLIER}τ & R̂<{RHAT_CONVERGENCE_THRESHOLD} "
                         f"at {done} steps — stopping early.")
            converged_early = True
            break

    if mcmc_failed and done == 0:
        master_print("  No usable samples produced — skipping target.")
        run_tracker.star_end(star_name, "failed", reason="MCMC produced no samples",
                             setup_seconds=setup_seconds, nm_seconds=last_nm_seconds,
                             resumed_from=already)
        continue
    if converged_early:
        master_print(f"  MCMC sampling complete (converged at {done} steps).")
    elif not MCMC_FIXED_MODE and done >= MCMC_MAX_NITERS:
        master_print(f"  MCMC sampling complete (reached ceiling {MCMC_MAX_NITERS} "
                     "without full convergence — see diagnostics below).")
    else:
        master_print("  MCMC sampling complete.")

    # ── 5G: AUTOCORRELATION CONVERGENCE DIAGNOSTICS ───────────────────────────
    master_print("\n  [5G] Convergence diagnostics — Integrated Autocorrelation Time τ...")
    try:
        # emcee.autocorr.integrated_time wants time on axis 0 (PHOEBE's native
        # (n_steps, n_walkers, n_params) layout); steps_first guards a transposed one.
        raw_chains = steps_first(bundle.get_value("samples", solution="best_solution"))
        burn_used  = adaptive_burn_in(raw_chains)
        tau_values = assess_autocorrelation_convergence(raw_chains[burn_used:])
        master_print(f"  burn-in used: {burn_used} steps "
                     f"(max({MCMC_BURN_IN_STEPS}, 2·τ_max)); τ measured after it")

        master_print(f"  {'Parameter':<40} {'τ (steps)':>12}  {'Status'}")
        master_print("  " + "─" * 70)
        for i, param_twig in enumerate(free_params.keys()):
            tau_val = tau_values[i] if isinstance(tau_values, np.ndarray) else np.inf
            status  = "converged" if (done > CONVERGENCE_MULTIPLIER * tau_val) else "under-converged"
            master_print(f"  {param_twig:<40} {tau_val:>12.1f}  {status}")
    except Exception as autocorr_error:
        master_print(f"  Autocorrelation diagnostic failed: {autocorr_error}")

    # ── 5H: ADOPT SOLUTION & FINAL COMPUTE ───────────────────────────────────
    try:
        burn_used
    except NameError:
        burn_used = MCMC_BURN_IN_STEPS        # τ diagnostic failed above
    master_print(f"\n  [5H] Adopting posterior median into physical parameters "
                 f"(burn-in {burn_used})...")
    try:
        bundle.adopt_solution(
            "best_solution",
            burnin=burn_used,
            adopt_values=True,
            adopt_distributions=True,
        )
    except Exception as adopt_error:
        # Never write results from a bundle that still holds the Nelder-Mead values
        # with posterior σ attached (Entry 76): fail the star instead.
        master_print(f"  Adoption error: {adopt_error} — star NOT written.")
        run_tracker.star_end(star_name, "failed", reason=f"adopt_solution error: {adopt_error}",
                             setup_seconds=setup_seconds, nm_seconds=last_nm_seconds,
                             resumed_from=already, final_steps=done,
                             steps_this_session=done - already, mcmc_seconds=sum(chunk_seconds))
        continue

    try:
        # Wilson irradiation accounts for the large mutual radiative flux exchange
        # inside the shared envelope of contact binaries, which significantly
        # shapes the out-of-eclipse maxima. Horvat is used for detached systems.
        irrad_method = "wilson" if is_contact_binary else "horvat"
        bundle.run_compute(
            model="solution_model", overwrite=True,
            irrad_method=irrad_method, distortion_method="roche",
        )
    except Exception as compute_error:
        master_print(f"  Final compute error: {compute_error}")
        run_tracker.star_end(star_name, "failed",
                             reason=f"final compute error: {compute_error}",
                             setup_seconds=setup_seconds, nm_seconds=last_nm_seconds,
                             resumed_from=already, final_steps=done,
                             steps_this_session=done - already,
                             mcmc_seconds=sum(chunk_seconds))
        continue

    chi2, chi2_red, n_obs = compute_reduced_chi2(bundle, n_free_params)
    master_print(f"\n  Final reduced χ² = {chi2_red:.4f}  at the marginal medians  "
                 f"(N = {n_obs}, dof = {n_obs - n_free_params})")

    # χ² at the maximum-posterior SAMPLE as well: for correlated or railed
    # parameters the vector of marginal medians can sit off the posterior ridge
    # and inflate χ² (Entry 76). err_scale uses the smaller (best-fit) value.
    chi2_red_map = np.nan
    try:
        _chain = steps_first(bundle.get_value("samples", solution="best_solution"))
        _lnp   = np.asarray(bundle.get_value("lnprobabilities", solution="best_solution"), float)
        if _lnp.shape[0] != _chain.shape[0]:
            _lnp = _lnp.T
        _post  = _lnp[burn_used:]
        _k     = np.unravel_index(np.nanargmax(_post), _post.shape)
        _best  = _chain[burn_used:][_k[0], _k[1], :]
        _twigs = list(bundle.get_value("fitted_twigs", solution="best_solution"))
        _saved = {t: bundle.get_value(t, context=["component", "dataset"]) for t in _twigs}
        for t, v in zip(_twigs, _best):
            bundle.set_value(t, context=["component", "dataset"], value=float(v))
        bundle.run_compute(model="map_model", overwrite=True)
        _obs = bundle.get_value("fluxes", dataset="lc01", context="dataset")
        _sig = bundle.get_value("sigmas", dataset="lc01", context="dataset")
        _syn = bundle.get_value("fluxes", dataset="lc01", model="map_model")
        chi2_red_map = float(np.sum(((_obs - _syn) / _sig) ** 2) / max(1, len(_obs) - n_free_params))
        for t, v in _saved.items():
            bundle.set_value(t, context=["component", "dataset"], value=v)
        master_print(f"  Reduced χ² at the maximum-posterior sample = {chi2_red_map:.4f}")
    except Exception as map_error:
        master_print(f"  [χ² MAP] unavailable: {map_error}")

    # ── 5I: GELMAN-RUBIN R̂ CONVERGENCE CHECK ─────────────────────────────────
    # The stop REASON is already printed above ("converged at N" vs "reached
    # ceiling"), but it is ~40 lines earlier and the per-parameter Status column
    # below is computed from R̂ ALONE. A star that ran out of budget with τ far
    # short of the gate therefore reads as "CONVERGED" on every row. Repeating
    # the reason in this header keeps the table from being quoted out of context
    # — three of the four batch-4 stars stopped at the ceiling while every row
    # said CONVERGED. See Pipeline_Log Entries 60/62/63.
    if MCMC_FIXED_MODE:
        _stop_reason = f"fixed-length mode, {done} steps — gate NOT applied"
    elif converged_early:
        _stop_reason = f"gate satisfied at {done} steps"
    else:
        _stop_reason = (f"stopped at ceiling {MCMC_MAX_NITERS} — gate NOT satisfied; "
                        "see τ above")
    master_print(f"\n  [5I] Gelman-Rubin R̂ convergence diagnostics  [{_stop_reason}]:")
    try:
        # R̂ from the shared helper (the same one the convergence gate uses), so the
        # post-hoc report and the stop criterion can never diverge.
        raw_chains   = np.asarray(bundle.get_value("samples", solution="best_solution"))
        fitted_twigs = bundle.get_value("fitted_twigs", solution="best_solution")

        # Orient steps-first so the burn-in slice below is unambiguous.
        oriented = raw_chains
        if oriented.ndim == 3 and oriented.shape[0] == MCMC_N_WALKERS \
                and oriented.shape[1] != MCMC_N_WALKERS:
            oriented = np.swapaxes(oriented, 0, 1)
        trim = min(burn_used, max(0, oriented.shape[0] - 2))

        # Both variants are reported: R̂_full is what this pipeline printed through
        # batch 4 (keeping older run logs comparable), R̂_trim is the honest figure
        # the gate now decides on. See Pipeline_Log Entry 47.
        r_hats_full = gelman_rubin_rhat(oriented)
        r_hats      = gelman_rubin_rhat(oriented[trim:])

        master_print(f"  {'Parameter':<40} {'R̂_full':>8} {'R̂_trim':>8}  {'Status (R̂ only)'}")
        master_print("  " + "─" * 70)
        for i, param_name in enumerate(fitted_twigs):
            r_hat  = float(r_hats[i]) if i < len(r_hats) else float("inf")
            r_full = float(r_hats_full[i]) if i < len(r_hats_full) else float("inf")
            status = "CONVERGED" if r_hat < RHAT_CONVERGENCE_THRESHOLD else "UNCONVERGED"
            if r_full < RHAT_CONVERGENCE_THRESHOLD <= r_hat:
                status += "  (burn-in masked)"
            master_print(f"  {param_name:<40} {r_full:>8.4f} {r_hat:>8.4f}  {status}")
    except Exception as rhat_error:
        master_print(f"  Gelman-Rubin diagnostic failed: {rhat_error}")

    # ── 5I-b: PER-WALKER CHAIN HEALTH + TRACE PLOT ────────────────────────────
    # R̂ above is an ensemble aggregate and can be misleading on its own: a single
    # frozen walker inflates it, and burn-in scatter used to mask it (Entry 47).
    # This scores the chain walker-by-walker using the same thresholds as the
    # standalone audit — one definition, in Audits/audit_mcmc_chains.py — and
    # writes a trace figure for EVERY star so the chain can be inspected later
    # without re-running anything. Diagnostics must never kill a completed fit,
    # so the whole block is defensive.
    #
    # The shared audit CSVs are deliberately NOT written here: concurrent batch
    # processes would interleave appends. `python Audits/audit_mcmc_chains.py
    # <catalogue> --per-walker` regenerates them in one pass instead.
    if chain_audit is not None:
        master_print("\n  [5I-b] Per-walker chain health:")
        try:
            lnprob = np.asarray(
                bundle.get_value("lnprobabilities", solution="best_solution"),
                dtype=float)
            audit = chain_audit.audit_chain_arrays(
                star_name,
                bundle.get_value("samples", solution="best_solution"),
                lnprob,
                bundle.get_value("fitted_twigs", solution="best_solution"),
                burn=burn_used,
                chi2_red=chi2_red,
            )
            star_audit = audit
            master_print(f"  verdict: {' + '.join(audit['verdicts'])}")
            if audit.get("detail"):
                for line in audit["detail"].split("; "):
                    master_print(f"    - {line}")

            if mpi_rank == 0:
                trace_path = chain_audit.plot_walker_traces(
                    audit, out_dir=paths.PLOT_MCMC_DIR)
                if trace_path:
                    master_print(f"  Chain plot saved:   {trace_path}")
        except Exception as chain_error:
            master_print(f"  Chain health diagnostic failed: {chain_error}")

    # ── 5J: RESULT EXTRACTION ─────────────────────────────────────────────────
    # Summarise the post-burn-in chain first, then attach ±σ to the adopted medians.
    posterior = compute_posterior_summary(bundle, "best_solution", burn_used)
    results   = extract_best_fit_parameters(bundle, free_params, posterior)

    # ── 5J-a: RAILING FLAGS ──────────────────────────────────────────────────
    # A parameter is RAILED when its median lies within 2 raw σ (or 0.1 % of the
    # range) of a prior bound: its interval is then truncated by the prior, not
    # by the data, and must be reported as a limit (Entry 76: V358 Cep i and r₂,
    # FI Boo T₂/T₁, V857 Her / TV Mus / FN Cam f, GQ Cnc / V1128 Tau i).
    _sma = bundle.get_value("sma", component="binary", context="component")
    _t1  = results["T1_K"][0]
    result_bounds: dict[str, tuple[float, float]] = {}
    for _key, _twig, _conv in (("T2_K", "teff@secondary", 1.0),
                               ("T2_over_T1", "teff@secondary", 1.0 / _t1),
                               ("incl_deg", "incl@binary", 1.0),
                               ("q", "q@binary", 1.0),
                               ("fillout_factor", "fillout_factor@contact_envelope", 1.0),
                               ("r1_frac", "requiv@primary", 1.0 / _sma),
                               ("r2_frac", "requiv@secondary", 1.0 / _sma)):
        if _twig in free_params and _key in results:
            _lo, _hi = prior_bounds(bundle, _twig, free_params[_twig][:2])
            result_bounds[_key] = (_lo * _conv, _hi * _conv)
    railed = []
    for _key, (_lo, _hi) in result_bounds.items():
        _v, _sm, _sp = results[_key]
        if not np.isfinite(_v):
            continue
        _tol_lo = max(2.0 * (_sm if np.isfinite(_sm) else 0.0), 1e-3 * (_hi - _lo))
        _tol_hi = max(2.0 * (_sp if np.isfinite(_sp) else 0.0), 1e-3 * (_hi - _lo))
        if _v - _lo <= _tol_lo:
            railed.append(f"{_key}@low")
        elif _hi - _v <= _tol_hi:
            railed.append(f"{_key}@high")
    if railed:
        master_print(f"  [railed] {', '.join(railed)} — interval truncated by the prior; "
                     "report as a limit.")

    # ── 5J-b: MORPHOLOGY PROVENANCE + CONTACT-MODEL VALIDITY FLAG ─────────────
    # morphology_source records WHY the model was contact/detached (GCVS label vs a
    # MORPHOLOGY_OVERRIDE). For contact stars, assess_contact_validity flags whether
    # the fit actually wants to be detached — a heads-up for human review, never an
    # automatic switch. (Runs on rank 0 only, like the rest of this post-run block.)
    morphology_src  = morphology_source(star_name)
    morphology_kind = resolve_morphology_kind(star_name, variable_type)
    if is_contact_binary:
        morphology_flag = assess_contact_validity(bundle)
        if morphology_flag == "RECONSIDER_DETACHED":
            master_print("\n  " + "!" * 72)
            master_print(f"  [morphology] {star_name}: adopted contact geometry FAILS "
                         "run_checks (requiv near critical).")
            master_print("  VERIFY against the folded light curve FIRST — a genuine EW curve "
                         "(continuous")
            master_print("  variation / O'Connell) stays CONTACT; only a detached-shaped curve "
                         "(flat")
            master_print("  out-of-eclipse) warrants the switch. The detector cannot tell these "
                         "apart.")
            master_print(f"  If detached: add \"{star_name}\": DETACHED to MORPHOLOGY_OVERRIDE "
                         "(morphology.py),")
            master_print(f"  re-run Script 04, clear {results_path}, re-run Script 05.")
            master_print("  " + "!" * 72)
    else:
        morphology_flag = ("SEMIDETACHED_MODEL" if morphology_kind == SEMIDETACHED
                           else "DETACHED_MODEL")

    master_print("\n  ─── FINAL PARAMETER POSTERIOR ──────────────────────────────")
    for key, (value, sig_minus, sig_plus) in results.items():
        if np.isfinite(value):
            label, unit = PARAM_DISPLAY_LABELS.get(key, (key, ""))
            if np.isfinite(sig_minus) and np.isfinite(sig_plus):
                master_print(f"  {label:<35} {value:>10.4f}  (−{sig_minus:.4f} / +{sig_plus:.4f})  {unit}")
            else:
                master_print(f"  {label:<35} {value:>10.4f}  {unit}")

    # ── 5K: FILE I/O (RANK 0 ONLY) ────────────────────────────────────────────
    # MPI worker ranks must not write files; simultaneous writes would corrupt
    # the output. All I/O is gated on rank 0.
    # Order matters: the final bundle and plot are written BEFORE the results CSV.
    # A results file is what marks a star complete, so it must never exist without
    # its final bundle (before 2026-10 a crash between the two left a results file
    # with a stale or missing _final, and the star was then skipped forever).
    err_scale = 1.0
    if mpi_rank == 0:
        output_ok = False
        try:
            obs_flux  = bundle.get_value("fluxes", dataset="lc01", context="dataset")
            obs_times = bundle.get_value("times",  dataset="lc01", context="dataset")
            obs_sigma = bundle.get_value("sigmas", dataset="lc01", context="dataset")
            syn_flux  = bundle.get_value("fluxes", dataset="lc01", model="solution_model")
            syn_times = bundle.get_value("times",  dataset="lc01", model="solution_model")

            period_final  = bundle.get_value("period@binary")
            t0_final      = bundle.get_value("t0_supconj@binary")
            obs_phase     = ((obs_times - t0_final) / period_final + 0.5) % 1.0 - 0.5
            syn_phase     = ((syn_times - t0_final) / period_final + 0.5) % 1.0 - 0.5
            sort_idx      = np.argsort(syn_phase)
            residuals     = obs_flux - np.interp(
                obs_phase, syn_phase[sort_idx], syn_flux[sort_idx],
            )

            fig, (ax_lc, ax_res) = plt.subplots(
                2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]},
            )
            fig.suptitle(
                f"{star_name} — MCMC Optimised Fit  (χ²_red = {chi2_red:.2f})",
                fontsize=14, fontweight="bold",
            )

            ax_lc.errorbar(obs_phase, obs_flux, yerr=obs_sigma, fmt="k.", ms=6,
                           alpha=0.5, label="TESS Observations")
            ax_lc.plot(syn_phase[sort_idx], syn_flux[sort_idx], "r-", lw=2.5,
                       label="Optimised Model")
            t2_val = results.get("T2_K", (np.nan, np.nan, np.nan))[0]
            param_annotation = (
                f"T₁ = {results['T1_K'][0]:.0f} K  |  "
                f"T₂ = {t2_val:.0f} K  |  "
                f"i = {results['incl_deg'][0]:.1f}°  |  {morphology_kind}"
            )
            ax_lc.text(0.02, 0.04, param_annotation, transform=ax_lc.transAxes,
                       fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9))
            ax_lc.set_ylabel("Normalised Flux")
            ax_lc.legend(loc="upper right")
            ax_lc.grid(alpha=0.3)

            ax_res.plot(obs_phase, residuals, ".", ms=5, color="steelblue", alpha=0.7)
            ax_res.axhline(0.0, color="r", lw=1.5, ls="--")
            ax_res.set_xlabel("Orbital Phase φ")
            ax_res.set_ylabel("Residuals")
            ax_res.grid(alpha=0.3)

            plt.tight_layout()
            plot_path = os.path.join(paths.PLOT_FINAL_DIR, f"phoebe_final_{safe_name}.png")
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            plt.close()

            final_bundle_path = os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name}_final.phoebe")
            bundle.save(final_bundle_path)
            master_print(f"  Final bundle saved: {final_bundle_path}")
            master_print(f"  Final plot saved:   {plot_path}")
            output_ok = True
        except Exception as output_error:
            master_print(f"  Could not generate final plot / bundle: {output_error}")
            run_tracker.count_error("final plot/bundle")

        if output_ok:
            # Rescale the posterior uncertainties so the reduced χ² is effectively 1.
            # When the formal photometric errors underestimate unmodelled systematics
            # (starspots, the O'Connell effect), the raw posterior widths are
            # unrealistically tight; multiplying them by sqrt(χ²_red) is the standard
            # correction (Kallrath & Milone 2009). Only inflates (never shrinks).
            # χ²_red here is the BEST-FIT value (maximum-posterior sample when
            # available, else the marginal-median model).
            _chi2_for_scale = chi2_red_map if np.isfinite(chi2_red_map) else chi2_red
            err_scale = (max(1.0, float(np.sqrt(_chi2_for_scale)))
                         if np.isfinite(_chi2_for_scale) else 1.0)
            flags = list(railed)
            clipped = []
            try:
                _hdr = phoebe_light_curve_flags(safe_name)
            except Exception:
                _hdr = {}
            flags += _hdr.get("flags", [])
            if morphology_kind == SEMIDETACHED:
                flags.append("r2_constrained_to_roche_lobe")
            with open(results_path, "w", newline="", encoding="utf-8") as results_file:
                writer = csv.writer(results_file)
                writer.writerow(["Parameter", "Symbol", "Value", "Sigma_minus", "Sigma_plus", "Unit"])
                for key, (value, sig_minus, sig_plus) in results.items():
                    label, unit = PARAM_DISPLAY_LABELS.get(key, (key, ""))
                    sm = sig_minus * err_scale if np.isfinite(sig_minus) else sig_minus
                    sp = sig_plus  * err_scale if np.isfinite(sig_plus)  else sig_plus
                    # Scaled intervals must not cross the prior bounds (they did:
                    # i up to 90.6°, r₂ past the lobe cap — Entry 76).
                    if key in result_bounds and np.isfinite(value):
                        _lo, _hi = result_bounds[key]
                        if np.isfinite(sm) and value - sm < _lo:
                            sm = value - _lo
                            clipped.append(f"{key}-")
                        if np.isfinite(sp) and value + sp > _hi:
                            sp = _hi - value
                            clipped.append(f"{key}+")
                    writer.writerow([label, key, value, sm, sp, unit])
                if clipped:
                    flags.append("interval_clipped_at_bound:" + ",".join(clipped))
                writer.writerow(["Reduced chi-squared", "chi2_red", chi2_red, "", "", "—"])
                writer.writerow(["Reduced chi-squared at max-posterior sample", "chi2_red_map",
                                 chi2_red_map, "", "", "—"])
                writer.writerow(["Error scale (sqrt best-fit reduced chi2)", "err_scale",
                                 err_scale, "", "", "—"])
                writer.writerow(["Morphology flag", "morphology_flag", morphology_flag, "", "", "—"])
                writer.writerow(["Morphology source", "morphology_source", morphology_src, "", "", "—"])
                writer.writerow(["Morphology", "morphology", morphology_kind, "", "", "—"])
                # Provenance (Entry 76): which code, input and chain produced this row.
                writer.writerow(["Run id", "run_id", run_tracker.run_id, "", "", "—"])
                writer.writerow(["Code sha", "code_sha", run_tracker.code_sha, "", "", "—"])
                writer.writerow(["Input bundle sha", "input_bundle_sha", input_sha, "", "", "—"])
                writer.writerow(["Chain steps", "steps", done, "", "", "—"])
                writer.writerow(["Burn-in steps", "burn_in", burn_used, "", "", "—"])
                writer.writerow(["Stop reason", "stop_reason", _stop_reason, "", "", "—"])
                writer.writerow(["Flux source", "flux_source", _hdr.get("FLUXSRC", ""), "", "", "—"])
                writer.writerow(["Flags", "flags", "; ".join(flags), "", "", "—"])
            master_print(f"\n  Results saved: {results_path}"
                         + (f"  [flags: {'; '.join(flags)}]" if flags else ""))

            # Target finished — drop the mid-run checkpoint (and its sidecar) so a
            # future relaunch is recognised as complete rather than resumed.
            for _f in (mcmc_checkpoint, sidecar):
                if os.path.exists(_f):
                    os.remove(_f)
        else:
            master_print("  Results NOT written (final bundle/plot failed); the checkpoint is "
                         "kept, so a relaunch resumes and retries.")

    # ── 5L: PER-STAR TIMING ROW (Runs/run_stats/05Optimization_stars.csv) ─────
    # s_per_iter covers only steps run in THIS session (a resumed star's earlier
    # steps were timed by a previous process — see resumed_from).
    _steps_now = done - already
    _mcmc_s    = sum(chunk_seconds)
    # Proposals whose forward model failed (lnprob = −inf), keyed by PHOEBE's error
    # message. PHOEBE carries this dict across continue_from, so it covers the whole
    # chain, resumed sessions included. Distinct from acceptance: a walker that does
    # not move may have been Metropolis-rejected with a perfectly valid model.
    _failed_n, _failed_top = read_failed_models(bundle)
    _failed_frac = (_failed_n / (done * MCMC_N_WALKERS)
                    if isinstance(_failed_n, int) and done else "")
    run_tracker.star_end(
        star_name, "done", reason=mcmc_error_text,
        setup_seconds=setup_seconds, nm_seconds=last_nm_seconds,
        resumed_from=already, steps_this_session=_steps_now, final_steps=done,
        n_chunks=len(chunk_seconds), mcmc_seconds=_mcmc_s,
        s_per_iter=(_mcmc_s / _steps_now) if _steps_now > 0 else "",
        chunk_first_s=chunk_seconds[0] if chunk_seconds else "",
        chunk_min_s=min(chunk_seconds) if chunk_seconds else "",
        chunk_max_s=max(chunk_seconds) if chunk_seconds else "",
        chunk_last_s=chunk_seconds[-1] if chunk_seconds else "",
        stop_reason=_stop_reason, chi2_red=chi2_red,
        err_scale=err_scale,
        median_accept=star_audit["median_accept"] if star_audit else "",
        min_accept=star_audit["min_accept"] if star_audit else "",
        failed_models=_failed_n, failed_frac=_failed_frac, failed_top_reason=_failed_top,
        verdicts=" + ".join(star_audit["verdicts"]) if star_audit else "",
        morphology_flag=morphology_flag,
    )

master_print("\n" + "=" * 70)
master_print("SCRIPT 05 COMPLETE — Statistical optimisation finished for all targets.")
master_print("=" * 70)
