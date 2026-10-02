"""
================================================================================
AUDIT — PER-WALKER MCMC CHAIN HEALTH (POST-SCRIPT-05 QA GATE)
================================================================================
Summary:
    Inspects the INDIVIDUAL WALKERS of each star's emcee ensemble, which no
    stage of the pipeline does. Script 05 reduces the ensemble to aggregate
    statistics only — Gelman-Rubin R̂ averages over walkers, emcee's
    integrated_time() takes a median over walkers, and compute_posterior_summary()
    flattens walkers away with reshape(-1, n_params) — so a single pathological
    walker is invisible in every number the pipeline reports.

    Pipeline_Log Entry 45 is the motivating case: GQ Cnc's R̂ = 1.41 on the
    inclination was recorded as a multimodal posterior (the "RT Hyi pattern")
    when it was in fact ONE frozen walker (walker 8, last accepted move at step
    929 of 6000) dragging the between-walker variance upward. Excluding it drops
    R̂ to ~1.01. The diagnosis required plotting the raw per-walker trace by
    hand; this audit makes that check routine and names the four distinct
    failure signatures apart, because they call for different responses:

      DEAD_WALKER          a few walkers frozen, the rest healthy. Excluding
                           them is defensible (the GQ Cnc remedy).
      ACCEPTANCE_COLLAPSE  the WHOLE ensemble barely moves. Excluding walkers
                           does not help and can make R̂ worse; the reported
                           posterior WIDTHS are not trustworthy at all.
      MULTIMODAL           walkers mix normally but settle in ln-probability
                           basins far apart — genuine posterior structure.
      BURNIN_MASKED        R̂ passes on the full chain but fails once burn-in
                           is trimmed, i.e. the convergence gate was optimistic.

    Two caveats the numbers here are meant to expose rather than hide:

    (1) Script 05 evaluates R̂ on the chain INCLUDING burn-in (the `chains`
        passed to gelman_rubin_rhat() in chains_meet_convergence() and in the
        Section 5I table are untrimmed) while the posterior it adopts IS
        trimmed. Burn-in scatter inflates the within-chain variance W, and
        R̂ = sqrt(V̂/W), so the gate is biased toward declaring convergence.
        Both variants are reported side by side: rhat_full reproduces what the
        pipeline printed, rhat_postburnin is the honest figure.

    (2) Acceptance collapse tracks χ²_red, because the sampler runs against the
        UN-rescaled TESS sigmas. A star with χ²_red = 28 presents a likelihood
        ~28x too sharp for the stretch move, almost every proposal is rejected,
        and walkers pin to plateaus of the forward model. err_scale = sqrt(χ²_red)
        is then applied to the error bars after the fact, which rescales the
        output but cannot recover sampling that never happened. So an
        ACCEPTANCE_COLLAPSE star's quoted sigmas are lower bounds, not
        uncertainties, and the audit flags them as such.

    When ACCEPTANCE_COLLAPSE and MULTIMODAL fire together, read it as "walkers
    trapped in separate basins", not as established multimodality: a sampler
    that rejects ~99% of proposals cannot cross between modes, so genuine
    structure and mere trapping are not distinguishable from the chain alone.

Inputs:
    data/bundles/bundle_<star>_final.phoebe  — Script 05 solutions. Each carries
        samples        (n_steps, n_walkers, n_params)
        lnprobabilities (n_steps, n_walkers)
        fitted_twigs   parameter names
      under solution="best_solution". Stars without a final bundle (Script 05
      not yet run) are reported as missing and skipped, never raised.
    data/results/results_<star>.csv          — optional, for the χ²_red join.

Outputs:
    Fixed-width verdict table printed to stdout.
    Audits/mcmc_chain_audit.csv    — one row per star.
    Audits/mcmc_chain_walkers.csv  — one row per star x walker  (--per-walker).
    data/plots/mcmc_diag/mcmc_chains_<star>.png — per-walker traces (--plot).

Usage:
    Run from Code/ (the Entry 39 convention). The 60-star working sample lives in
    the batch catalogues; neglected_targets.csv is the full 393-star selection,
    so pair it with --skip-missing unless the un-run stars are the point.
      python Audits/audit_mcmc_chains.py Targets/neglected_targets_batch*.csv
      python Audits/audit_mcmc_chains.py Targets/neglected_targets_batch4.csv --plot
      python Audits/audit_mcmc_chains.py Targets/neglected_targets.csv --skip-missing

Dependencies:
    pip install phoebe numpy emcee matplotlib

References:
    Gelman & Rubin 1992, Statist. Sci. 7, 457   — R̂ / potential scale reduction
    Goodman & Weare 2010, Comm. App. Math. Comp. Sci. 5, 65 — stretch move
    Foreman-Mackey et al. 2013, PASP 125, 306   — emcee, autocorrelation practice
    Hogg & Foreman-Mackey 2018, ApJS 236, 11    — MCMC diagnostics in astronomy
================================================================================
"""

import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np

import matplotlib
matplotlib.use("Agg")          # headless: must precede any pyplot/phoebe import
import matplotlib.pyplot as plt

# Audits/ sits one level below Code/, where paths.py lives.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

import emcee                                              # noqa: E402
import phoebe                                             # noqa: E402


# --- Sampler settings mirrored from 05Optimization.py -----------------------
# Kept as module constants rather than imported: 05Optimization.py executes its
# whole pipeline on import (no __main__ guard around the main loop).
MCMC_BURN_IN_STEPS = int(os.environ.get("MCMC_BURNIN", 350))
RHAT_CONVERGENCE_THRESHOLD = 1.1
CONVERGENCE_MULTIPLIER = 50

# --- Verdict thresholds, calibrated against the 25 completed stars ----------
# FROZEN_FRAC_DEAD: the measured distribution is bimodal with a clean gap —
# genuinely dead walkers sit at 0.75-0.91 of the chain, the next tier down at
# 0.36 — so 0.50 separates them without tuning.
FROZEN_FRAC_DEAD = 0.50
# Healthy ensembles accept 0.24-0.62 of proposals; collapsed ones 0.006-0.13.
ACCEPT_FRAC_COLLAPSE = 0.10
ACCEPT_FRAC_LOW = 0.20
# A ln-probability gap between walker groups this large is structure, not noise.
LNPROB_GAP_MULTIMODAL = 10.0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def gelman_rubin_rhat(chains: np.ndarray) -> np.ndarray:
    """
    R̂ per parameter for a (n_steps, n_walkers, n_params) array.

    Deliberately reproduces 05Optimization.py's gelman_rubin_rhat() arithmetic
    verbatim, including its ddof=0 variances (which understate B by a factor
    (m-1)/m = 0.93 against Gelman & Rubin's definition — a ~3% effect). Matching
    the pipeline matters more than the 3%: it lets rhat_full be checked against
    the R̂ values the batch logs actually printed, so a disagreement means this
    audit is wrong rather than the pipeline.
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 3:
        return np.full(1, np.inf)
    n_s, n_w, n_p = chains.shape
    rhats = np.full(n_p, np.inf)
    if n_s < 2 or n_w < 2:
        return rhats
    for i in range(n_p):
        pc = chains[:, :, i].T                       # -> (n_walkers, n_steps)
        within = np.mean(np.var(pc, axis=1))         # W
        between = n_s * np.var(np.mean(pc, axis=1))  # B
        if within > 0:
            var_plus = (1 - 1 / n_s) * within + (1 / n_s) * between
            rhats[i] = np.sqrt(var_plus / within)
    return rhats


def _walker_metrics(chains: np.ndarray, lnprob: np.ndarray, burn: int) -> list[dict]:
    """
    Per-walker mixing metrics. `chains` is (n_steps, n_walkers, n_params).

    A move is counted as accepted when the walker's position changes in any
    parameter; emcee keeps a rejected proposal's previous position, so an
    unchanged row is a rejection. frozen_tail is measured over the FULL chain
    (a walker that dies during burn-in still died), everything else post-burn-in.
    """
    n_steps, n_walkers, _ = chains.shape
    out = []
    for w in range(n_walkers):
        moved = np.any(np.abs(np.diff(chains[:, w, :], axis=0)) > 0, axis=1)
        idx = np.flatnonzero(moved)
        last_move = int(idx[-1]) + 1 if idx.size else 0
        frozen_tail = n_steps - 1 - last_move

        post = chains[burn:, w, :]
        moved_post = moved[burn:] if moved.size > burn else np.array([], dtype=bool)
        out.append(dict(
            walker=w,
            last_move=last_move,
            frozen_tail=frozen_tail,
            frozen_frac=frozen_tail / n_steps,
            accept_frac=float(moved_post.mean()) if moved_post.size else float("nan"),
            n_distinct=int(np.unique(post, axis=0).shape[0]),
            median_lnprob=float(np.median(lnprob[burn:, w])),
        ))

    consensus = float(np.median([m["median_lnprob"] for m in out]))
    for m in out:
        m["lnprob_offset"] = m["median_lnprob"] - consensus
    return out


def _max_group_gap(values: list[float], min_side: int = 2) -> tuple[float, int]:
    """
    Largest gap between consecutive sorted values that leaves at least
    `min_side` values on each side. Returns (gap, n_below). (0.0, 0) if the
    sample is too small to split.
    """
    vals = sorted(values)
    if len(vals) < 2 * min_side:
        return 0.0, 0
    best_gap, best_below = 0.0, 0
    for i in range(min_side - 1, len(vals) - min_side):
        gap = vals[i + 1] - vals[i]
        if gap > best_gap:
            best_gap, best_below = gap, i + 1
    return best_gap, best_below


def _read_chi2_red(star_name: str, results_dir: str) -> float | None:
    path = os.path.join(results_dir, f"results_{star_name.replace(' ', '_')}.csv")
    try:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("Symbol") == "chi2_red":
                    return float(row["Value"])
    except Exception:
        return None
    return None


def audit_chain_arrays(star_name: str,
                       chains,
                       lnprob,
                       twigs,
                       burn: int = MCMC_BURN_IN_STEPS,
                       chi2_red: float | None = None) -> dict:
    """
    Audit chains already held in memory. Returns a verdict dict; never raises.

    This is the array-level entry point, split out so 05Optimization.py can score
    a chain it has just sampled without re-loading the bundle from disk (the final
    bundle does not exist yet at that point in the run). `audit_mcmc_chains()` is
    the on-disk wrapper around it.

    `verdicts` is a LIST — the signatures are not mutually exclusive (RW Dor is
    both DEAD_WALKER and ACCEPTANCE_COLLAPSE). Uppercase entries denote a real
    problem, lowercase ones are informational, per the Audits/ convention.

    chi2_red is passed in rather than read from results_<star>.csv, because the
    caller may not have written that file yet.
    """
    result: dict = {"star": star_name, "verdicts": ["ERROR"], "detail": "", "params": {}}

    chains = np.asarray(chains, dtype=float)
    lnprob = np.asarray(lnprob, dtype=float)
    twigs = [str(t) for t in twigs]

    if chains.ndim != 3:
        result["detail"] = f"unexpected samples shape {chains.shape}"
        return result
    # Orient steps-first. n_walkers is the axis matching lnprob's walker axis,
    # which avoids 05Optimization.py's habit of comparing against a hardcoded 14.
    if chains.shape[1] != lnprob.shape[1] and chains.shape[0] == lnprob.shape[1]:
        chains = np.swapaxes(chains, 0, 1)
    if lnprob.shape[0] != chains.shape[0] and lnprob.shape[1] == chains.shape[0]:
        lnprob = lnprob.T

    n_steps, n_walkers, n_params = chains.shape
    burn = min(max(0, burn), max(0, n_steps - 2))

    walkers = _walker_metrics(chains, lnprob, burn)
    accepts = [m["accept_frac"] for m in walkers if not np.isnan(m["accept_frac"])]
    median_accept = float(np.median(accepts)) if accepts else float("nan")
    min_accept = float(np.min(accepts)) if accepts else float("nan")
    dead = [m["walker"] for m in walkers if m["frozen_frac"] > FROZEN_FRAC_DEAD]
    alive = [w for w in range(n_walkers) if w not in dead]

    # --- per-parameter R̂ triplet, tau, ESS --------------------------------
    rhat_full = gelman_rubin_rhat(chains)
    rhat_post = gelman_rubin_rhat(chains[burn:])
    rhat_excl = (gelman_rubin_rhat(chains[burn:][:, alive, :])
                 if dead and len(alive) >= 2 else rhat_post)
    try:
        # A fully frozen walker gives acf[0] == 0, so emcee's normalisation warns
        # about a 0/0 divide. That is exactly the case this audit exists to find;
        # the resulting non-finite tau is reported as tau_trustworthy=False.
        with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
            warnings.simplefilter("ignore", RuntimeWarning)
            tau = np.atleast_1d(emcee.autocorr.integrated_time(chains[burn:], tol=0))
    except Exception:
        tau = np.full(n_params, np.inf)

    n_post = n_steps - burn
    for i, twig in enumerate(twigs[:n_params]):
        x = chains[burn:, :, i]
        within_sd = float(np.sqrt(np.mean(x.var(axis=0, ddof=1))))
        between_sd = float(np.std(np.median(x, axis=0)))
        result["params"][twig] = dict(
            rhat_full=float(rhat_full[i]),
            rhat_postburnin=float(rhat_post[i]),
            rhat_excl_dead=float(rhat_excl[i]),
            tau=float(tau[i]) if i < len(tau) else float("inf"),
            ess=(n_post * n_walkers / float(tau[i])
                 if i < len(tau) and np.isfinite(tau[i]) and tau[i] > 0 else float("nan")),
            within_sd=within_sd,
            between_walker_sd=between_sd,
            between_over_within=(between_sd / within_sd if within_sd > 0 else float("inf")),
            tau_trustworthy=bool(i < len(tau) and np.isfinite(tau[i])
                                 and n_post > CONVERGENCE_MULTIPLIER * tau[i]),
        )

    # --- verdicts ----------------------------------------------------------
    verdicts: list[str] = []
    details: list[str] = []

    if dead:
        verdicts.append("DEAD_WALKER")
        worst = max((m for m in walkers if m["walker"] in dead),
                    key=lambda m: m["frozen_frac"])
        details.append(
            f"walker(s) {dead} frozen; worst w{worst['walker']} last moved at step "
            f"{worst['last_move']}/{n_steps} ({worst['frozen_frac']:.0%} of chain dead)"
        )

    collapsed = (not np.isnan(median_accept)) and median_accept < ACCEPT_FRAC_COLLAPSE
    if collapsed:
        verdicts.append("ACCEPTANCE_COLLAPSE")
        details.append(
            f"ensemble median acceptance {median_accept:.3f} — posterior WIDTHS unreliable"
        )
    elif (not np.isnan(median_accept)) and median_accept < ACCEPT_FRAC_LOW:
        verdicts.append("acceptance-low — review")
        details.append(f"ensemble median acceptance {median_accept:.3f}")

    # Multimodality is only meaningful among the walkers that are actually moving.
    live_offsets = [m["lnprob_offset"] for m in walkers if m["walker"] in alive]
    gap, n_below = _max_group_gap(live_offsets)
    if gap > LNPROB_GAP_MULTIMODAL:
        verdicts.append("MULTIMODAL")
        details.append(
            f"walker ln-prob splits {n_below}/{len(live_offsets) - n_below} across a "
            f"{gap:.1f} gap" + (" (trapping, not established structure — see acceptance)"
                                if collapsed else "")
        )

    masked = [t for t, p in result["params"].items()
              if p["rhat_full"] < RHAT_CONVERGENCE_THRESHOLD
              and p["rhat_postburnin"] >= RHAT_CONVERGENCE_THRESHOLD]
    if masked:
        verdicts.append("BURNIN_MASKED")
        details.append(
            "gate passed on full-chain R̂ but fails trimmed: "
            + ", ".join(f"{t.split('@')[0]} {result['params'][t]['rhat_full']:.2f}"
                        f"->{result['params'][t]['rhat_postburnin']:.2f}" for t in masked)
        )

    if not verdicts:
        unconverged = [t for t, p in result["params"].items()
                       if p["rhat_postburnin"] >= RHAT_CONVERGENCE_THRESHOLD]
        if unconverged:
            verdicts.append("unconverged — review")
            details.append("trimmed R̂ above threshold: " + ", ".join(
                f"{t.split('@')[0]} {result['params'][t]['rhat_postburnin']:.2f}"
                for t in unconverged))
        else:
            verdicts.append("healthy")

    result.update(
        verdicts=verdicts,
        detail="; ".join(details),
        n_steps=n_steps,
        n_walkers=n_walkers,
        burn_in=burn,
        chi2_red=chi2_red,
        median_accept=median_accept,
        min_accept=min_accept,
        worst_frozen_frac=max(m["frozen_frac"] for m in walkers),
        median_n_distinct=float(np.median([m["n_distinct"] for m in walkers])),
        dead_walkers=dead,
        posterior_width_unreliable=bool(collapsed),
        walkers=walkers,
        twigs=twigs[:n_params],
        _chains=chains,
        _lnprob=lnprob,
    )
    return result


def audit_mcmc_chains(star_name: str,
                      bundle_dir: str | None = None,
                      results_dir: str | None = None,
                      burn: int = MCMC_BURN_IN_STEPS) -> dict:
    """
    Audit one star's saved chain. Thin on-disk wrapper around
    `audit_chain_arrays()`: loads `bundle_<star>_final.phoebe`, pulls the chain
    out of it, and joins chi2_red from `results_<star>.csv`. Never raises.
    """
    bundle_dir = bundle_dir if bundle_dir is not None else paths.BUNDLE_DIR
    results_dir = results_dir if results_dir is not None else paths.RESULTS_DIR
    safe_name = star_name.replace(" ", "_")
    bundle_path = os.path.join(bundle_dir, f"bundle_{safe_name}_final.phoebe")

    result: dict = {"star": star_name, "verdicts": ["ERROR"], "detail": "", "params": {}}

    if not os.path.exists(bundle_path):
        result["verdicts"] = ["missing — Script 05 not run"]
        result["detail"] = f"no {os.path.basename(bundle_path)}"
        return result

    try:
        bundle = phoebe.load(bundle_path)
        chains = np.asarray(bundle.get_value("samples", solution="best_solution"), dtype=float)
        lnprob = np.asarray(bundle.get_value("lnprobabilities", solution="best_solution"), dtype=float)
        twigs = [str(t) for t in bundle.get_value("fitted_twigs", solution="best_solution")]
    except Exception as exc:
        result["detail"] = f"could not read chain: {exc}"
        return result

    return audit_chain_arrays(star_name, chains, lnprob, twigs, burn=burn,
                              chi2_red=_read_chi2_red(star_name, results_dir))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_walker_traces(result: dict, out_dir: str | None = None) -> str | None:
    """
    One trace panel per free parameter plus a ln-probability panel. Dead walkers
    are drawn in red over the healthy ensemble in grey, which is what made the
    GQ Cnc walker-8 freeze obvious by eye.
    """
    if "_chains" not in result:
        return None
    out_dir = out_dir if out_dir is not None else paths.PLOT_MCMC_DIR
    os.makedirs(out_dir, exist_ok=True)

    chains, lnprob = result["_chains"], result["_lnprob"]
    twigs, dead, burn = result["twigs"], result["dead_walkers"], result["burn_in"]
    n_steps, n_walkers, n_params = chains.shape
    steps = np.arange(n_steps)

    n_rows = n_params + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 2.3 * n_rows + 1.0), sharex=True)
    axes = np.atleast_1d(axes)
    fig.suptitle(
        f"{result['star']} — per-walker MCMC chains  "
        f"({n_walkers} walkers, {n_steps} steps, "
        f"median acceptance {result['median_accept']:.3f})",
        fontsize=14, fontweight="bold",
    )

    panels = [(chains[:, :, i], twigs[i].split("@")[0]) for i in range(n_params)]
    panels.append((lnprob, "ln probability"))

    for ax, (data, label) in zip(axes, panels):
        for w in range(n_walkers):
            if w in dead:
                continue
            ax.plot(steps, data[:, w], "-", color="grey", lw=0.6, alpha=0.55)
        for w in dead:
            ax.plot(steps, data[:, w], "-", color="red", lw=1.3,
                    label=f"walker {w} (frozen)")
        ax.axvline(burn, color="steelblue", ls="--", lw=1.2,
                   label=f"burn-in = {burn}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)

    # ln-prob panel: a frozen walker sits far below consensus, so a linear scale
    # flattens the healthy ensemble into a line. Clip to the live walkers' range.
    alive = [w for w in range(n_walkers) if w not in dead]
    if alive:
        live = lnprob[burn:][:, alive]
        lo, hi = np.percentile(live, [0.5, 100.0])
        pad = 0.08 * max(hi - lo, 1e-6)
        axes[-1].set_ylim(lo - pad, hi + pad)

    axes[-1].set_xlabel("MCMC step")
    verdict_line = " + ".join(result["verdicts"])
    axes[-1].text(
        0.01, 0.04, f"verdict: {verdict_line}", transform=axes[-1].transAxes,
        fontsize=9, bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9),
    )

    plt.tight_layout()
    out_path = os.path.join(
        out_dir, f"mcmc_chains_{result['star'].replace(' ', '_')}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------
_AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
STAR_CSV = os.path.join(_AUDIT_DIR, "mcmc_chain_audit.csv")
WALKER_CSV = os.path.join(_AUDIT_DIR, "mcmc_chain_walkers.csv")


def write_star_csv(results: list[dict], path: str = STAR_CSV) -> None:
    fields = ["star", "verdicts", "n_steps", "n_walkers", "burn_in", "chi2_red",
              "median_accept", "min_accept", "median_n_distinct", "worst_frozen_frac",
              "dead_walkers", "posterior_width_unreliable",
              "param", "rhat_full", "rhat_postburnin", "rhat_excl_dead",
              "tau", "tau_trustworthy", "ess", "between_over_within", "detail"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            base = {k: r.get(k, "") for k in fields if k in r and k != "params"}
            base["verdicts"] = "|".join(r.get("verdicts", []))
            base["dead_walkers"] = "|".join(str(w) for w in r.get("dead_walkers", []))
            if not r.get("params"):
                writer.writerow({**{k: "" for k in fields}, **base})
                continue
            for twig, p in r["params"].items():          # one row per parameter
                writer.writerow({**{k: "" for k in fields}, **base, "param": twig,
                                 **{k: p[k] for k in ("rhat_full", "rhat_postburnin",
                                                      "rhat_excl_dead", "tau",
                                                      "tau_trustworthy", "ess",
                                                      "between_over_within")}})


def write_walker_csv(results: list[dict], path: str = WALKER_CSV) -> None:
    fields = ["star", "n_steps", "walker", "last_move", "frozen_tail", "frozen_frac",
              "accept_frac", "n_distinct", "median_lnprob", "lnprob_offset", "is_dead"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            for m in r.get("walkers", []):
                writer.writerow({"star": r["star"], "n_steps": r["n_steps"],
                                 "is_dead": int(m["walker"] in r["dead_walkers"]),
                                 **{k: m[k] for k in
                                    ("walker", "last_move", "frozen_tail", "frozen_frac",
                                     "accept_frac", "n_distinct", "median_lnprob",
                                     "lnprob_offset")}})


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Run timing → Runs/run_stats/run_summary.txt (written at exit). Inside the
    # __main__ guard so that importing this module never starts a tracker.
    import sys as _rs_sys, os as _rs_os
    _rs_sys.path.insert(0, _rs_os.path.dirname(_rs_os.path.dirname(_rs_os.path.abspath(__file__))))
    import runstats
    _run_tracker = runstats.RunTracker("audit_mcmc_chains", catalogue=_rs_sys.argv[1:],
                                       per_star=False)
    parser = argparse.ArgumentParser(
        description="Audit per-walker MCMC chain health for Script 05 solutions.")
    parser.add_argument("catalogs", nargs="+",
                        help="target catalogue CSV(s) with a 'name' column")
    parser.add_argument("--per-walker", action="store_true",
                        help=f"also write {os.path.basename(WALKER_CSV)}")
    parser.add_argument("--plot", action="store_true",
                        help="write trace figures for flagged stars")
    parser.add_argument("--plot-all", action="store_true",
                        help="write trace figures for every star")
    parser.add_argument("--burnin", type=int, default=MCMC_BURN_IN_STEPS,
                        help=f"steps to trim (default {MCMC_BURN_IN_STEPS})")
    parser.add_argument("--skip-missing", action="store_true",
                        help="omit stars without a final bundle from the table "
                             "(they are still counted in the summary)")
    args = parser.parse_args()

    # The Windows console defaults to cp1252, which cannot encode the R̂ / χ² /
    # en-dash characters this pipeline uses throughout its output.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    phoebe.logger(clevel="ERROR")      # suppress PHOEBE's chatter during the audit
    paths.ensure_dirs()

    stars: list[str] = []
    for pattern in args.catalogs:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for catalog_path in matches:
            try:
                with open(catalog_path, newline="", encoding="utf-8") as f:
                    stars.extend(row["name"] for row in csv.DictReader(f))
            except Exception as exc:
                print(f"  ERROR: could not read catalogue — {catalog_path}: {exc}")
    seen: set[str] = set()
    stars = [s for s in stars if not (s in seen or seen.add(s))]
    if not stars:
        print("No stars found in the given catalogue(s).")
        sys.exit(1)

    header = (f"{'star':<13} {'n':>5} {'chi2r':>8} {'acc_med':>8} {'acc_min':>8} "
              f"{'uniq':>6} {'frozen':>7} {'Rhat_p':>8}  verdict")
    print(header)
    print("-" * 104)

    results, counts = [], {}
    for star in stars:
        r = audit_mcmc_chains(star, burn=args.burnin)
        results.append(r)
        for v in r["verdicts"]:
            counts[v] = counts.get(v, 0) + 1
        if args.skip_missing and not r.get("params"):
            continue

        worst_rhat = (max(p["rhat_postburnin"] for p in r["params"].values())
                      if r.get("params") else float("nan"))
        def _f(key, fmt):
            v = r.get(key)
            return format(v, fmt) if isinstance(v, (int, float)) and v == v else "n/a"
        print(f"{star:<13} {r.get('n_steps', 0) or 0:>5} {_f('chi2_red', '8.2f'):>8} "
              f"{_f('median_accept', '8.3f'):>8} {_f('min_accept', '8.3f'):>8} "
              f"{_f('median_n_distinct', '6.0f'):>6} "
              f"{_f('worst_frozen_frac', '7.3f'):>7} "
              f"{(format(worst_rhat, '8.3f') if worst_rhat == worst_rhat else 'n/a'):>8}  "
              f"{' + '.join(r['verdicts'])}")
        # Details run long on multi-verdict stars; a continuation line keeps the
        # numeric columns aligned instead of pushing them out of register.
        for part in r["detail"].split("; "):
            if part:
                print(f"{'':<13} {'':>47}   - {part}")

        if args.plot_all or (args.plot and any(v[0].isupper() for v in r["verdicts"])):
            out = plot_walker_traces(r)
            if out:
                print(f"{'':<13} -> {out}")

    print("-" * len(header))
    print("Summary:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    write_star_csv(results)
    print(f"Wrote {STAR_CSV}")
    if args.per_walker:
        write_walker_csv(results)
        print(f"Wrote {WALKER_CSV}")
