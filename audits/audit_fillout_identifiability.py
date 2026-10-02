"""
================================================================================
AUDIT — IS THE CONTACT FILL-OUT FACTOR IDENTIFIABLE AT ALL? (χ² PROFILE SCAN)
================================================================================
Summary:
    Profiles χ²(f) for a contact system by stepping the fill-out factor across
    its full prior range with every other parameter held at the adopted solution,
    and running one forward model per step. Answers a question the MCMC cannot:
    whether single-band TESS photometry with q held fixed constrains f at all.

    Motivation (Pipeline_Log Entry 57): eight contact stars reported f within
    0.4% of each other (FU Dra 0.1273 ... V842 Her 0.1282). That was traced to an
    initialisation-ordering bug — every contact star entered the sampler at the
    same seed point (i = 82°, f = 0.15, q = 0.5) because the emcee ball was built
    before the Nelder-Mead pre-wash was adopted. The bug is fixed, but it leaves
    the underlying question open: if the likelihood is genuinely flat in f, then
    f is a nuisance parameter rather than a measurement no matter how the sampler
    is seeded, and the thesis must say so.

    This is a PROFILE, not a marginal posterior: it holds i and T2 fixed, so it
    measures how sharply the data alone pin f at the best-fit geometry. It
    deliberately ignores the i–f degeneracy, which is the point — a flat profile
    means f is unconstrained even in the most favourable case.

    Reads the FINAL bundle, whose parameters adopt_solution() has already set to
    the posterior median, so the scan is centred on the reported solution.
    Nothing is written back: the bundle is mutated in memory only and never saved.

Inputs:
    bundle_<star>_final.phoebe  — post-MCMC bundles from Script 05.

Outputs:
    <PLOT_MCMC_DIR>/fillout_chi2_profile_<star>.png
    Diagnostic summary table printed to stdout.

Usage:
    Run from Code/ (bundles are located via paths.BUNDLE_DIR):
    python Audits/audit_fillout_identifiability.py "RT Hyi" "DF Hya" "V1918 Cyg"

    With no arguments it scans the three default diagnostic stars above:
    one severe pile-up star (RT Hyi, χ²ᵥ = 187), one pile-up star that
    nonetheless fits well (DF Hya, χ²ᵥ = 5.5, so goodness-of-fit is controlled
    for), and one cleanly-sampled control far from the seed (V1918 Cyg, f = 0.31).

    COST: one PHOEBE forward model per grid point (~40 per star). Minutes, not
    hours — but it is still real compute, so do not launch it inside a live MCMC
    batch.

Dependencies:
    pip install phoebe numpy matplotlib
================================================================================
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phoebe

# Audits/ sits one level below Code/, where paths.py lives.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths                                              # noqa: E402

# Matches FREE_PARAMS_CONTACT in 05Optimization.py — the scan covers exactly the
# range the sampler was allowed to explore, no more.
F_MIN: float = 0.03
F_MAX: float = 0.50
N_GRID: int = 40

# The three fitted parameters of a contact fit, for the dof denominator. Kept
# local rather than imported: importing 05Optimization.py would re-execute it
# (it has no __main__ guard).
N_FREE_PARAMS: int = 3

DEFAULT_STARS = ("RT Hyi", "DF Hya", "V1918 Cyg")


def _chi2(bundle, model: str, dataset: str = "lc01") -> tuple[float, float]:
    """Raw and reduced χ², matching compute_reduced_chi_squared() in Script 05."""
    obs_flux  = bundle.get_value("fluxes", dataset=dataset, context="dataset")
    obs_sigma = bundle.get_value("sigmas", dataset=dataset, context="dataset")
    syn_flux  = bundle.get_value("fluxes", dataset=dataset, model=model)
    dof   = max(1, len(obs_flux) - N_FREE_PARAMS)
    chi2  = float(np.sum(((obs_flux - syn_flux) / obs_sigma) ** 2))
    return chi2, chi2 / dof, dof


def scan_fillout(star_name: str, bundle_dir: str | None = None) -> dict:
    """
    Profile χ² against fill-out factor for one star. Never raises.

    Returns a dict with the grid, the χ²ᵥ curve, the adopted f, the profile
    minimum, and the range of f within 1σ of that minimum after the same
    err_scale inflation Script 05 applies to its reported error bars.
    """
    bundle_dir = bundle_dir if bundle_dir is not None else paths.BUNDLE_DIR
    safe_name  = star_name.replace(" ", "_")
    path       = os.path.join(bundle_dir, f"bundle_{safe_name}_final.phoebe")

    result = {"star": star_name, "ok": False, "detail": ""}
    try:
        bundle = phoebe.load(path)
    except Exception as exc:
        result["detail"] = f"could not load final bundle: {exc}"
        return result

    try:
        f_adopted = float(bundle.get_value("fillout_factor", context="component"))
    except Exception as exc:
        result["detail"] = f"not a contact system (no fillout_factor): {exc}"
        return result

    grid  = np.linspace(F_MIN, F_MAX, N_GRID)
    chi2r = np.full(N_GRID, np.nan)
    dof   = 1

    for i, f in enumerate(grid):
        try:
            bundle.set_value("fillout_factor", context="component", value=float(f))
            # Same physics as the production final compute in Script 05 (5H):
            # Wilson irradiation for the shared envelope, Roche distortion.
            # progressbar=False: one bar per forward model buries the actual
            # output under tens of thousands of tqdm frames in a redirected log.
            bundle.run_compute(model="scan_model", overwrite=True,
                               irrad_method="wilson", distortion_method="roche",
                               progressbar=False)
            _, chi2r[i], dof = _chi2(bundle, "scan_model")
        except Exception:
            # Mesh failures near the bounds are expected and informative — leave
            # the point as NaN rather than aborting the scan.
            pass

    # Restore the adopted value so an interactive caller is not left holding a
    # bundle silently parked at the end of the grid.
    try:
        bundle.set_value("fillout_factor", context="component", value=f_adopted)
    except Exception:
        pass

    good = np.isfinite(chi2r)
    if not np.any(good):
        result["detail"] = "every grid point failed to compute"
        return result

    i_min   = int(np.nanargmin(chi2r))
    chi2min = float(chi2r[i_min])

    # THRESHOLD. A bare Δχ² = 1 is the 1σ likelihood-ratio unit only for RAW χ²
    # with correctly scaled sigmas. These fits have χ²ᵥ up to ~190, i.e. the TESS
    # sigmas are understated, which is exactly why Script 05 inflates the reported
    # error bars by err_scale = max(1, √χ²ᵥ). Applying the same inflation here,
    # σ → kσ with k² = χ²ᵥ_min, turns the 1σ condition into
    #     Δχ²_raw ≤ χ²ᵥ_min   ⟺   Δχ²ᵥ ≤ χ²ᵥ_min / dof.
    # Using Δχ²ᵥ = 1 instead (an earlier version of this script) is arbitrary: it
    # reads as far too tight for a badly-fitting star and far too loose for a
    # well-fitting one, and it flips the verdict in both directions.
    threshold = max(1.0, chi2min) / float(dof)

    # The interval is the FULL set of grid points within the threshold, not the
    # first crossing either side of the minimum. These profiles are not smooth
    # parabolas — RT Hyi wiggles by several χ²ᵥ units across a long flat floor —
    # and a first-crossing bracket stops at the first bump and badly understates
    # the range. Disjoint sets are reported as such rather than hidden.
    within = np.where(good & (chi2r <= chi2min + threshold))[0]
    lo = float(grid[within[0]]) if within.size else np.nan
    hi = float(grid[within[-1]]) if within.size else np.nan
    contiguous = bool(within.size and np.all(np.diff(within) == 1))

    # THE HEADLINE NUMBER. How strongly do the data prefer the profile minimum
    # over f = 0.15, the seed every contact star started from (Entry 57)? Raw Δχ²
    # is the likelihood-ratio unit; dividing by err_scale² = χ²ᵥ_min expresses it
    # in the same inflated σ the pipeline quotes, so it is directly comparable to
    # the published error bars. Read as σ²: ≫1 means the data genuinely pull f
    # away from the seed and a chain that stayed there simply failed to move;
    # ≈1 means f is weakly constrained and the reported value is the seed
    # showing through. Both are artifacts, but they are different failures.
    i_seed = int(np.argmin(np.abs(grid - 0.15)))
    seed_raw = (float(chi2r[i_seed]) - chi2min) * float(dof) if good[i_seed] else np.nan
    seed_scaled = seed_raw / max(1.0, chi2min)

    span = float(np.nanmax(chi2r) - chi2min)
    result.update(
        seed_dchi2_raw=seed_raw, seed_dchi2_scaled=seed_scaled,
        ok=True, grid=grid, chi2r=chi2r, f_adopted=f_adopted,
        f_best=float(grid[i_min]), chi2_min=chi2min,
        chi2_span=span, threshold=threshold,
        conf_lo=lo, conf_hi=hi, contiguous=contiguous,
        adopted_within=bool(
            good[int(np.argmin(np.abs(grid - f_adopted)))]
            and chi2r[int(np.argmin(np.abs(grid - f_adopted)))] <= chi2min + threshold),
        n_failed=int(np.sum(~good)),
    )
    return result


def plot_profile(result: dict, out_dir: str | None = None) -> str | None:
    """Write fillout_chi2_profile_<star>.png. Returns the path, or None."""
    if not result.get("ok"):
        return None
    out_dir = out_dir if out_dir is not None else paths.PLOT_MCMC_DIR
    os.makedirs(out_dir, exist_ok=True)

    grid, chi2r = result["grid"], result["chi2r"]
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.plot(grid, chi2r, "-o", ms=3.5, color="steelblue", lw=1.4, label=r"$\chi^2_\nu(f)$")
    ax.axvline(result["f_adopted"], color="crimson", lw=1.8,
               label=f"adopted f = {result['f_adopted']:.4f}")
    ax.axvline(result["f_best"], color="seagreen", lw=1.4, ls="--",
               label=f"profile minimum f = {result['f_best']:.4f}")
    # 0.15 is the Script 04 seed every contact star started from (Entry 57).
    ax.axvline(0.15, color="0.45", lw=1.2, ls=":", label="seed f = 0.15")
    if np.isfinite(result["conf_lo"]) and np.isfinite(result["conf_hi"]):
        ax.axvspan(result["conf_lo"], result["conf_hi"], color="seagreen", alpha=0.10,
                   label=r"1$\sigma$ after err_scale inflation")
    ax.axhline(result["chi2_min"] + result["threshold"], color="0.65", lw=0.9, ls="-.")

    ax.set_xlabel("Contact fill-out factor $f$")
    ax.set_ylabel(r"Reduced $\chi^2$")
    ax.set_title(f"{result['star']} — $\\chi^2$ profile in fill-out factor "
                 f"(i, $T_2$ held at the adopted solution)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, framealpha=0.92)
    plt.tight_layout()

    safe = result["star"].replace(" ", "_")
    out_path = os.path.join(out_dir, f"fillout_chi2_profile_{safe}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Dump the grid alongside the figure. The profile costs ~40 forward models;
    # writing it out means the summary statistic can be revised later without
    # paying for the scan again.
    with open(os.path.join(out_dir, f"fillout_chi2_profile_{safe}.csv"),
              "w", encoding="utf-8", newline="") as fh:
        fh.write("fillout_factor,chi2_reduced\n")
        for f_val, c_val in zip(result["grid"], result["chi2r"]):
            fh.write(f"{f_val:.6f},{c_val:.6f}\n")
    return out_path


if __name__ == "__main__":
    # Run timing → Runs/run_stats/run_summary.txt (written at exit). Inside the
    # __main__ guard so that importing this module never starts a tracker.
    import sys as _rs_sys, os as _rs_os
    _rs_sys.path.insert(0, _rs_os.path.dirname(_rs_os.path.dirname(_rs_os.path.abspath(__file__))))
    import runstats
    _run_tracker = runstats.RunTracker("audit_fillout_identifiability", catalogue=_rs_sys.argv[1:],
                                       per_star=False)
    phoebe.logger(clevel="ERROR")  # suppress PHOEBE's verbose per-model logging

    stars = sys.argv[1:] or list(DEFAULT_STARS)

    print(f"{'star':<12} {'f_adopt':>8} {'f_best':>8} {'chi2v_min':>10} "
          f"{'1-sigma range':>18} {'seed d-chi2':>12} {'in sigma^2':>11} {'fail':>5}")
    print("-" * 95)
    for star in stars:
        r = scan_fillout(star)
        if not r.get("ok"):
            print(f"{star:<12} ERROR — {r['detail']}")
            continue
        rng = (f"[{r['conf_lo']:.3f}, {r['conf_hi']:.3f}]"
               if np.isfinite(r["conf_lo"]) else "none")
        if not r["contiguous"]:
            rng += "*"          # disjoint: the set has gaps inside that span
        print(f"{star:<12} {r['f_adopted']:>8.4f} {r['f_best']:>8.4f} "
              f"{r['chi2_min']:>10.2f} {rng:>18} {r['seed_dchi2_raw']:>12.1f} "
              f"{r['seed_dchi2_scaled']:>11.2f} {r['n_failed']:>5}")
        p = plot_profile(r)
        if p:
            print(f"{'':<12} plot: {p}")
    print("-" * 95)
    print("* = the 1-sigma set is disjoint; the printed range is its outer envelope.")
    print("  Treat a narrow interval with caution when chi2v_min is large: the threshold")
    print("  can then fall below the grid's own point-to-point scatter.")
    print()
    print("READ THE LAST TWO COLUMNS. They give how far the data prefer f_best over the")
    print("f = 0.15 seed every contact star started from: raw Delta-chi2, then the same")
    print("quantity in units of the pipeline's own inflated sigma (err_scale = sqrt(chi2v)).")
    print("  >> 1  the data DO pin f away from the seed; a star still reporting ~0.15")
    print("        simply never moved there  -> chain failure (Pipeline_Log Entry 57)")
    print("  ~ 1   f is weakly constrained; the reported value is the seed showing")
    print("        through  -> f is a nuisance parameter for that star, not a measurement")
