"""
================================================================================
AUDIT — q PROFILE OF A SEMI-DETACHED FIT, IN BOTH RADIUS BASINS
================================================================================
Summary:
    Pipeline_Log Entry 87. Both semi-detached Phase-E fits (V358 Cep, OT Cep)
    rail at q = 1.0 with a small hot primary (r1/a ~ 0.16) and a big lobe-filling
    cool secondary. Is that the data's preference (q > 1 wanted), or a local
    Nelder-Mead basin chosen by Script 05's r1 = 0.17 a seed, with the better
    solution a big hot primary and a smaller lobe-filling secondary (RZ Dra-like)?

    One JOB fixes q and minimises chi-squared over (T2, i, r1) with Nelder-Mead
    (scipy, bounded by clipping) from a chosen r1 start, using the SAME model
    configuration as Script 05 (5A gravity darkening / albedo, Horvat
    irradiation, 5B dataset rebuild with exptime + fti carry-over and
    dataset-scaled flux, 5C linear limb darkening 0.6). With --free-q, q is a
    fourth free parameter in [0.1, 3].

    MODES
      job         one (star, q, r1 start) minimisation -> one JSON in --outdir
      equivalence chi2 of this configuration vs a Script-05 checkpoint at the
                  chain's maximum-posterior sample (must agree)
      collect     merge the JSONs into Audits/q_profile.csv and draw
                  data/plots/audit_q/q_profile_<star>.png (+ best-fit overlays)

    Heavy-ish: each job runs ~100 PHOEBE forward models (seconds each; 600 s
    stars with exposure integration ~5x slower). Run only when no MCMC queue
    is running, or with the user's agreement. WSL python3, no MPI.

Usage (from Code/):
    python3 Audits/audit_q_profile.py equivalence --bundle <base.phoebe> --checkpoint <ckpt.phoebe>
    python3 Audits/audit_q_profile.py job --star "V358 Cep" --bundle <base.phoebe> --q 0.5 \
            --r1-start large --outdir Runs/q_investigation/jobs [--maxfev 110] [--free-q] \
            [--t2 4435 --incl 88]
    python3 Audits/audit_q_profile.py collect --outdir Runs/q_investigation/jobs
================================================================================
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR  = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)

RADIATIVE_CONVECTIVE_BOUNDARY_K = 8000.0      # as Script 05
# False = the pre-Entry-93 rule (8000 K step for every star), which the test tools
# need as their reference; they set gravity darkening explicitly (--gd).
GD_BY_MODEL_TYPE = False
LITERATURE_Q = {"RZ Dra": (0.40, "spectroscopic, +-0.04"),
                "GW Gem": (0.46, "photometric q-search, Lee et al. 2009")}


# ── model configuration (mirrors Script 05 sections 5A-5C) ─────────────────────
def configure_like_05(bundle, kind: str = "semidetached") -> None:
    for comp in ("primary", "secondary"):
        teff = bundle.get_value(f"teff@{comp}")
        if GD_BY_MODEL_TYPE and kind != "contact":
            # Entry 93: non-contact stars follow the ramp (as staged Script 05 does).
            import gravdark
            beta, albedo = gravdark.coeffs_for(kind, teff)
            bundle.set_value(f"gravb_bol@{comp}", beta)
            bundle.set_value(f"irrad_frac_refl_bol@{comp}", albedo)
        elif teff >= RADIATIVE_CONVECTIVE_BOUNDARY_K:
            bundle.set_value(f"gravb_bol@{comp}", 1.0)
            bundle.set_value(f"irrad_frac_refl_bol@{comp}", 1.0)
        else:
            bundle.set_value(f"gravb_bol@{comp}", 0.32)
            bundle.set_value(f"irrad_frac_refl_bol@{comp}", 0.6)
    bundle.set_value("irrad_method", compute="phoebe01",
                     value="wilson" if kind == "contact" else "horvat")
    times  = bundle.get_value("times",  dataset="lc01", context="dataset")
    fluxes = bundle.get_value("fluxes", dataset="lc01", context="dataset")
    sigmas = bundle.get_value("sigmas", dataset="lc01", context="dataset")
    band   = bundle.get_value("passband", dataset="lc01", context="dataset")
    exptime = float(bundle.get_value("exptime", dataset="lc01", context="dataset",
                                     check_visible=False))
    fti = bundle.get_value("fti_method", compute="phoebe01", dataset="lc01", check_visible=False)
    ovs = bundle.get_value("fti_oversample", compute="phoebe01", dataset="lc01", check_visible=False)
    uniq = np.unique(times, return_index=True)[1]
    bundle.remove_dataset("lc01")
    bundle.add_dataset("lc", times=times[uniq], fluxes=fluxes[uniq], sigmas=sigmas[uniq],
                       passband=band, dataset="lc01")
    if exptime > 0:
        bundle.set_value("exptime", dataset="lc01", context="dataset", value=exptime)
        bundle.set_value("fti_method", compute="phoebe01", dataset="lc01", value=fti)
        if fti != "none":
            bundle.set_value("fti_oversample", compute="phoebe01", dataset="lc01", value=int(ovs))
    bundle.set_value("pblum_mode", dataset="lc01", value="dataset-scaled")
    bundle.set_value("ld_mode_bol@primary", "manual")
    bundle.set_value("ld_mode_bol@secondary", "manual")
    for comp in ("primary", "secondary"):
        bundle.set_value(f"ld_mode@{comp}@lc01", "manual")
        bundle.set_value(f"ld_func@{comp}@lc01", "linear")
        bundle.set_value(f"ld_coeffs@{comp}@lc01", [0.6])


def chi2(bundle) -> float:
    """chi-squared of lc01 against model 'qprof'; inf if the model fails."""
    try:
        bundle.run_compute(model="qprof", overwrite=True, progressbar=False)
    except Exception:
        return float("inf")
    obs = bundle.get_value("fluxes", dataset="lc01", context="dataset")
    sig = bundle.get_value("sigmas", dataset="lc01", context="dataset")
    syn = bundle.get_value("fluxes", dataset="lc01", model="qprof")
    return float(np.sum(((obs - syn) / sig) ** 2))


def set_param(bundle, twig, value):
    bundle.set_value(twig, context="component", value=float(value))


def load(path):
    import logging
    logging.disable(logging.WARNING)
    import phoebe
    return phoebe.load(path)


# ── modes ──────────────────────────────────────────────────────────────────────
def mode_equivalence(a) -> int:
    ck = load(a.checkpoint)
    tw = list(ck.get_value("fitted_twigs", solution="best_solution"))
    S  = np.asarray(ck.get_value("samples", solution="best_solution"), float)
    L  = np.asarray(ck.get_value("lnprobabilities", solution="best_solution"), float)
    k  = np.unravel_index(np.nanargmax(L), L.shape)
    best = S[k[0], k[1], :]
    for t, v in zip(tw, best):
        ck.set_value(t, context="component", value=float(v))
    c_ck = chi2(ck)
    b = load(a.bundle)
    configure_like_05(b)
    for t, v in zip(tw, best):
        b.set_value(t, context="component", value=float(v))
    c_rep = chi2(b)
    rel = abs(c_rep - c_ck) / c_ck
    print(json.dumps({"map_sample": dict(zip(tw, map(float, best))), "lnp_max": float(L[k]),
                      "chi2_checkpoint": c_ck, "chi2_replicated": c_rep, "rel_diff": rel,
                      "pass": rel < 1e-6}, indent=1))
    return 0 if rel < 1e-6 else 1


def mode_job(a) -> int:
    from scipy.optimize import minimize
    t_start = time.time()
    b = load(a.bundle)
    configure_like_05(b)
    sma = float(b.get_value("sma", component="binary", context="component"))
    t1  = float(b.get_value("teff", component="primary", context="component"))
    t2_lo, t2_hi = 0.70 * t1, 1.10 * t1                 # Script 05's non-contact T2 prior
    t2_reg = float(b.get_value("teff", component="secondary", context="component"))
    if t2_reg < RADIATIVE_CONVECTIVE_BOUNDARY_K:
        t2_hi = min(t2_hi, RADIATIVE_CONVECTIVE_BOUNDARY_K)
    set_param(b, "q@binary", a.q)

    def r1max():
        return float(b.get_value("requiv_max", component="primary", context="component"))
    r1_start = (0.17 * sma if a.r1_start == "small"
                else 0.9 * r1max() if a.r1_start == "large" else float(a.r1_start) * sma)
    x0 = [a.t2 or t2_reg, a.incl, r1_start] + ([a.q] if a.free_q else [])
    scale = np.array([200.0, 1.5, 0.02 * sma] + ([0.1] if a.free_q else []))
    evals = {"n": 0, "best": (float("inf"), None)}

    def unpack(z):
        x = np.asarray(z) * scale
        t2 = float(np.clip(x[0], t2_lo, t2_hi))
        inc = float(np.clip(x[1], 60.0, 90.0))
        q = float(np.clip(x[3], 0.1, 3.0)) if a.free_q else a.q
        return t2, inc, float(x[2]), q, x

    def objective(z):
        t2, inc, r1, q, x = unpack(z)
        evals["n"] += 1
        set_param(b, "teff@secondary", t2)
        set_param(b, "incl@binary", inc)
        if a.free_q:
            set_param(b, "q@binary", q)
        r1_hi = r1max()
        penalty = 0.0
        if not (0.05 * sma <= r1 <= r1_hi):
            penalty = 1e6 * (abs(r1 - np.clip(r1, 0.05 * sma, r1_hi)) / sma + 1e-3)
            r1 = float(np.clip(r1, 0.05 * sma, r1_hi * 0.999))
        # out-of-range T2 / i / q are clipped above; penalise the excursion so
        # the simplex is pushed back rather than sliding along a flat wall.
        penalty += 1e3 * (abs(x[0] - t2) / 200 + abs(x[1] - inc) / 1.5
                          + (abs(x[3] - q) / 0.1 if a.free_q else 0.0))
        set_param(b, "requiv@primary", r1)
        c = chi2(b) + penalty
        if c < evals["best"][0]:
            evals["best"] = (c, (t2, inc, r1, q))
        return c

    res = minimize(objective, np.asarray(x0) / scale, method="Nelder-Mead",
                   options={"maxfev": a.maxfev, "xatol": 1e-3, "fatol": 1e-2,
                            "initial_simplex": None})
    c_best, (t2, inc, r1, q) = evals["best"]
    set_param(b, "q@binary", q)
    set_param(b, "teff@secondary", t2)
    set_param(b, "incl@binary", inc)
    set_param(b, "requiv@primary", r1)
    c_final = chi2(b)
    r2 = float(b.get_value("requiv", component="secondary", context="component"))
    phases = ((b.get_value("times", dataset="lc01", context="dataset")
               - b.get_value("t0_supconj", component="binary", context="component"))
              / b.get_value("period", component="binary", context="component")) % 1.0
    out = {
        "star": a.star, "mode": "free_q" if a.free_q else "fixed_q", "q_start": a.q,
        "r1_start": a.r1_start, "q": q, "chi2": c_final, "n": len(phases),
        "chi2_red": c_final / max(1, len(phases) - (4 if a.free_q else 3)),
        "T2": t2, "incl": inc, "r1_a": r1 / sma, "r2_a": r2 / sma,
        "r1_over_r1max": r1 / r1max(), "nm_message": str(res.message), "nfev": evals["n"],
        "wall_s": round(time.time() - t_start, 1),
        "phase": list(map(float, phases)),
        "obs": list(map(float, b.get_value("fluxes", dataset="lc01", context="dataset"))),
        "model": list(map(float, b.get_value("fluxes", dataset="lc01", model="qprof"))),
    }
    os.makedirs(a.outdir, exist_ok=True)
    tag = f"{a.star.replace(' ', '_')}_{out['mode']}_q{a.q:.2f}_{a.r1_start}"
    with open(os.path.join(a.outdir, tag + ".json"), "w", encoding="utf-8") as handle:
        json.dump(out, handle)
    print(f"{tag}: chi2 {c_final:.1f} q {q:.3f} T2 {t2:.0f} i {inc:.2f} r1 {r1/sma:.3f} "
          f"r2 {r2/sma:.3f} ({evals['n']} models, {out['wall_s']:.0f} s)")
    return 0


def mode_collect(a) -> int:
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [json.load(open(p, encoding="utf-8")) for p in sorted(glob.glob(os.path.join(a.outdir, "*.json")))]
    if not rows:
        print("no job files")
        return 1
    cols = ["star", "mode", "q_start", "r1_start", "q", "chi2", "chi2_red", "T2", "incl",
            "r1_a", "r2_a", "r1_over_r1max", "nfev", "wall_s", "nm_message"]
    out_csv = os.path.join(AUDIT_DIR, "q_profile.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["star"], r["mode"], r["r1_start"], r["q_start"])):
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()})
    plot_dir = os.path.join(CODE_DIR, "data", "plots", "audit_q")
    os.makedirs(plot_dir, exist_ok=True)
    for star in sorted({r["star"] for r in rows}):
        sr = [r for r in rows if r["star"] == star and r["mode"] == "fixed_q"]
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
        for start, mk in (("small", "o-"), ("large", "s-")):
            pts = sorted((r["q"], r["chi2_red"]) for r in sr if r["r1_start"] == start)
            if pts:
                ax.plot(*zip(*pts), mk, label=f"r1 start {start}")
        if star in LITERATURE_Q:
            ax.axvline(LITERATURE_Q[star][0], ls="--", c="k", lw=1,
                       label=f"literature q {LITERATURE_Q[star][0]} ({LITERATURE_Q[star][1]})")
        ax.set_yscale("log")
        ax.set_xlabel("q (fixed)")
        ax.set_ylabel("reduced chi-squared after NM")
        ax.set_title(f"{star}: q profile")
        ax.legend(fontsize=8)
        for start, c in (("small", "C0"), ("large", "C1")):
            cand = [r for r in rows if r["star"] == star and r["r1_start"] == start]
            if cand:
                best = min(cand, key=lambda r: r["chi2"])
                o = np.argsort(best["phase"])
                ph = np.asarray(best["phase"])[o]
                if start == "small":
                    ax2.plot(ph, np.asarray(best["obs"])[o], "k.", ms=3, label="data")
                ax2.plot(ph, np.asarray(best["model"])[o], c=c, lw=1.2,
                         label=f"best {start}: q {best['q']:.2f}, r1 {best['r1_a']:.2f}, "
                               f"r2 {best['r2_a']:.2f}, chi2r {best['chi2_red']:.0f}")
        ax2.set_xlabel("phase")
        ax2.set_ylabel("flux")
        ax2.legend(fontsize=7)
        ax2.set_title("best fit of each basin")
        fig.tight_layout()
        path = os.path.join(plot_dir, f"q_profile_{star.replace(' ', '_')}.png")
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print("figure:", path)
    print("table:", out_csv)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["equivalence", "job", "collect"])
    p.add_argument("--bundle")
    p.add_argument("--checkpoint")
    p.add_argument("--star")
    p.add_argument("--q", type=float, default=1.0)
    p.add_argument("--r1-start", default="small")
    p.add_argument("--t2", type=float, default=None)
    p.add_argument("--incl", type=float, default=87.0)
    p.add_argument("--maxfev", type=int, default=110)
    p.add_argument("--free-q", action="store_true")
    p.add_argument("--outdir", default=os.path.join(CODE_DIR, "Runs", "q_investigation", "jobs"))
    a = p.parse_args()
    return {"equivalence": mode_equivalence, "job": mode_job, "collect": mode_collect}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
