"""
================================================================================
CONTACT q-SEARCH — ADOPT A MASS RATIO FOR EACH CONTACT STAR (between 04 and 05)
================================================================================
Summary:
    Script 05 holds the mass ratio of contact stars FIXED. Until Entry 90 it was
    0.5 for every star, which biased the fitted inclination by 5-16 degrees and
    the fill-out wherever the true q differs (II UMa, Y Sex, FP Boo, TY UMa,
    FN Cam against their published solutions). This stage chooses the value:

      1. q-SEARCH. For a grid of fixed q, minimise chi-squared over (T2, i, f)
         with Nelder-Mead from two fill-out starts, on the star's own Script-04
         bundle configured exactly as Script 05 configures it. q is the MODEL
         ratio m2/m1 with star 1 the hotter star eclipsed at phase 0, so q > 1
         covers W-type systems (hotter star less massive). The curve chi2(q)
         gives the photometric mass ratio and its interval.
      2. ADOPTION. If Targets/contact_q_literature.csv has a SPECTROSCOPIC mass
         ratio for the star, adopt it, on whichever branch (q or 1/q) the
         photometry prefers. Otherwise adopt the q-search minimum. A flat-bottomed
         curve (typically partial eclipses) is flagged q_weak: photometric mass
         ratios are reliable only with total eclipses (Terrell & Wilson 2005).

    Script 04 reads data/qsearch/q_adopted_<star>.txt when it (re)builds a contact
    bundle, and Script 05's preflight refuses a contact bundle whose q is not
    the adopted one.

Modes (from Code/, WSL python3, no MPI):
    probe-mesh  --bundle B                  fill-out x q grid of forward models
    plan        CATALOGUE --stage coarse|refine|literature   -> job lines on stdout
    job         --star S --q Q --f F        one fixed-q fit -> data/qsearch/jobs/<star>/
    adopt       CATALOGUE                   curves, q_adopted_<star>.txt, validation table
    Driver: Runs/qsearch/run_qsearch.sh (xargs -P 16, one retry per job).

    HEAVY: ~38 jobs x ~110 forward models per star. Run only with the MCMC queues
    stopped, or with the user's agreement.

Outputs (data/qsearch/):
    jobs/<star>/q<q>_f<f>.json   one per job
    qsearch_<star>.csv / .png    the curve
    q_adopted_<star>.txt         key = value lines read by 04 and 05
    Audits/contact_q_validation.csv / data/plots/audit_q/contact_q_validation.png
================================================================================
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

import numpy as np

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
import paths                                              # noqa: E402
from morphology import CONTACT_FILLOUT_CEIL, resolve_morphology_kind   # noqa: E402

RADIATIVE_CONVECTIVE_BOUNDARY_K = 8000.0      # as Script 05
COARSE_Q = [0.08, 0.13, 0.2, 0.3, 0.45, 0.7, 1.0, 1.4, 2.2, 3.3, 5.0, 7.7, 12.0]
F_STARTS = [0.15, 0.55]
FILLOUT_BOUNDS = (0.03, CONTACT_FILLOUT_CEIL)      # Script 05's contact prior (see probe-mesh)
INTERVAL_DCHI2 = 4.0          # scaled delta chi2 defining the q-search interval
WEAK_FRACTION = 0.5           # interval wider than this fraction of q_min -> q_weak
LITERATURE_CSV = os.path.join(CODE_DIR, "Targets", "contact_q_literature.csv")
JOBS_DIR = os.path.join(paths.QSEARCH_DIR, "jobs")
# Mesh size override for a re-test (Entry 93: at the bundles' 2000 triangles chi2
# jumps by up to 80 for a shallow W-type star). QSEARCH_NTRIANGLES=4000 fits with
# that mesh and keeps its results apart, in jobs_n4000/.
NTRIANGLES_OVERRIDE = int(os.environ.get("QSEARCH_NTRIANGLES", "0") or 0)
if NTRIANGLES_OVERRIDE:
    JOBS_DIR = os.path.join(paths.QSEARCH_DIR, f"jobs_n{NTRIANGLES_OVERRIDE}")


# ── model (mirrors Script 05 sections 5A-5C; verified equal to a 05 checkpoint
#    and to a Script-04 contact bundle, Entries 87 and 89) ─────────────────────
def load(path):
    import logging
    logging.disable(logging.WARNING)
    import phoebe
    return phoebe.load(path)


def configure_like_05(bundle) -> None:
    for comp in ("primary", "secondary"):
        teff = bundle.get_value(f"teff@{comp}")
        hot = teff >= RADIATIVE_CONVECTIVE_BOUNDARY_K
        bundle.set_value(f"gravb_bol@{comp}", 1.0 if hot else 0.32)
        bundle.set_value(f"irrad_frac_refl_bol@{comp}", 1.0 if hot else 0.6)
    bundle.set_value("irrad_method", compute="phoebe01", value="wilson")
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
    try:
        bundle.flip_constraint("fillout_factor", solve_for="pot")
    except Exception:
        pass                                   # already flipped by Script 04


def chi2(bundle) -> float:
    try:
        bundle.run_compute(model="qsearch", overwrite=True, progressbar=False)
    except Exception:
        return float("inf")
    obs = bundle.get_value("fluxes", dataset="lc01", context="dataset")
    sig = bundle.get_value("sigmas", dataset="lc01", context="dataset")
    syn = bundle.get_value("fluxes", dataset="lc01", model="qsearch")
    return float(np.sum(((obs - syn) / sig) ** 2))


def put(bundle, twig, value):
    bundle.set_value(twig, context="component", value=float(value))


def safe_name(star: str) -> str:
    return star.replace(" ", "_")


def contact_targets(catalogue: str) -> list:
    rows = list(csv.DictReader(open(catalogue, encoding="utf-8")))
    return [r for r in rows if resolve_morphology_kind(r["name"], r.get("type", "")) == "contact"
            and os.path.exists(os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name(r['name'])}.phoebe"))]


def literature() -> dict:
    """{star: row} for rows usable to FIX q (method == spectroscopic)."""
    if not os.path.exists(LITERATURE_CSV):
        return {}
    out = {}
    for r in csv.DictReader(open(LITERATURE_CSV, encoding="utf-8")):
        try:
            r["q_lit"] = float(r["q_lit"])
        except (ValueError, KeyError):
            continue
        out[r["name"]] = r
    return out


def job_path(star: str, q: float, f: float) -> str:
    return os.path.join(JOBS_DIR, safe_name(star), f"q{q:.4f}_f{f:.2f}.json")


def read_jobs(star: str) -> list:
    return [json.load(open(p, encoding="utf-8"))
            for p in sorted(glob.glob(os.path.join(JOBS_DIR, safe_name(star), "*.json")))]


def stuck(jobs: list) -> list:
    """Fits that never left their start: (T2, i, f) within a fraction of the first
    simplex step of (0.97 T1, 75, f_start) AND finished in under 0.6 of the star's
    median time, i.e. most trial models failed at once. Measured 12 Oct 2026:
    PHOEBE cannot build the contact lobe at q = 10.0 and 10.3503 (8.93, 10.7 and
    12.0 compute). Such a fit's chi2 is the start point's, not a minimum."""
    if not jobs:
        return []
    med = float(np.median([j["wall_s"] for j in jobs]))
    return [j for j in jobs
            if j["wall_s"] < 0.6 * med and abs(j["incl"] - 75.0) <= 0.5
            and abs(j["fillout"] - j["f_start"]) <= 0.01
            and abs(j["T2_over_T1"] - 0.97) <= 0.008]


def curve(jobs: list) -> list:
    """[(q, best chi2, job)] sorted by q — the minimum over fill-out starts.
    Stuck fits (see stuck()) are left out."""
    bad = {id(j) for j in stuck(jobs)}
    jobs = [j for j in jobs if id(j) not in bad]
    best = {}
    for j in jobs:
        if np.isfinite(j["chi2"]) and (j["q"] not in best or j["chi2"] < best[j["q"]]["chi2"]):
            best[j["q"]] = j
    return [(q, best[q]["chi2"], best[q]) for q in sorted(best)]


# ── modes ──────────────────────────────────────────────────────────────────────
def mode_probe_mesh(a) -> int:
    b = load(a.bundle)
    configure_like_05(b)
    print("rows: fill-out; columns: q.  ok = forward model computed")
    qs = [0.08, 0.2, 0.5, 1.0, 3.0, 10.0]
    print("f\\q   " + "  ".join(f"{q:>5}" for q in qs))
    for f in (0.03, 0.5, 0.7, 0.8, 0.9, 0.95):
        cells = []
        for q in qs:
            try:
                put(b, "q@binary", q)
                put(b, "fillout_factor@contact_envelope", f)
                cells.append("   ok" if np.isfinite(chi2(b)) else " FAIL")
            except Exception:
                cells.append(" FAIL")
        print(f"{f:<5} " + "  ".join(cells), flush=True)
    return 0


def mode_plan(a) -> int:
    lit = literature()
    for r in contact_targets(a.catalogue):
        star = r["name"]
        if a.only and star not in a.only:
            continue
        if a.stage == "coarse":
            qs = COARSE_Q
        elif a.stage == "literature":
            row = lit.get(star)
            qs = ([round(row["q_lit"], 4), round(1.0 / row["q_lit"], 4)]
                  if row and row.get("method") == "spectroscopic" else [])
        else:                                   # refine around the coarse minimum
            c = curve(read_jobs(star))
            if len(c) < 3:
                continue
            k = int(np.argmin([x[1] for x in c]))
            qs = []
            for lo, hi in ((c[max(k - 1, 0)][0], c[k][0]), (c[k][0], c[min(k + 1, len(c) - 1)][0])):
                if hi > lo:
                    qs += [round(lo * (hi / lo) ** (1 / 3), 4), round(lo * (hi / lo) ** (2 / 3), 4)]
        for q in qs:
            for f in F_STARTS:
                if not os.path.exists(job_path(star, q, f)):
                    print(f"{star}|{q}|{f}")
    return 0


def mode_job(a) -> int:
    from scipy.optimize import minimize
    t_start = time.time()
    b = load(os.path.join(paths.BUNDLE_DIR, f"bundle_{safe_name(a.star)}.phoebe"))
    configure_like_05(b)
    if NTRIANGLES_OVERRIDE:
        b.set_value_all("ntriangles", NTRIANGLES_OVERRIDE)
    t1 = float(b.get_value("teff", component="primary", context="component"))
    t2_seed = float(b.get_value("teff", component="secondary", context="component"))
    # Script 05's contact prior: 0.80-1.25 T1, clamped to the secondary's regime.
    t2_lo, t2_hi = 0.80 * t1, 1.25 * t1
    if t2_seed < RADIATIVE_CONVECTIVE_BOUNDARY_K:
        t2_hi = min(t2_hi, RADIATIVE_CONVECTIVE_BOUNDARY_K)
    else:
        t2_lo = max(t2_lo, RADIATIVE_CONVECTIVE_BOUNDARY_K)
    f_lo, f_hi = FILLOUT_BOUNDS
    put(b, "q@binary", a.q)
    scale = np.array([200.0, 1.5, 0.05])
    evals = {"n": 0, "best": (float("inf"), None)}

    def objective(z):
        x = np.asarray(z) * scale
        t2 = float(np.clip(x[0], t2_lo, t2_hi))
        inc = float(np.clip(x[1], 40.0, 90.0))
        f = float(np.clip(x[2], f_lo, f_hi))
        penalty = 1e3 * (abs(x[0] - t2) / 200 + abs(x[1] - inc) / 1.5 + abs(x[2] - f) / 0.05)
        evals["n"] += 1
        put(b, "teff@secondary", t2)
        put(b, "incl@binary", inc)
        put(b, "fillout_factor@contact_envelope", f)
        c = chi2(b) + penalty
        if c < evals["best"][0]:
            evals["best"] = (c, (t2, inc, f))
        return c

    res = minimize(objective, np.array([0.97 * t1, 75.0, a.f]) / scale, method="Nelder-Mead",
                   options={"maxfev": a.maxfev, "xatol": 1e-3, "fatol": 1e-2})
    if evals["best"][1] is None:
        print(f"{a.star} q {a.q}: no valid model")
        return 1
    t2, inc, f = evals["best"][1]
    put(b, "teff@secondary", t2)
    put(b, "incl@binary", inc)
    put(b, "fillout_factor@contact_envelope", f)
    c_final = chi2(b)
    sma = float(b.get_value("sma", component="binary", context="component"))
    phases = ((b.get_value("times", dataset="lc01", context="dataset")
               - b.get_value("t0_supconj", component="binary", context="component"))
              / b.get_value("period", component="binary", context="component")) % 1.0
    out = {"star": a.star, "q": a.q, "f_start": a.f, "chi2": c_final, "n": len(phases),
           "chi2_red": c_final / max(1, len(phases) - 3), "T2": t2, "T2_over_T1": t2 / t1,
           "incl": inc, "fillout": f,
           "r1_a": float(b.get_value("requiv", component="primary", context="component")) / sma,
           "r2_a": float(b.get_value("requiv", component="secondary", context="component")) / sma,
           "nfev": evals["n"], "nm_message": str(res.message),
           "ntriangles": NTRIANGLES_OVERRIDE or 2000,
           "wall_s": round(time.time() - t_start, 1), "phase": list(map(float, phases)),
           "obs": list(map(float, b.get_value("fluxes", dataset="lc01", context="dataset"))),
           "model": list(map(float, b.get_value("fluxes", dataset="lc01", model="qsearch")))}
    path = job_path(a.star, a.q, a.f)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w", encoding="utf-8") as handle:
        json.dump(out, handle)
    os.replace(path + ".tmp", path)
    print(f"{a.star} q {a.q:<7} f0 {a.f}: chi2r {out['chi2_red']:.2f} T2/T1 {t2 / t1:.3f} "
          f"i {inc:.2f} f {f:.3f} ({evals['n']} models, {out['wall_s']:.0f} s)")
    return 0


def interval(c: list, k: int) -> tuple:
    """(q_lo, q_hi, touches_edge): where scaled delta chi2 <= INTERVAL_DCHI2,
    interpolated linearly in log q between grid points."""
    qs = np.log([x[0] for x in c])
    n = c[k][2]["n"]
    d = (np.array([x[1] for x in c]) - c[k][1]) / (c[k][1] / max(1, n - 3))
    def walk(step):
        i = k
        while 0 <= i + step < len(c) and d[i + step] <= INTERVAL_DCHI2:
            i += step
        if not (0 <= i + step < len(c)):
            return qs[i], True
        frac = (INTERVAL_DCHI2 - d[i]) / (d[i + step] - d[i])
        return qs[i] + frac * (qs[i + step] - qs[i]), False
    lo, edge_lo = walk(-1)
    hi, edge_hi = walk(+1)
    return float(math.exp(lo)), float(math.exp(hi)), bool(edge_lo or edge_hi)


def mode_adopt(a) -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lit = literature()
    rows = []
    for r in contact_targets(a.catalogue):
        star, safe = r["name"], safe_name(r["name"])
        all_jobs = read_jobs(star)
        for j in stuck(all_jobs):
            print(f"{star}: fit at q {j['q']} (f0 {j['f_start']}) never left its start "
                  f"({j['wall_s']:.0f} s) — left out")
        c = curve(all_jobs)
        if len(c) < 5:
            print(f"{star}: only {len(c)} q points — skipped")
            continue
        k = int(np.argmin([x[1] for x in c]))
        q_min, c_min, best = c[k]
        red_min = c_min / max(1, best["n"] - 3)
        q_lo, q_hi, edge = interval(c, k)
        weak = bool(edge or (q_hi - q_lo) / q_min > WEAK_FRACTION)
        row = lit.get(star)
        q_adopted, source, ref, lit_pen = q_min, "qsearch", "", ""
        if row and row.get("method") == "spectroscopic":
            cand = [x for x in c if any(abs(x[0] - v) < 2e-4 * max(1, v)
                                        for v in (round(row["q_lit"], 4), round(1 / row["q_lit"], 4)))]
            if cand:
                q_adopted, c_ad, _ = min(cand, key=lambda x: x[1])
                source, ref = "literature_spectroscopic", row.get("reference", "")
                lit_pen = round((c_ad - c_min) / red_min, 1)
            else:
                print(f"{star}: literature q {row['q_lit']} has no q-search job yet "
                      "(run the 'literature' stage) — q-search minimum adopted for now")
        adopted_job = next(x[2] for x in c if x[0] == q_adopted)
        # The interval test alone flagged 4 of 41 stars, yet the q-search minimum is
        # well off the spectroscopic q for 9 of 22 (Audits/contact_q_validation.csv):
        # a photometric q is weak whatever its formal interval says.
        interval_weak = weak
        weak = bool(weak or source == "qsearch")
        out = {
            "star": star, "q_adopted": q_adopted, "q_source": source, "reference": ref,
            "branch": "A" if q_adopted <= 1 else "W",
            "q_search_min": q_min, "q_search_lo": round(q_lo, 4), "q_search_hi": round(q_hi, 4),
            "q_weak": weak, "q_interval_weak": interval_weak, "chi2_red_at_search_min": round(red_min, 3),
            "chi2_red_at_adopted": round(adopted_job["chi2_red"], 3),
            "lit_scaled_dchi2": lit_pen,
            "incl_at_adopted": round(adopted_job["incl"], 2),
            "fillout_at_adopted": round(adopted_job["fillout"], 3),
            "T2_over_T1_at_adopted": round(adopted_job["T2_over_T1"], 4),
            "q_lit": row["q_lit"] if row else "", "lit_method": row.get("method", "") if row else "",
            "i_lit": row.get("i_lit", "") if row else "",
        }
        rows.append(out)
        with open(os.path.join(paths.QSEARCH_DIR, f"q_adopted_{safe}.txt"), "w", encoding="utf-8") as h:
            h.write(f"# Adopted mass ratio for {star} (ContactQSearch.py, {time.strftime('%Y-%m-%d')})\n")
            for key in ("q_adopted", "q_source", "reference", "branch", "q_search_min",
                        "q_search_lo", "q_search_hi", "q_weak", "q_interval_weak",
                        "lit_scaled_dchi2",
                        "incl_at_adopted", "fillout_at_adopted", "T2_over_T1_at_adopted"):
                h.write(f"{key} = {out[key]}\n")
        with open(os.path.join(paths.QSEARCH_DIR, f"qsearch_{safe}.csv"), "w", newline="",
                  encoding="utf-8") as h:
            w = csv.writer(h)
            w.writerow(["q", "chi2", "chi2_red", "T2_over_T1", "incl", "fillout", "f_start"])
            for q, cc, j in c:
                w.writerow([q, round(cc, 2), round(j["chi2_red"], 3), round(j["T2_over_T1"], 4),
                            round(j["incl"], 2), round(j["fillout"], 3), j["f_start"]])
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
        ax.plot([x[0] for x in c], [x[2]["chi2_red"] for x in c], "o-", ms=4)
        ax.axvspan(q_lo, q_hi, color="C0", alpha=0.12, label=f"q-search interval {q_lo:.2f}-{q_hi:.2f}")
        ax.axvline(q_adopted, c="C3", lw=1.2, label=f"adopted {q_adopted:.3f} ({source})")
        if row:
            for v in (row["q_lit"], 1 / row["q_lit"]):
                ax.axvline(v, c="k", ls=":", lw=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("q = m2/m1 (star 1 = hotter, eclipsed at phase 0)")
        ax.set_ylabel("reduced chi-squared"); ax.set_title(f"{star}: q-search" + ("  [weak]" if weak else ""))
        ax.legend(fontsize=7)
        o = np.argsort(adopted_job["phase"])
        ax2.plot(np.asarray(adopted_job["phase"])[o], np.asarray(adopted_job["obs"])[o], "k.", ms=3)
        ax2.plot(np.asarray(adopted_job["phase"])[o], np.asarray(adopted_job["model"])[o], "C3-", lw=1)
        ax2.set_xlabel("phase"); ax2.set_ylabel("flux")
        ax2.set_title(f"NM fit at adopted q: i {adopted_job['incl']:.1f}, f {adopted_job['fillout']:.2f}")
        fig.tight_layout()
        fig.savefig(os.path.join(paths.QSEARCH_DIR, f"qsearch_{safe}.png"), dpi=120)
        plt.close(fig)
    if not rows:
        return 1
    val = os.path.join(CODE_DIR, "Audits", "contact_q_validation.csv")
    with open(val, "w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # literature (folded to <= 1) against the q-search minimum (folded to <= 1)
    pts = [(r["q_lit"], min(r["q_search_min"], 1 / r["q_search_min"]), r["q_weak"], r["lit_method"])
           for r in rows if r["q_lit"] != ""]
    if pts:
        fig, ax = plt.subplots(figsize=(5.2, 5))
        for ql, qs_, weak, m in pts:
            ax.plot(ql, qs_, "o" if m == "spectroscopic" else "s", c="C1" if weak else "C0", ms=6)
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_xlabel("literature q (circle: spectroscopic, square: photometric)")
        ax.set_ylabel("q-search minimum (folded to <= 1; orange = weak)")
        ax.set_title("Photometric q-search against published mass ratios")
        os.makedirs(os.path.join(paths.PLOT_DIR, "audit_q"), exist_ok=True)
        fig.tight_layout()
        fig.savefig(os.path.join(paths.PLOT_DIR, "audit_q", "contact_q_validation.png"), dpi=130)
        plt.close(fig)
    cols = ["star", "q_adopted", "q_source", "branch", "q_search_min", "q_search_lo",
            "q_search_hi", "q_weak", "chi2_red_at_adopted", "incl_at_adopted",
            "fillout_at_adopted", "q_lit", "lit_method", "lit_scaled_dchi2"]
    print("  ".join(cols))
    for r in rows:
        print("  ".join(str(r[c]) for c in cols))
    print("table:", val)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["probe-mesh", "plan", "job", "adopt"])
    p.add_argument("catalogue", nargs="?")
    p.add_argument("--stage", choices=["coarse", "refine", "literature"], default="coarse")
    p.add_argument("--only", nargs="*", default=[])
    p.add_argument("--bundle")
    p.add_argument("--star")
    p.add_argument("--q", type=float)
    p.add_argument("--f", type=float, default=0.15)
    p.add_argument("--maxfev", type=int, default=110)
    a = p.parse_args()
    return {"probe-mesh": mode_probe_mesh, "plan": mode_plan, "job": mode_job,
            "adopt": mode_adopt}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
