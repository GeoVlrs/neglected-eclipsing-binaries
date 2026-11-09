"""Third-light scaling audit (user-requested 3 Nov 2026; Pipeline_Log Entry 136).

With pblum_mode='dataset-scaled' (set by 04 and used by 05 and audit_model_tests.py),
PHOEBE 2.4.22 adds third light as  l3_frac/(1-l3_frac) * sum(pblum_abs)  to the
unscaled model fluxes (frontend/bundle.py, _scale_fluxes, ~line 12106). That is plain
dilution, but not by the fraction l3_frac: the third light actually in the model obeys

    l3_eff/(1-l3_eff) = k * l3_frac/(1-l3_frac)

with k = sum(pblum_abs) / mean(unscaled model flux). Y Sex: k = 27.98 (Runs/thirdlight/diag/).

For each star with mode free/fixed in Targets/third_light.csv this script:
  - loads its current input bundle (as 04 left it; never saved), computes the model with
    l3_frac 0 and 0.05 and measures k by a linear fit m(0.05) = a*m(0) + c (a pure dilution
    leaves no residual; the rms is reported);
  - converts the prior maximum and, for fixed stars, gives the l3_frac that would put the
    published fraction into the model;
  - where a current 05 result has an l3_frac row (results file with input_bundle_sha),
    measures k again on the final bundle (posterior-median state) and converts the median
    and its quantile ends. Two definitions: share of the MEAN flux (PHOEBE's sense) and
    share of the flux at MAXIMUM light (the usual light-curve-solution convention).

Writes Audits/l3_scaling.csv. Reads bundles and results only; writes nothing else.

    python3 Audits/audit_l3_scaling.py              # all free/fixed stars
    python3 Audits/audit_l3_scaling.py "GK Cep"     # named stars only
    python3 Audits/audit_l3_scaling.py --any "TZ Lyr" "FI Boo"
                                                    # named stars even if not free/fixed
With names, only those rows of Audits/l3_scaling.csv are replaced or added; the
others are kept (added 4 Nov 2026, Entry 137).
"""
import csv
import logging
import os
import sys
import time

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
import paths                                                    # noqa: E402

TEST_L3 = 0.05
OUT = os.path.join(AUDIT_DIR, "l3_scaling.csv")


def r_of(l):
    return l / (1.0 - l)


def frac_of(r):
    return r / (1.0 + r)


def ld_like_05(b):
    """05's limb darkening (05Optimization.py step 5C): bolometric LD manual, TESS LD
    linear 0.6. Applied before every measurement, so k is measured as 05 sees it
    (added 4 Nov 2026: raw bundles of rebuilt semi-detached stars fail the
    bolometric LD lookup)."""
    b.set_value("ld_mode_bol@primary", "manual")
    b.set_value("ld_mode_bol@secondary", "manual")
    for comp in ("primary", "secondary"):
        b.set_value(f"ld_mode@{comp}@lc01", "manual")
        b.set_value(f"ld_func@{comp}@lc01", "linear")
        b.set_value(f"ld_coeffs@{comp}@lc01", [0.6])


def models(b, l3_values):
    """Model fluxes (dataset-scaled, as fitted) for each l3_frac value; bundle not saved."""
    ld_like_05(b)
    b.set_value("l3_mode", dataset="lc01", value="fraction")
    out = []
    for v in l3_values:
        b.set_value("l3_frac", dataset="lc01", context="dataset", value=float(v))
        b.run_compute(model="l3audit", overwrite=True, progressbar=False)
        out.append(np.asarray(b.get_value("fluxes", dataset="lc01", model="l3audit"), float))
    return out


def measure_k(b):
    m0, m1 = models(b, (0.0, TEST_L3))
    A = np.vstack([m0, np.ones_like(m0)]).T
    (a, c), *_ = np.linalg.lstsq(A, m1, rcond=None)
    resid = float(np.std(m1 - A @ np.array([a, c])))
    k = (c / a / m0.mean()) / r_of(TEST_L3)
    # ratio of maximum to mean flux of the l3-free model: converts "share of mean" to
    # "share at maximum light"
    return float(k), resid, float(m0.max() / m0.mean())


def eff(l3_frac, k, max_over_mean):
    """Third light actually in the model for a set l3_frac: (share of mean, share at max)."""
    r_mean = k * r_of(l3_frac)                  # l3 flux / mean system flux
    return frac_of(r_mean), frac_of(r_mean / max_over_mean)


def set_for(target, k):
    """l3_frac that puts a share 'target' of the mean flux into the model."""
    return frac_of(r_of(target) / k)


def read_result(star):
    p = os.path.join(CODE_DIR, paths.RESULTS_DIR, f"results_{star.replace(' ', '_')}.csv")
    if not os.path.exists(p):
        return None
    rows = {r["Symbol"]: r for r in csv.DictReader(open(p, encoding="utf-8"))}
    if "l3_frac" not in rows or not rows.get("input_bundle_sha", {}).get("Value"):
        return None
    r = rows["l3_frac"]
    med = float(r["Value"])
    lo = med - float(r["Sigma_minus"] or 0.0)
    hi = med + float(r["Sigma_plus"] or 0.0)
    return med, lo, hi


def state(b):
    g = lambda tw: float(b.get_value(tw, context="component"))      # noqa: E731
    return {"T1": g("teff@primary"), "T2": g("teff@secondary"),
            "incl": g("incl@binary"), "q": g("q@binary")}


def main(argv):
    logging.disable(logging.WARNING)
    import phoebe
    any_mode = "--any" in argv
    names = [a for a in argv if a != "--any"]
    table = list(csv.DictReader(open(os.path.join(CODE_DIR, paths.TARGETS_DIR, "third_light.csv"),
                                     encoding="utf-8")))
    rows = [r for r in table if r["mode"] in ("free", "fixed")]
    if names:
        rows = [r for r in rows if r["name"] in names]
        if any_mode:
            have = {r["name"] for r in rows}
            by_name = {r["name"]: r for r in table}
            for n in names:
                if n not in have:
                    base = by_name.get(n, {})
                    rows.append({"name": n, "mode": base.get("mode") or "not in table",
                                 "prior_max": base.get("prior_max", ""),
                                 "l3_published": base.get("l3_published", "")})
    queue = [r["name"] for r in csv.DictReader(open(os.path.join(CODE_DIR, paths.TARGETS_DIR,
                                                                  "phaseF_queue.csv"), encoding="utf-8"))]
    out = []
    for r in rows:
        star, mode = r["name"], r["mode"]
        u = star.replace(" ", "_")
        rec = {"star": star, "queue_pos": queue.index(star) + 1 if star in queue else "",
               "mode": mode, "prior_max": r["prior_max"], "l3_published": r["l3_published"]}
        t = time.time()
        try:
            b = phoebe.load(os.path.join(CODE_DIR, paths.BUNDLE_DIR, f"bundle_{u}.phoebe"))
            k, resid, mxm = measure_k(b)
        except Exception as exc:                                    # one star must not stop the rest
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:150]}"
            print(f"{star}: input bundle FAILED: {rec['error']}", flush=True)
            out.append(rec)
            continue
        st = state(b)
        rec.update({f"in_{kk}": round(v, 3) for kk, v in st.items()})
        rec.update({"k_input": round(k, 3), "dilution_resid_rms": f"{resid:.1e}",
                    "max_over_mean": round(mxm, 4),
                    "prior_max_eff_mean": (round(eff(float(r["prior_max"]), k, mxm)[0], 3)
                                           if r["prior_max"] else "")})
        if r["l3_published"]:
            pub = float(r["l3_published"])
            rec["published_eff_if_set_mean"] = round(eff(pub, k, mxm)[0], 3)
            rec["l3_frac_to_set_for_published"] = round(set_for(pub, k), 5)
        msg = (f"{star}: k {k:.3f} (input bundle; i {st['incl']:.2f} q {st['q']:.3f} T2 {st['T2']:.0f}; "
               f"resid {resid:.1e}; max/mean {mxm:.3f})")
        res = read_result(star)
        if res is not None:
            try:
                bf = phoebe.load(os.path.join(CODE_DIR, paths.BUNDLE_DIR, f"bundle_{u}_final.phoebe"))
                kf, resid_f, mxm_f = measure_k(bf)
                rec.update({"k_final": round(kf, 3), "final_resid_rms": f"{resid_f:.1e}"})
                rec.update({f"fin_{kk}": round(v, 3) for kk, v in state(bf).items()})
                for tag, v in zip(("med", "lo", "hi"), res):
                    e_mean, e_max = eff(v, kf, mxm_f)
                    rec[f"l3_frac_{tag}"] = round(v, 6)
                    rec[f"l3_eff_mean_{tag}"] = round(e_mean, 4)
                    rec[f"l3_eff_max_{tag}"] = round(e_max, 4)
                msg += (f"; final k {kf:.3f}: l3_frac {res[0]:.5f} [{res[1]:.5f}, {res[2]:.5f}] -> "
                        f"{rec['l3_eff_mean_med']:.4f} [{rec['l3_eff_mean_lo']:.4f}, {rec['l3_eff_mean_hi']:.4f}] "
                        f"of mean light, {rec['l3_eff_max_med']:.4f} at maximum")
            except Exception as exc:
                rec["error"] = f"final: {type(exc).__name__}: {str(exc)[:150]}"
                msg += f"; final bundle FAILED: {rec['error']}"
        print(msg + f"  ({time.time() - t:.0f} s)", flush=True)
        out.append(rec)
    if names and os.path.exists(OUT):
        # named run: replace or add these stars' rows, keep every other row
        done = {rec["star"]: rec for rec in out}
        old = list(csv.DictReader(open(OUT, encoding="utf-8")))
        merged = [done.pop(o["star"]) if o["star"] in done else o for o in old]
        out = merged + [rec for rec in out if rec["star"] in done]
    keys = []
    for rec in out:
        keys += [kk for kk in rec if kk not in keys]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {OUT} ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
