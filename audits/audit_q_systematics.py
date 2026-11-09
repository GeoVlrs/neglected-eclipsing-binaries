"""
================================================================================
AUDIT — SYSTEMATIC ERROR ON i AND FILL-OUT FROM THE FIXED MASS RATIO (plan C2)
================================================================================
Summary:
    Contact stars are fitted with q FIXED at the adopted value (Entry 92), so the
    MCMC error bars on i and fill-out f do not include the uncertainty of q. The
    q-search already records the best i and f at every q it tried
    (data/qsearch/qsearch_<star>.csv, the curve after the stuck-fit filter).
    This audit reads that curve and evaluates i(q) and f(q), interpolated
    linearly in log q, at the ends of a q range:

      spectroscopic q  : q_lit +- its published error, converted to the model
                         convention (W-type: q_model = 1 / q_lit);
      q-search q (weak), by eclipse type (Entry 157; Terrell & Wilson 2005:
                         photometric q is reliable for complete eclipses only):
          complete eclipses: q_adopted x (1 -+ 0.15);
          partial eclipses : q_adopted / F .. q_adopted x F, F = exp of the 68th
                             percentile of |ln(q_search / q_spec)| over the
                             partial-eclipse stars with a spectroscopic q
                             (Audits/contact_q_validation.csv; F = 1.59 on
                             9 Nov 2026, 17 stars). Some partial stars miss by
                             a factor > 2.5, so F is a 68 % level, not a bound.
      Complete vs partial: sphere approximation with Eggleton (1983) lobe radii,
      complete if cos i <= r1 - r2 (i at the adopted q); approximate near 0.

    The formal q-search interval (scaled delta chi2 <= 4; far too narrow,
    Entry 92) is reported alongside for reference.

    incl_sys_q / fillout_sys_q = the larger |change| at the two ends.

    CAVEATS: the curve comes from single Nelder-Mead fits (mesh noise; Entry 93)
    with no third light; for stars whose q-search sat in a wrong basin (TZ Boo,
    Entry 111) the curve describes that basin. A range end outside the q values
    the search tried gives NaN.

Usage (from Code/, light, seconds):
    python Audits/audit_q_systematics.py [Targets/phaseF_queue.csv]
Output:
    Audits/q_systematics.csv, table printed
================================================================================
"""
import csv
import math
import os
import sys

import numpy as np

AUDIT_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(AUDIT_DIR)
sys.path.insert(0, CODE_DIR)
import morphology                                         # noqa: E402
import paths                                              # noqa: E402

WEAK_BAND = 0.15                 # complete eclipses (user decision 9 Nov 2026, Entry 157)
ASSUMED_SPEC_FRAC = 0.05
PARTIAL_PERCENTILE = 0.6827      # partial eclipses: factor F from this percentile (Entry 157)


def lobe_radius(q: float) -> float:
    """Eggleton (1983) volume-equivalent Roche-lobe radius (a = 1) of the star
    whose mass ratio to its companion is q."""
    a = q ** (2 / 3)
    return 0.49 * a / (0.6 * a + math.log(1 + q ** (1 / 3)))


def totality_margin(q: float, incl_deg: float) -> float:
    """(r1 - r2) - cos i for spheres with Eggleton lobe radii; > 0 means complete
    eclipses. Contact stars over-fill and are elongated: approximate near 0."""
    q = q if q <= 1 else 1 / q
    return (lobe_radius(1 / q) - lobe_radius(q)) - math.cos(math.radians(incl_deg))


def partial_factor() -> tuple:
    """F for partial-eclipse q-search stars: exp(68th percentile of
    |ln(q_search_min / q_lit)|) over the partial-eclipse stars of
    Audits/contact_q_validation.csv with a spectroscopic q (both folded to <= 1).
    Returns (F, n_partial, n_complete, complete 68th-percentile factor)."""
    fold = lambda q: q if q <= 1 else 1 / q
    with open(os.path.join(AUDIT_DIR, "contact_q_validation.csv"), encoding="utf-8-sig") as handle:
        rows = [r for r in csv.DictReader(handle)
                if r["lit_method"] == "spectroscopic" and r["q_lit"] and r["q_search_min"] and r["incl_at_adopted"]]
    groups = {True: [], False: []}
    for r in rows:
        d = abs(math.log(fold(float(r["q_search_min"])) / fold(float(r["q_lit"]))))
        groups[totality_margin(float(r["q_lit"]), float(r["incl_at_adopted"])) > 0].append(d)

    def pct(v, p):
        v = sorted(v)
        k = (len(v) - 1) * p
        i = int(k)
        return v[i] + (v[min(i + 1, len(v) - 1)] - v[i]) * (k - i)
    return (math.exp(pct(groups[False], PARTIAL_PERCENTILE)), len(groups[False]), len(groups[True]),
            math.exp(pct(groups[True], PARTIAL_PERCENTILE)))


def read_curve(star: str):
    path = os.path.join(paths.QSEARCH_DIR, f"qsearch_{star.replace(' ', '_')}.csv")
    with open(path, encoding="utf-8") as handle:
        rows = [r for r in csv.DictReader(handle)]
    rows.sort(key=lambda r: float(r["q"]))
    q = np.array([float(r["q"]) for r in rows])
    return q, np.array([float(r["incl"]) for r in rows]), np.array([float(r["fillout"]) for r in rows])


def local_fit(qs, ys, qa, need=0.0, bracket=None):
    """Smooth y(ln q) near qa: a quadratic through the q-search points within
    |ln(q/qa)| <= 0.3 (0.5 if fewer than 4), linear with 2-3 points. The window
    is widened to `need` (the |ln q| span of the tested q range) when the range is
    wider (HI Pup, q 0.19 +- 0.06), so its ends are not left outside the fit. The
    q-search's single Nelder-Mead fits carry mesh noise (Entry 93), which
    point-to-point interpolation would turn into spurious slopes.
    With `bracket` = (ln q_lo/qa, ln q_hi/qa) (partial-eclipse band, Entry 157) the
    window grows (0.5, 0.7, 0.9, 1.2) until its points also bracket both ends, so
    the wide band is not left outside the fit.
    Returns (callable, number of points, rms scatter about the fit)."""
    x = np.log(qs / qa)
    if bracket is None:
        widths = [w for w in (0.3, 0.5, 0.8) if w >= need] or [need + 1e-6]
    else:
        widths = [w for w in (0.5, 0.7, 0.9, 1.2) if w >= need] or [need + 1e-6]
    for width in widths:
        m = np.abs(x) <= width
        if m.sum() >= 4 and (bracket is None or
                             (x[m].min() <= bracket[0] + 1e-9 and x[m].max() >= bracket[1] - 1e-9)):
            break
    deg = 2 if m.sum() >= 4 else 1
    if m.sum() < 2:
        return (lambda q: float("nan")), int(m.sum()), float("nan")
    c = np.polyfit(x[m], ys[m], deg)
    rms = float(np.sqrt(np.mean((np.polyval(c, x[m]) - ys[m]) ** 2)))
    lo, hi = x[m].min(), x[m].max()

    def f(q):
        xq = math.log(q / qa)
        return float(np.polyval(c, xq)) if lo - 1e-9 <= xq <= hi + 1e-9 else float("nan")
    return f, int(m.sum()), rms


def main() -> int:
    catalogue = sys.argv[1] if len(sys.argv) > 1 else os.path.join(paths.TARGETS_DIR, "phaseF_queue.csv")
    with open(catalogue, encoding="utf-8") as handle:
        targets = [(r["name"], r["type"]) for r in csv.DictReader(handle)]
    with open(os.path.join(paths.TARGETS_DIR, "contact_q_literature.csv"), encoding="utf-8") as handle:
        lit = {r["name"]: r for r in csv.DictReader(handle)}
    factor, n_part, n_comp, f_comp = partial_factor()
    print(f"Partial-eclipse factor F = {factor:.3f} ({n_part} spectroscopic stars; complete-eclipse "
          f"68th percentile {f_comp:.3f} from {n_comp} stars, band kept at +-{WEAK_BAND:.0%})")
    out = []
    for star, gtype in targets:
        if morphology.resolve_morphology_kind(star, gtype) != morphology.CONTACT:
            continue
        adopted = morphology.read_q_adopted(paths.QSEARCH_DIR, star)
        if not adopted:
            continue
        qa = adopted["q_adopted"]
        source = adopted.get("q_source", "")
        w_type = qa > 1
        i_ad = adopted.get("incl_at_adopted")
        margin = totality_margin(qa, i_ad) if i_ad else float("nan")
        eclipse = "complete" if margin > 0 else ("partial" if margin <= 0 else "")
        bracket = None
        if source == "literature_spectroscopic":
            ql, qe = float(lit[star]["q_lit"]), float(lit[star]["q_err"] or "nan")
            if math.isnan(qe):          # no error in our table: assume 5 % (median published 3.5 %)
                qe = round(ASSUMED_SPEC_FRAC * ql, 4)
                note = "error assumed 5 %"
            else:
                note = ""
            if w_type:
                lo_hi, basis = (1 / (ql + qe), 1 / (ql - qe)), f"spectroscopic {ql} +- {qe} (1/q)"
            else:
                lo_hi, basis = (ql - qe, ql + qe), f"spectroscopic {ql} +- {qe}"
            if note:
                basis += f"; {note}"
        elif eclipse == "complete":
            lo_hi, basis = (qa * (1 - WEAK_BAND), qa * (1 + WEAK_BAND)), "q-search +-15 % (complete eclipses)"
        else:
            lo_hi = (qa / factor, qa * factor)
            basis = f"q-search x/: {factor:.2f} (partial eclipses)"
            bracket = (math.log(lo_hi[0] / qa), math.log(lo_hi[1] / qa))
        qs, inc, fil = read_curve(star)
        need = max(abs(math.log(lo_hi[0] / qa)), abs(math.log(lo_hi[1] / qa)))
        fi, n_pts, rms_i = local_fit(qs, inc, qa, need, bracket)
        ff, _, rms_f = local_fit(qs, fil, qa, need, bracket)
        i0, f0 = fi(qa), ff(qa)
        i_lo, i_hi = fi(lo_hi[0]), fi(lo_hi[1])
        f_lo, f_hi = ff(lo_hi[0]), ff(lo_hi[1])
        # one-sided when an end lies beyond the q values tried (e.g. V857 Her at the grid edge)
        di = np.nanmax([abs(i_lo - i0), abs(i_hi - i0)]) if not (math.isnan(i_lo) and math.isnan(i_hi)) else float("nan")
        df = np.nanmax([abs(f_lo - f0), abs(f_hi - f0)]) if not (math.isnan(f_lo) and math.isnan(f_hi)) else float("nan")
        fl, fh = adopted.get("q_search_lo"), adopted.get("q_search_hi")
        _ends = [abs(fi(fl) - i0), abs(fi(fh) - i0)] if fl and fh else []
        di_formal = float(np.nanmax(_ends)) if any(not math.isnan(v) for v in _ends) else float("nan")
        # nearest q-search point on each side of q_adopted: how far the interpolation reaches
        gap = min(abs(math.log(x / qa)) for x in qs if abs(x - qa) > 1e-6 * qa)
        out.append({"star": star, "q_adopted": qa, "q_source": source, "basis": basis,
                    "q_lo": round(lo_hi[0], 4), "q_hi": round(lo_hi[1], 4),
                    "incl_at_q": round(i0, 2), "incl_lo": round(i_lo, 2), "incl_hi": round(i_hi, 2),
                    "incl_sys_q": round(di, 2),
                    "fillout_at_q": round(f0, 3), "fillout_lo": round(f_lo, 3), "fillout_hi": round(f_hi, 3),
                    "fillout_sys_q": round(df, 3),
                    "incl_sys_formal_interval": round(di_formal, 2),
                    "nearest_grid_dlnq": round(gap, 3), "n_points_fit": n_pts,
                    "incl_scatter_about_fit": round(rms_i, 2), "fillout_scatter_about_fit": round(rms_f, 3),
                    "eclipse": eclipse, "totality_margin": round(margin, 3)})
    path = os.path.join(AUDIT_DIR, "q_systematics.csv")
    with open(path, "w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"{'star':<10} {'source':<5} {'q':>7} {'q range':>16} {'i':>6} {'i_sys':>6} {'f':>6} {'f_sys':>6} "
          f"{'i_sys formal':>12} {'n':>3} {'rms_i':>5}")
    for r in out:
        print(f"{r['star']:<10} {'spec' if r['q_source'].startswith('lit') else 'qs':<5} {r['q_adopted']:>7.4g} "
              f"{r['q_lo']:>7.4g}-{r['q_hi']:<8.4g} {r['incl_at_q']:>6} {r['incl_sys_q']:>6} "
              f"{r['fillout_at_q']:>6} {r['fillout_sys_q']:>6} {r['incl_sys_formal_interval']:>12} "
              f"{r['n_points_fit']:>3} {r['incl_scatter_about_fit']:>5}{'  (assumed error)' if 'assumed' in r['basis'] else ''}")
    print(f"\n{len(out)} contact stars. Wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
