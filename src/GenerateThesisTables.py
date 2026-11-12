"""
================================================================================
UTILITY — LaTeX RESULTS TABLE GENERATOR
================================================================================
Summary:
    Reads the results_<star>.csv files produced by Script 05 and renders a
    publication-ready LaTeX table of derived physical and thermodynamic
    parameters, including the asymmetric 1σ MCMC uncertainties. If a results
    file is missing (e.g. the MCMC run has not yet completed for a target), the
    corresponding row is filled with a "[Pending Optimization]" placeholder so
    the table compiles without errors.

Inputs:
    results_<star>.csv  — Parameter output files from Script 05. The current
                          schema is:
                              Parameter, Symbol, Value, Sigma_minus, Sigma_plus, Unit
                          The older 4-column schema (no sigma columns) is still
                          read; such rows simply render without error bars.

    An optional target catalogue may be passed as the first CLI argument
    (e.g. Targets/neglected_targets_batch1.csv); its 'name' column then defines
    which targets appear in the table and in what order. With no argument, every
    star in Targets/neglected_targets_batch*.csv is listed, in catalogue order.

    Since the 2026-10 pipeline audit (Pipeline_Log Entry 76) the table states
    what each number IS, not only its value:
      * fixed parameters (T1 always; q for contact stars) are marked as fixed;
      * the morphology (C contact, D detached, SD semi-detached) is a column;
      * χ²ν and a Notes column carry the stop reason, prior railing, blends,
        SAP flux, interval clipping, and whether the row is a PRE-AUDIT result
        (no provenance rows) that the corrected pipeline has yet to replace.

Outputs:
    Prints the complete LaTeX table block to stdout. Redirect to a .tex file and
    \\input{} it in the thesis:
        python GenerateThesisTables.py Targets/neglected_targets_batch1.csv > tables/results_table.tex

Dependencies:
    Python standard library only (csv, math, os, sys).
================================================================================
"""

import csv
import math
import os
import paths
import sys
from typing import Optional


def _to_float(text: Optional[str]) -> float:
    """Parse a CSV cell to float, mapping blanks / 'nan' / junk to NaN."""
    if text is None:
        return float("nan")
    try:
        return float(text)
    except (ValueError, TypeError):
        return float("nan")


def read_results_file(star_name: str) -> Optional[dict[str, tuple[float, float, float]]]:
    """
    Parse the CSV results file for a given target system.

    Rows are indexed by the 'Symbol' column so parameters can be retrieved by
    their programmatic key regardless of row order. Each entry is returned as a
    (value, sigma_minus, sigma_plus) tuple; sigmas are NaN when the file predates
    the uncertainty columns or the parameter was held fixed.

    Parameters
    ----------
    star_name:
        Common variable star designation, e.g. "VW Cep".

    Returns
    -------
    A symbol → (value, −σ, +σ) mapping, or None if the file does not exist.
    """
    safe_name    = star_name.replace(" ", "_")
    results_path = os.path.join(paths.RESULTS_DIR, f"results_{safe_name}.csv")

    if not os.path.exists(results_path):
        return None

    data: dict[str, tuple[float, float, float]] = {}
    text: dict[str, str] = {}
    with open(results_path, "r", encoding="utf-8") as results_file:
        reader = csv.reader(results_file)
        header = next(reader, None)  # skip header row
        # Detect whether the sigma columns are present (new 6-column schema).
        has_sigmas = header is not None and len(header) >= 5

        for row in reader:
            if len(row) < 3:
                continue
            symbol = row[1]
            value  = _to_float(row[2])
            if has_sigmas and len(row) >= 5:
                sigma_minus = _to_float(row[3])
                sigma_plus  = _to_float(row[4])
            else:
                sigma_minus = sigma_plus = float("nan")
            data[symbol] = (value, sigma_minus, sigma_plus)
            text[symbol] = row[2]           # raw cell, for the non-numeric rows

    data["_text"] = text                     # type: ignore[assignment]
    return data


def format_with_uncertainty(
    entry: Optional[tuple[float, float, float]],
    decimals: int,
) -> str:
    """
    Render a (value, −σ, +σ) tuple as a LaTeX math cell.

    - Symmetric-looking errors collapse to a single $\\pm$ term.
    - Asymmetric errors use the stacked $^{+p}_{-m}$ form.
    - Missing / fixed parameters (NaN σ) render as the bare value.
    """
    if entry is None:
        return "--"
    value, minus, plus = entry
    if math.isnan(value):
        return "--"

    val_str = f"{value:.{decimals}f}"
    if math.isnan(minus) or math.isnan(plus) or (minus == 0.0 and plus == 0.0):
        return f"${val_str}$"

    # Round the two errors; if they agree to the displayed precision, use ±.
    m_str = f"{minus:.{decimals}f}"
    p_str = f"{plus:.{decimals}f}"
    if m_str == p_str:
        return f"${val_str} \\pm {p_str}$"
    return f"${val_str}^{{+{p_str}}}_{{-{m_str}}}$"


def load_target_names(catalog_path: Optional[str]) -> list[str]:
    """Target names from a catalogue CSV, or from every batch catalogue in order."""
    import glob
    paths_in = ([catalog_path] if catalog_path else
                sorted(glob.glob(os.path.join(paths.TARGETS_DIR, "neglected_targets_batch[0-9].csv"))))
    names: list[str] = []
    for path in paths_in:
        with open(path, newline="", encoding="utf-8-sig") as csv_file:
            for row in csv.DictReader(csv_file):
                if row["name"] not in names:
                    names.append(row["name"])
    return names


# Footnote letters for the Notes column (defined once in the table caption).
NOTE_LETTERS = {
    # After Phase E the only pre-audit rows left are the 22 KEPT stars (Entry 78):
    # their smeared fold cost < 3 % of the light-curve amplitude (per-star values in
    # Audits/kept_results_caveats.csv). Before Phase E finishes, re-run stars also
    # carry this note until their new results replace them.
    "pre_audit":  ("a", "pre-audit pipeline (Entry 76): folded at a slightly wrong "
                        "period, costing $<3$\\,\\% of the light-curve amplitude; "
                        "kept and documented (Entry 78)"),
    "ceiling":    ("b", "MCMC stopped at the step ceiling; the $50\\tau$ gate was not met"),
    "railed":     ("c", "posterior railed at a prior bound; quoted interval is a limit"),
    "blend":      ("d", "blended with a variable neighbour (BW/BV Dra, 16\\arcsec)"),
    "sap":        ("e", "SAP flux (PDCSAP distorted every sector)"),
    "clipped":    ("f", "error-scaled interval clipped at a prior bound"),
    # Entry 89 (multi-start Nelder-Mead, wider semi-detached q prior).
    "q_gt_1":     ("g", "$q > 1$: the lobe-filling secondary is the more massive star"),
    "near_lobe":  ("h", "primary within 3\\,\\% of its Roche lobe (near-contact)"),
    # Entry 90 (adopted mass ratio for contact stars).
    "q_weak":     ("i", "flat $q$-search curve: the photometric $q$ is poorly constrained"),
    "q_half":     ("j", "fitted with the former placeholder $q = 0.5$ (superseded)"),
}
MORPH_SHORT = {"contact": "C", "detached": "D", "semidetached": "SD"}


def q_systematics() -> dict:
    """{star: (incl_sys_q, fillout_sys_q)} from Audits/q_systematics.csv
    (Audits/audit_q_systematics.py, Entry 112): the change in i and f over the
    q range (published error, or +-15 % for a q-search value). {} if absent."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audits", "q_systematics.csv")
    out: dict = {}
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                out[row["star"]] = (_to_float(row.get("incl_sys_q")), _to_float(row.get("fillout_sys_q")))
    except OSError:
        return {}
    return out


def with_systematic(cell: str, sys_value: float, decimals: int, label: str = "q") -> str:
    r"""Append a second, systematic term to a math cell: $78.6 \pm 0.4$ -> $78.6 \pm 0.4 \pm 0.2_{q}$."""
    if cell == "--" or math.isnan(sys_value) or not cell.endswith("$"):
        return cell
    return cell[:-1] + r" \pm " + f"{sys_value:.{decimals}f}" + "_{" + label + "}$"


def t1_term(eb: dict) -> tuple:
    """(lo, hi) of the T1 term on T2 (Entry 135, 142): T2_sys_T1_lo/_hi include the
    gravity-darkening ramp, so they are asymmetric for non-contact stars inside
    6600-8000 K; the blackbody-only T2_sys_T1 is the fallback."""
    lo, hi = _to_float(eb.get("T2_sys_T1_lo")), _to_float(eb.get("T2_sys_T1_hi"))
    if math.isnan(lo) or math.isnan(hi):
        s = _to_float(eb.get("T2_sys_T1"))
        return -s, s
    return lo, hi


def with_systematic_range(cell: str, lo: float, hi: float, decimals: int, label: str) -> str:
    r"""Like with_systematic, but an asymmetric term prints as ${}^{+hi}_{-|lo|}$:
    $5259 \pm 13$ -> $5259 \pm 13\,{}^{+220}_{-129}{}_{T_1}$. Symmetric (to the
    printed precision) terms keep the $\pm$ form."""
    if cell == "--" or math.isnan(lo) or math.isnan(hi) or not cell.endswith("$"):
        return cell
    if lo <= 0.0 <= hi and f"{-lo:.{decimals}f}" == f"{hi:.{decimals}f}":
        return with_systematic(cell, hi, decimals, label)
    return (cell[:-1] + r"\,{}^{" + f"{hi:+.{decimals}f}" + "}_{" + f"{lo:+.{decimals}f}"
            + "}{}_{" + label + "}$")


def audit_rows(name: str) -> dict:
    """{star: row} from Audits/<name> (written by the audit scripts); {} if absent."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Audits", name)
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return {row["star"]: row for row in csv.DictReader(handle)}
    except OSError:
        return {}


def t1_source_mark(source: str) -> str:
    """Superscript for a T1 that is not Gaia DR3 MSC (Entry 126; t1_anchor.csv)."""
    if source.startswith("GSP-Phot"):
        return "G"
    if "LAMOST" in source:
        return "L"
    if "(SIMBAD)" in source:
        return "S"
    return ""


def notes_for(data: dict) -> list[str]:
    """Footnote keys that apply to one results file."""
    text = data.get("_text", {})
    flags = text.get("flags", "")
    keys = []
    if "input_bundle_sha" not in text:
        keys.append("pre_audit")
    if "ceiling" in text.get("stop_reason", ""):
        keys.append("ceiling")
    if "@low" in flags or "@high" in flags:
        keys.append("railed")
    if "blended_with_variable_neighbour" in flags:
        keys.append("blend")
    if "SAP_flux" in flags or text.get("flux_source", "") == "SAP_CROWDCORR":
        keys.append("sap")
    if "interval_clipped_at_bound" in flags:
        keys.append("clipped")
    if "q_gt_1_lobe_filler_more_massive" in flags:
        keys.append("q_gt_1")
    if "primary_near_roche_lobe" in flags:
        keys.append("near_lobe")
    if "q_weak_photometric" in flags:
        keys.append("q_weak")
    if "fillout_factor" in data and not text.get("q_source"):
        keys.append("q_half")
    return keys


def render_latex_results_table(target_names: list[str]) -> None:
    """
    Print a complete LaTeX table of derived parameters for all target systems.

    Columns:
        Target | Model | T₁ (K, fixed) | T₂ (K) | i (°) | q | f  or  r₁, r₂ | χ²ν | Notes

    Values carry their asymmetric 1σ MCMC uncertainties (already multiplied by
err_scale in the results file). q is FIXED for contact stars at the adopted
    value (dagger: published spectroscopic; section mark: q-search); T₁ is always fixed (Gaia DR3 or the period-colour
    estimate). A "[Pending Optimization]" placeholder spans the data columns
    for any target whose Script 05 results file does not yet exist.
    """
    used_notes: list[str] = []
    rows: list[str] = []
    q_sys = q_systematics()
    budget = audit_rows("error_budget.csv")        # Audits/audit_error_budget.py (Entry 126)
    t1_anchor = audit_rows("t1_anchor.csv")       # Audits/audit_t1_anchor.py (Entry 126)
    for star_name in target_names:
        parameter_data = read_results_file(star_name)
        if parameter_data is None:
            rows.append(fr"{star_name} & \multicolumn{{9}}{{c}}{{[Pending Optimization]}} \\")
            continue
        text  = parameter_data.get("_text", {})
        morph = text.get("morphology", "")
        if not morph:   # pre-audit files: infer from the parameters present
            morph = "contact" if "fillout_factor" in parameter_data else "detached"
        eb = budget.get(star_name, {})
        t1_cell   = format_with_uncertainty(parameter_data.get("T1_K"),     decimals=0)
        mark = t1_source_mark(t1_anchor.get(star_name, {}).get("T1_source", ""))
        if mark and t1_cell.endswith("$"):
            t1_cell = t1_cell[:-1] + "^{" + mark + "}$"
        t2_cell   = format_with_uncertainty(parameter_data.get("T2_K"),     decimals=0)
        # T2 is tied to the fixed T1 (Entry 110): second term = sigma_T1 x dT2/dT1,
        # with the gravity-darkening ramp's change where it applies (Entries 135, 142).
        t2_cell   = with_systematic_range(t2_cell, *t1_term(eb), 0, "T_1")
        ratio_cell = format_with_uncertainty(parameter_data.get("T2_over_T1"), decimals=3)
        incl_cell = format_with_uncertainty(parameter_data.get("incl_deg"), decimals=1)
        q_entry   = parameter_data.get("q")
        q_fixed   = q_entry is not None and (math.isnan(q_entry[1]) or q_entry[1] == 0.0)
        q_cell    = format_with_uncertainty(q_entry, decimals=3)
        if q_fixed and q_cell != "--":
            # Fixed (contact) q: a dagger for a published spectroscopic value, a
            # section mark for the photometric q-search minimum (Entry 90).
            # Pre-Entry-90 rows (no q_source) carry note j instead of a mark.
            mark = {"literature_spectroscopic": r"\dagger", "qsearch": r"\S"}.get(
                text.get("q_source", ""), "")
            if mark:
                q_cell = q_cell[:-1] + "^{" + mark + "}$"
        if morph == "contact":
            geom_cell = format_with_uncertainty(parameter_data.get("fillout_factor"), decimals=3)
            if star_name in q_sys:      # systematic from the fixed q (Entry 112)
                incl_cell = with_systematic(incl_cell, q_sys[star_name][0], decimals=1)
                geom_cell = with_systematic(geom_cell, q_sys[star_name][1], decimals=3)
        else:
            r1 = format_with_uncertainty(parameter_data.get("r1_frac"), decimals=4)
            r2_entry = parameter_data.get("r2_frac")
            if morph == "semidetached" and not math.isnan(_to_float(eb.get("r2"))):
                # r2 follows q through the lobe constraint: its error comes from the
                # q posterior (audit_error_budget.py), the value stays the results'.
                r2_entry = (r2_entry[0] if r2_entry else _to_float(eb.get("r2")),
                            _to_float(eb.get("r2_lo")), _to_float(eb.get("r2_hi")))
            r2 = format_with_uncertainty(r2_entry, decimals=4)
            if morph == "semidetached":
                r2 = r2[:-1] + r"\,{}^{\ddagger}$" if r2 != "--" else r2
            geom_cell = f"{r1}, {r2}"
        chi2 = parameter_data.get("chi2_red", (float("nan"),))[0]
        chi2_cell = "--" if math.isnan(chi2) else f"{chi2:.2f}"
        keys = notes_for(parameter_data)
        # Near-lobe note from the posterior (share of draws above 0.97 of the lobe).
        if _to_float(eb.get("r1_lobe_p97")) >= 0.10 and "near_lobe" not in keys:
            keys.append("near_lobe")
        for k in keys:
            if k not in used_notes:
                used_notes.append(k)
        notes_cell = ",".join(NOTE_LETTERS[k][0] for k in keys)
        rows.append(f"{star_name} & {MORPH_SHORT.get(morph, morph)} & {t1_cell} & {t2_cell} & "
                    f"{ratio_cell} & {incl_cell} & {q_cell} & {geom_cell} & {chi2_cell} & "
                    f"{notes_cell} \\\\")

    legend = "; ".join(f"$^{{{NOTE_LETTERS[k][0]}}}$ {NOTE_LETTERS[k][1]}"
                       for k in NOTE_LETTERS if k in used_notes)
    print("\n% --- AUTOMATED RESULTS TABLE (generated by GenerateThesisTables.py) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\sffamily\scriptsize")
    print(r"\caption{Derived parameters. Model: C contact, D detached, SD semi-detached. "
          r"$T_1$ is held fixed: Gaia DR3 MSC unless marked ($^{G}$\,Gaia GSP-Phot, "
          r"$^{S}$\,spectral type, $^{L}$\,LAMOST and spectral type). The light curve "
          r"measures $T_2/T_1$; $T_2$ is quoted for that $T_1$, and its second term "
          r"$\pm s_{T_1}$ is the change of $T_2$ over the uncertainty of $T_1$ at a "
          r"fixed TESS-band surface-brightness ratio (blackbody atmospheres); for "
          r"detached and semi-detached stars it includes the change of gravity darkening "
          r"along the 6600--8000\,K ramp, which can make it asymmetric. "
          r"Contact $q$ is held fixed at the adopted value: "
          r"$^{\dagger}$\,published spectroscopic mass ratio, $^{\S}$\,photometric "
          r"$q$-search minimum; $q = m_2/m_1$ with star 1 the hotter component, so "
          r"$q > 1$ marks a W-type system. "
          r"$^{\ddagger}$\,set by the Roche-lobe constraint. Uncertainties are 1$\sigma$ "
          r"MCMC intervals scaled by $\sqrt{\chi^2_\nu}$; for contact stars a second term "
          r"$\pm s_q$ on $i$ and $f$ is the systematic from the fixed $q$ (published error, or "
          r"$\pm15$\,\% for a $q$-search value). The residuals are correlated, so the "
          r"statistical terms are lower limits; Table~\ref{tab:error-budget} gives the "
          r"factor $\beta$ and the spot and limb-darkening terms. Geometry: fill-out $f$ (C) or "
          r"fractional radii $r_1, r_2$ (D, SD); a lobe-filling $r_2$ takes its error from "
          r"the $q$ posterior. "
          + (r"Notes: " + legend + "." if legend else "") + "}")
    print(r"\label{tab:results}")
    print(r"\begin{tabular}{l c c c c c c c c l}")
    print(r"\toprule")
    print(r"Target & Model & $T_1$ (K) & $T_2$ (K) & $T_2/T_1$ & $i$ ($^\circ$) & $q$ & "
          r"$f$ / $r_1, r_2$ & $\chi^2_\nu$ & Notes \\")
    print(r"\midrule")
    for row in rows:
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def render_error_budget_table(target_names: list[str]) -> None:
    """Second table (Entry 126): the terms behind each finished star's uncertainties,
    from Audits/error_budget.csv. Only stars with a current result appear."""
    budget = audit_rows("error_budget.csv")
    rows = []
    for star in target_names:
        eb = budget.get(star)
        if not eb:
            continue
        f = lambda k, d: ("--" if math.isnan(_to_float(eb.get(k))) else f"{_to_float(eb.get(k)):.{d}f}")
        contact = eb.get("morphology") == "contact"
        rng = lambda s: s.replace("..", "--") if s else "--"     # "+0.3..+1.0" -> "+0.3--+1.0"
        spot_i = rng(eb.get("spot_i_deg"))
        # l3 as a share of the system's mean light in the model (Entries 136-137);
        # error_budget.csv carries the converted values (l3_pct, l3_pct_lo/_hi)
        l3_txt = eb.get("l3", "none") or "none"
        if not math.isnan(_to_float(eb.get("l3_pct"))) and l3_txt.startswith("fitted"):
            l3 = f"{f('l3_pct', 1)} [{f('l3_pct_lo', 1)}, {f('l3_pct_hi', 1)}]"
        elif not math.isnan(_to_float(eb.get("l3_pct"))) and l3_txt.startswith("fixed"):
            l3 = f"{f('l3_pct', 1)} (fixed)"
        elif l3_txt == "none":
            l3 = "none"
        else:
            l3 = "k?"          # third light without a current k: do not print raw l3_frac
        lo, hi = t1_term(eb)
        if math.isnan(lo) or math.isnan(hi):
            t1t = "--"
        elif lo <= 0.0 <= hi and f"{-lo:.0f}" == f"{hi:.0f}":
            t1t = f"$\\pm{hi:.0f}$"
        else:
            t1t = f"${lo:+.0f}/{hi:+.0f}$"
        rows.append(
            f"{star} & {f('beta', 1)} & {f('T2_sig', 0)} & {t1t} & "
            f"{f('incl_sig', 2)} & {f('incl_sys_q', 2) if contact else '--'} & {spot_i} & "
            f"{rng(eb.get('ld_incl'))} & "
            f"{(f('fillout_sig', 3) + ' / ' + f('fillout_sys_q', 3)) if contact else '--'} & {l3} \\\\")
    if not rows:
        return
    print("\n% --- ERROR BUDGET (generated by GenerateThesisTables.py from Audits/error_budget.csv) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\sffamily\scriptsize")
    print(r"\caption{Error budget of the finished stars. $\beta$: correlated-residual factor "
          r"(time-averaging method; a lower limit), by which the statistical (MCMC) terms "
          r"should be multiplied. $T_2$: statistical term and the term from $T_1$ "
          r"(Table~\ref{tab:results}). $i$: statistical term, the term from the fixed $q$ "
          r"(contact), the shift a cool spot produces where the O'Connell effect "
          r"is $\geq 1$\,\% (sample-wide, from spot tests on eight stars), or the shift from the "
          r"star's own spotted fits where these exist, and the limb-darkening shift (linear 0.6 "
          r"against tabulated coefficients, sample-wide). $f$: statistical / fixed-$q$ terms. "
          r"$l_3$: third light in per cent of the system's mean light in the model "
          r"(fitted: median and 68\,\% interval; converted from PHOEBE's fractional "
          r"parameter), or none.}")
    print(r"\label{tab:error-budget}")
    print(r"\begin{tabular}{l c c c c c c c c l}")
    print(r"\toprule")
    print(r"Target & $\beta$ & $T_2$ stat & $T_2$ ($T_1$) & $i$ stat & $i$ ($q$) & $i$ (spot) "
          r"& $i$ (LD) & $f$ stat / ($q$) & $l_3$ (\%) \\")
    print(r"\midrule")
    for row in rows:
        print(row)
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    # Run timing → Runs/run_stats/run_summary.txt (written at exit). Inside the
    # __main__ guard so that importing this module never starts a tracker.
    import sys as _rs_sys, os as _rs_os
    _rs_sys.path.insert(0, _rs_os.path.dirname(_rs_os.path.abspath(__file__)))
    import runstats
    _run_tracker = runstats.RunTracker("GenerateThesisTables", catalogue=_rs_sys.argv[1:],
                                       per_star=False)
    catalog        = sys.argv[1] if len(sys.argv) > 1 else None
    thesis_targets = load_target_names(catalog)
    render_latex_results_table(thesis_targets)
    render_error_budget_table(thesis_targets)
