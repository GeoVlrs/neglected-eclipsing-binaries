"""
================================================================================
AUDIT — TEST REGISTRY: WHICH TEST SETTLED WHICH MODELLING CHOICE (Entry 147)
================================================================================
Summary:
    Reads the curated registry (Audits/test_registry.csv: one row per test batch
    or audit; Audits/model_choices.csv: one row per modelling choice) and the
    model-test table (Audits/model_tests.csv), and
      * counts, per label: fits, stars, converged / capped (Nelder-Mead stopped at
        its model cap) and the dates of the result files;
      * flags settings that differ from production (drift):
          - a non-contact fit whose primary gravity-darkening exponent is not the
            ramp value at its T1 (production: the ramp; only stars inside
            6600-8000 K can differ; batches that compare the rules are skipped),
          - irradiation not recorded,
          - limb darkening other than the production linear 0.6 ("legacy"),
          - T1 different from today's value (04Modeling.GAIA_TEFF_OVERRIDE_K, else
            Audits/t1_anchor.csv);
      * checks that every model-test label has a registry row, and the reverse;
      * writes Audits/test_registry_report.md and two LaTeX tables for the thesis:
        Text/generated/tab_model_choices.tex and Text/generated/tab_test_registry.tex
        (longtable). Pipeline_Log entry numbers stay out of the thesis tables (the
        log is not published); dates are given instead.

    Read-only on everything except its three outputs. No PHOEBE.
Usage (from Code/, Windows or WSL; seconds):
    python Audits/audit_test_registry.py
Exit status 1 if a label is missing from either side.
================================================================================
"""
import csv
import datetime
import glob
import os
import re
import sys
from collections import Counter, defaultdict

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from gravdark import envelope_coeffs  # noqa: E402

AUD = os.path.join(CODE, "Audits")
GEN = os.path.join(os.path.dirname(CODE), "Text", "generated")
TOOL_LABELS_KIND = "tool test"
GD_COMPARISON_LABELS = {"A2A3", "B", "GD"}      # batches that vary the gravity-darkening rule on purpose


def rows(name):
    with open(os.path.join(AUD, name), encoding="utf-8") as h:
        return list(csv.DictReader(h))


def current_t1() -> dict:
    """{star: T1} as production uses it now: 04's override list, else t1_anchor.csv."""
    out = {r["star"]: float(r["T1_pipeline"]) for r in rows("t1_anchor.csv") if r.get("T1_pipeline")}
    src = open(os.path.join(CODE, "04Modeling.py"), encoding="utf-8").read()
    m = re.search(r"GAIA_TEFF_OVERRIDE_K[^{]*\{(.*?)\n\}", src, re.S)
    if m:
        for star, val in re.findall(r'^\s*"([^"]+)":\s*([0-9.]+)', m.group(1), re.M):
            out[star] = float(val)
    return out


def label_dates(label):
    ts = sorted(os.path.getmtime(p) for p in glob.glob(os.path.join(CODE, "Runs", "model_tests", label, "*.json")))
    if not ts:
        return ""
    d = [datetime.date.fromtimestamp(t).strftime("%d %b") for t in (ts[0], ts[-1])]
    return d[0] if d[0] == d[1] else f"{d[0]} - {d[1]}"


def strip_entries(s: str) -> str:
    """Remove Pipeline_Log entry numbers (the log is not published): "(138)",
    "(136, 139)", "137: ", "after 106"."""
    s = re.sub(r"\s*\((?:\d{2,3}(?:,\s*)?)+\)", "", s)
    s = re.sub(r"\b\d{3}:\s*", "", s)
    s = re.sub(r"\s+after \d{3}\b", "", s)
    return s


def tex(s: str) -> str:
    """Plain registry text -> LaTeX (escapes and a few symbol tokens)."""
    s = (s or "").replace("\\", "/")
    for a, b in (("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")):
        s = s.replace(a, b)
    subs = [(r"\bchi2r\b", r"$\\chi^2_\\nu$"), (r"\bchi2\b", r"$\\chi^2$"), (r"->", r"$\\to$"),
            (r"\+-", r"$\\pm$"), (r"<=", r"$\\le$"), (r">=", r"$\\ge$"), (r"(?<![\w$])~", r"$\\sim$"),
            (r"\bR-hat\b", r"$\\hat{R}$"), (r"\btau\b", r"$\\tau$"), (r"\b4 pi\b", r"$4\\pi$"),
            (r"\bk3\b", r"$k_3$"), (r"\bl3\\_frac\b", r"$l_3$"), (r"\bsigma\\_T1\b", r"$\\sigma_{T_1}$"),
            (r"\bT1\b", r"$T_1$"), (r"\bT2\b", r"$T_2$"), (r"\bdeg\b", r"$^\\circ$"),
            (r"(?<=\d) (?=\d{3}\b)", r"\\,")]
    for a, b in subs:
        s = re.sub(a, b, s)
    return s.replace(" < ", r" $<$ ").replace(" > ", r" $>$ ")


def main() -> int:
    reg = rows("test_registry.csv")
    cho = rows("model_choices.csv")
    mt = rows("model_tests.csv")
    t1now = current_t1()

    per = defaultdict(lambda: {"n": 0, "conv": 0, "cap": 0, "blank": 0, "stars": set(), "drift": Counter()})
    for r in mt:
        p = per[r["label"]]
        p["n"] += 1
        p["stars"].add(r["star"])
        c = r.get("converged", "")
        p["conv" if c == "True" else "cap" if c == "False" else "blank"] += 1
        noncontact = r["geometry"] != "contact"
        if noncontact and r["label"] not in GD_COMPARISON_LABELS and not r.get("gd_primary_teff"):
            # production: the ramp for non-contact stars. The fit's primary beta is
            # compared with the ramp value at its T1, so only stars inside 6600-8000 K,
            # where the rules differ, are flagged.
            try:
                if abs(float(r["beta1"]) - envelope_coeffs(float(r["T1"]), "ramp")[0]) > 0.01:
                    p["drift"][f"non-contact gravity darkening not the ramp ({r['star']})"] += 1
            except (KeyError, ValueError):
                pass
        if not r.get("irrad"):
            p["drift"]["irradiation not recorded"] += 1
        if r.get("ld", "legacy") != "legacy" and r["label"] not in GD_COMPARISON_LABELS | {"A2A3", "B"}:
            p["drift"]["limb darkening not legacy 0.6"] += 1
        try:
            if abs(float(r["T1"]) - t1now[r["star"]]) > 5:
                p["drift"][f"T1 differs from today's ({r['star']})"] += 1
        except (KeyError, ValueError):
            pass

    reg_labels = {r["label"] for r in reg}
    mt_labels = set(per)
    missing_reg = sorted(mt_labels - reg_labels)
    missing_mt = sorted(l for l in reg_labels - mt_labels
                        if next(r for r in reg if r["label"] == l)["kind"] in ("model_tests", TOOL_LABELS_KIND)
                        and l not in ("SMOKE05",))

    # ── report ────────────────────────────────────────────────────────────────
    L = ["# Test registry report", "",
         f"Generated {datetime.date.today()} by Audits/audit_test_registry.py from "
         "Audits/test_registry.csv, Audits/model_choices.csv and Audits/model_tests.csv "
         f"({len(mt)} fits, {len(mt_labels)} labels).", "",
         "| label | kind | dates | fits | stars | converged | capped | blank | entries | verdict | superseded by |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in reg:
        p = per.get(r["label"])
        if p:
            L.append(f"| {r['label']} | {r['kind']} | {label_dates(r['label'])} | {p['n']} | {len(p['stars'])} | "
                     f"{p['conv']} | {p['cap']} | {p['blank']} | {r['entries']} | {r['verdict']} | {r['superseded_by']} |")
        else:
            L.append(f"| {r['label']} | {r['kind']} | | | | | | | {r['entries']} | {r['verdict']} | {r['superseded_by']} |")
    L += ["", "## Settings that differ from production (drift)", ""]
    any_drift = False
    for lab in sorted(per):
        if per[lab]["drift"]:
            any_drift = True
            L.append(f"- **{lab}**: " + "; ".join(f"{k}: {v}" for k, v in sorted(per[lab]["drift"].items())))
    if not any_drift:
        L.append("- none")
    L += ["", "Curated notes on settings (registry column settings_vs_production):", ""]
    L += [f"- **{r['label']}**: {r['settings_vs_production']}" for r in reg if r["settings_vs_production"]]
    L += ["", "## Label check", "",
          f"- model-test labels without a registry row: {', '.join(missing_reg) or 'none'}",
          f"- registry model-test rows without fits: {', '.join(missing_mt) or 'none'}", ""]
    tot = Counter()
    for p in per.values():
        tot.update(conv=p["conv"], cap=p["cap"], blank=p["blank"])
    L.append(f"Totals: {len(mt)} fits; converged {tot['conv']}, capped {tot['cap']}, not recorded {tot['blank']}.")
    open(os.path.join(AUD, "test_registry_report.md"), "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")

    # ── LaTeX: model choices ──────────────────────────────────────────────────
    os.makedirs(GEN, exist_ok=True)
    T = [r"% generated by Code/Audits/audit_test_registry.py from Audits/model_choices.csv -- do not edit",
         r"\begin{table}[htbp]", r"\centering", r"\sffamily\scriptsize",
         r"\caption{Modelling choices and the tests behind them (Section~\ref{sec:tests}).}",
         r"\label{tab:model-choices}",
         r"\begin{tabular}{@{}p{0.12\linewidth}p{0.19\linewidth}p{0.25\linewidth}p{0.16\linewidth}p{0.14\linewidth}@{}}",
         r"\toprule", r"Choice & Alternatives tested & Evidence & Adopted & Enters as \\", r"\midrule"]
    for r in cho:
        T.append(" & ".join(tex(r[k]) for k in ("choice", "alternatives", "evidence", "adopted", "enters_as")) + r" \\")
    T += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(os.path.join(GEN, "tab_model_choices.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(T) + "\n")

    # ── LaTeX: appendix registry (longtable) ──────────────────────────────────
    A = [r"% generated by Code/Audits/audit_test_registry.py from Audits/test_registry.csv -- do not edit",
         r"{\sffamily\scriptsize",
         r"\begin{longtable}{@{}p{0.07\linewidth}p{0.09\linewidth}p{0.23\linewidth}p{0.06\linewidth}p{0.42\linewidth}@{}}",
         r"\caption{Test batches and audits behind the modelling choices. Fits: Nelder--Mead model "
         r"tests converged / stopped at the model cap. Tool smokes are omitted.}\label{tab:test-registry}\\",
         r"\toprule", r"Batch & Dates & Question & Fits & Verdict and decision \\", r"\midrule", r"\endfirsthead",
         r"\toprule", r"Batch & Dates & Question & Fits & Verdict and decision \\", r"\midrule", r"\endhead",
         r"\bottomrule", r"\endfoot"]
    for r in reg:
        if r["kind"] == TOOL_LABELS_KIND:
            continue
        p = per.get(r["label"])
        fits = f"{p['conv']}/{p['cap'] + p['blank']}" if p else "--"
        dates = label_dates(r["label"]) if p else "--"
        verdict = r["verdict"] + (f". Decision: {r['decision']}" if r["decision"] and r["decision"] != "(superseded)" else "")
        if r["superseded_by"]:
            verdict += f" (superseded: {r['superseded_by']})"
        verdict = strip_entries(verdict)
        A.append(" & ".join([tex(r["label"]), tex(dates), tex(r["question"]), fits, tex(verdict)]) + r" \\")
    A += [r"\end{longtable}", "}"]
    open(os.path.join(GEN, "tab_test_registry.tex"), "w", encoding="utf-8", newline="\n").write("\n".join(A) + "\n")

    print(f"{len(mt)} fits in {len(mt_labels)} labels; registry {len(reg)} rows; choices {len(cho)} rows")
    print(f"missing registry rows: {missing_reg or 'none'}; registry labels without fits: {missing_mt or 'none'}")
    for lab in sorted(per):
        if per[lab]["drift"]:
            print(f"  drift {lab}: " + "; ".join(f"{k} {v}" for k, v in sorted(per[lab]["drift"].items())))
    print("wrote Audits/test_registry_report.md, Text/generated/tab_model_choices.tex, tab_test_registry.tex")
    return 1 if (missing_reg or missing_mt) else 0


if __name__ == "__main__":
    sys.exit(main())
