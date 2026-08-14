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
    which targets appear in the table and in what order. With no argument, the
    original Andromeda cohort is used for backward compatibility.

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
    results_path = f"results_{safe_name}.csv"

    if not os.path.exists(results_path):
        return None

    data: dict[str, tuple[float, float, float]] = {}
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
    """Read target names from a catalogue CSV, or fall back to the And cohort."""
    if not catalog_path:
        return ["WZ And", "AA And", "AB And", "AD And", "BD And"]
    names: list[str] = []
    with open(catalog_path, newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            names.append(row["name"])
    return names


def render_latex_results_table(target_names: list[str]) -> None:
    """
    Print a complete LaTeX table of derived parameters for all target systems.

    Columns:
        Target | T₁ (K) | T₂ (K) | i (°) | q (m₂/m₁)

    Values carry their asymmetric 1σ MCMC uncertainties where available. A
    "[Pending Optimization]" placeholder spans the data columns for any target
    whose Script 05 results file does not yet exist.
    """
    print("\n% --- AUTOMATED RESULTS TABLE (generated by GenerateThesisTables.py) ---")
    print(r"\begin{table}[htbp]")
    print(r"\centering")
    print(r"\sffamily\small")
    print(r"\caption{Derived Physical and Thermodynamic Parameters}")
    print(r"\label{tab:results}")
    print(r"\begin{tabular}{l c c c c}")
    print(r"\toprule")
    print(r"Target & $T_1$ (K) & $T_2$ (K) & $i$ ($^\circ$) & $q\ (m_2/m_1)$ \\")
    print(r"\midrule")

    for star_name in target_names:
        parameter_data = read_results_file(star_name)

        if parameter_data is not None:
            t1_cell   = format_with_uncertainty(parameter_data.get("T1_K"),     decimals=0)
            t2_cell   = format_with_uncertainty(parameter_data.get("T2_K"),     decimals=0)
            incl_cell = format_with_uncertainty(parameter_data.get("incl_deg"), decimals=1)
            q_cell    = format_with_uncertainty(parameter_data.get("q"),        decimals=3)
            print(f"{star_name} & {t1_cell} & {t2_cell} & {incl_cell} & {q_cell} \\\\")
        else:
            # Results not yet available; insert a placeholder spanning the data
            # columns so the table remains well-formed.
            print(fr"{star_name} & \multicolumn{{4}}{{c}}{{[Pending Optimization]}} \\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    catalog        = sys.argv[1] if len(sys.argv) > 1 else None
    thesis_targets = load_target_names(catalog)
    render_latex_results_table(thesis_targets)
