"""
================================================================================
SCRIPT 01 — TARGET SELECTION & CATALOGUE CROSS-REFERENCING
================================================================================
Summary:
    Identifies "neglected" short-period eclipsing binary (EB) systems that are
    suitable for thermodynamic analysis using TESS space photometry. A system is
    defined as neglected when its most recently recorded eclipse timing, as held
    in the AAVSO Variable Star Index (VSX), is older than a configurable
    threshold — meaning the historical Kreiner (2004) linear ephemeris can no
    longer reliably predict the true orbital phase.

    Three criteria must all be satisfied for a system to be selected:
        (a) Neglected status  : VSX epoch older than MIN_YEARS_NEGLECTED years.
        (b) TESS coverage     : ≥1 TESS sector found in the MAST archive.
        (c) Morphological type: GCVS confirms the system as EW or EB.

    Candidates are drawn from the Kreiner (2004) VizieR catalogue
    (J/AcA/54/207), filtered locally to P < MAX_PERIOD_DAYS before any
    remote cross-referencing begins.

Inputs:
    None — all data is fetched live from VizieR, AAVSO VSX, and MAST.
    A hard-coded fallback list is used if VizieR is unreachable.

Outputs:
    neglected_targets.csv  — Selected systems passed downstream to Script 02.
    rejected_targets.csv   — Full rejection log with per-criterion reasons,
                             formatted for direct inclusion as a thesis addendum.

Dependencies:
    pip install requests astroquery astropy lightkurve

References:
    Kreiner (2004), Acta Astronomica 54, 207 — historical ephemerides baseline.
    Samus et al. (2017), Astronomy Reports 61, 80 — GCVS v5.1.
    Watson et al. (2006), SASS 25, 47 — AAVSO VSX.
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import time
import warnings
from datetime import datetime, timezone
from typing import Optional

# ── Astronomy libraries ───────────────────────────────────────────────────────
import astropy.units as u
import lightkurve as lk
from astroquery.vizier import Vizier

# Suppress non-critical deprecation warnings from astroquery and astropy
# so that progress messages remain readable during long catalogue queries.
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1 — CONFIGURATION
# All tunable parameters live here so that adapting this pipeline to a
# different science case requires editing only this block.
# =============================================================================

# GCVS morphological type codes accepted for this study.
# "EW" = W Ursae Majoris (contact); "EB" = Algol / beta-Lyrae (semi-detached).
ACCEPTED_VARIABLE_TYPES: list[str] = ["EW", "EB"]

# Systems with P ≥ this threshold are excluded: longer-period binaries tend to
# have better-maintained ephemerides and are less likely to be in deep contact.
MAX_ORBITAL_PERIOD_DAYS: float = 1.0

# A system is "neglected" if its last VSX epoch is older than this many years.
# Increasing this makes the selection more conservative (fewer candidates).
MIN_YEARS_SINCE_LAST_EPOCH: int = 10

# VizieR catalogue identifier for Kreiner (2004) linear ephemerides.
KREINER_2004_CATALOG_ID: str = "J/AcA/54/207"

# VizieR catalogue identifier for the GCVS (Samus et al. 2017, v5.1).
GCVS_CATALOG_ID: str = "B/gcvs/gcvs_cat"

# Capture the current UTC year once at startup to avoid repeated syscalls.
CURRENT_YEAR: int = datetime.now(timezone.utc).year

# JD of the Besselian epoch B1900.0, used to convert Julian Date → year.
# Formula: year ≈ 1900 + (JD − JD_B1900) / 365.25
JD_AT_B1900: float = 2_415_020.31352


# =============================================================================
# SECTION 2 — KREINER (2004) CATALOGUE DOWNLOAD
# =============================================================================

print("=" * 70)
print("STEP 1: Querying Kreiner (2004) eclipsing binary catalogue via VizieR")
print("=" * 70)

# ROW_LIMIT = 5000 is safely above the ~3000 entries in the catalogue.
# Wildcard columns ("**") returns all available fields without pre-specifying
# column names, which may differ between VizieR mirror servers.
Vizier.ROW_LIMIT = 5000
Vizier.columns = ["**"]

try:
    kreiner_result = Vizier.get_catalogs(KREINER_2004_CATALOG_ID)
    kreiner_table  = kreiner_result[0]
    print(f"  Retrieved {len(kreiner_table)} systems from Kreiner (2004).")

    # Filter locally rather than via VizieR server-side ROW_LIMIT to ensure
    # we never silently lose systems at the server's hard truncation boundary.
    print(f"\nSTEP 2: Filtering by orbital period (P < {MAX_ORBITAL_PERIOD_DAYS} d)...")
    short_period_mask = kreiner_table["Per"] < MAX_ORBITAL_PERIOD_DAYS
    short_period_table = kreiner_table[short_period_mask]
    print(f"  Retained {len(short_period_table)} systems with P < {MAX_ORBITAL_PERIOD_DAYS} d.")

    # Build a normalised candidate list. The 'type' field is populated later
    # from the GCVS; "Pending" acts as a sentinel for unresolved entries.
    candidates: list[dict] = [
        {
            "name":   " ".join(str(row["Name"]).split()),  # collapse internal whitespace
            "type":   "Pending",
            "period": float(row["Per"]),
        }
        for row in short_period_table
    ]
    print(f"  Built candidate list: {len(candidates)} systems for cross-referencing.")

except Exception as kreiner_error:
    # Offline fallback — allows the rest of the script to be tested without
    # a live VizieR connection. Replace with your own offline list as needed.
    print(f"  VizieR query failed: {kreiner_error}")
    print("  Falling back to hard-coded demonstration list.")
    candidates = [
        {"name": "V523 Cas",  "type": "EW", "period": 0.2337},
        {"name": "AW Lac",    "type": "EW", "period": 0.3087},
        {"name": "V776 Cas",  "type": "EW", "period": 0.4405},
        {"name": "SW Lac",    "type": "EW", "period": 0.3207},
        {"name": "V1191 Cyg", "type": "EW", "period": 0.3133},
    ]


# =============================================================================
# SECTION 3 — CATALOGUE CROSS-REFERENCE HELPERS (GCVS · VSX · MAST)
# =============================================================================

print("\nSTEP 3: Preparing GCVS, VSX, and MAST helper functions...")


def query_gcvs(star_name: str) -> Optional[object]:
    """
    Look up a star in the GCVS via a 5-arcsecond positional cone search.

    The 5-arcsecond radius is intentionally tight to prevent spurious matches
    with unrelated nearby variables in crowded fields. Vizier resolves the
    star name to equatorial coordinates via SIMBAD before searching.

    Parameters
    ----------
    star_name:
        Standard variable star designation, e.g. "AB And".

    Returns
    -------
    The first matching GCVS catalogue row (an astropy Row), or None if the
    star is not found within the search radius.
    """
    try:
        gcvs_result = Vizier.query_object(
            star_name,
            catalog=GCVS_CATALOG_ID,
            radius=5 * u.arcsec,
        )
        if gcvs_result and len(gcvs_result[0]) > 0:
            return gcvs_result[0][0]  # closest positional match
        return None
    except Exception:
        return None


def query_vsx(star_name: str) -> Optional[dict]:
    """
    Query the AAVSO Variable Star Index (VSX) REST API by star name.

    The VSX epoch field (Julian Date) tells us how recently the star's
    timing has been measured, which drives the "neglected" classification.

    Parameters
    ----------
    star_name:
        Standard variable star designation.

    Returns
    -------
    A dictionary of VSX fields, or None if the star is not found or the
    API request fails.
    """
    import requests  # local import to avoid forcing the dependency at module level

    # URL-encode the name to handle spaces and special characters in
    # variable star designations (e.g. "V* AB And").
    encoded_name = requests.utils.quote(star_name)
    api_url = (
        "https://www.aavso.org/vsx/index.php"
        f"?view=api.object&format=json&name={encoded_name}"
    )
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            vsx_object = response.json().get("VSXObject", None)
            # The API returns a list for ambiguous name matches but a dict
            # when exactly one star is found; handle both forms.
            if isinstance(vsx_object, list) and len(vsx_object) > 0:
                return vsx_object[0]
            elif isinstance(vsx_object, dict):
                return vsx_object
        return None
    except Exception:
        return None


def classify_neglected_status(
    star_name: str,
    vsx_data: Optional[dict],
) -> tuple[bool, str]:
    """
    Determine whether a star qualifies as "neglected" from its VSX record.

    Neglected status is True when the most recent timing epoch is older than
    MIN_YEARS_SINCE_LAST_EPOCH years, or when no parseable epoch exists at all
    (implying no precision timing has ever been published to VSX).

    Julian Date → calendar year conversion:
        year ≈ 1900 + (JD − JD_B1900) / 365.25

    Parameters
    ----------
    star_name:
        Used only for diagnostic logging messages.
    vsx_data:
        Dictionary returned by query_vsx(), or None if VSX has no record.

    Returns
    -------
    is_neglected:
        True if the star qualifies as neglected under Criterion (a).
    reason:
        Plain-English justification written for the thesis rejection log.
    """
    if vsx_data is None:
        return True, "No VSX entry — no modern timing data on record."

    period_vsx = vsx_data.get("Period", None)
    epoch_str  = vsx_data.get("Epoch",  None)

    if period_vsx is None and epoch_str is None:
        return True, "VSX entry exists but contains no period or epoch — effectively neglected."

    if epoch_str:
        try:
            # VSX sometimes appends an uncertainty bracket: "2459001.3(5)".
            # Strip the bracket so float() succeeds.
            epoch_jd   = float(str(epoch_str).split("(")[0])
            epoch_year = 1900.0 + (epoch_jd - JD_AT_B1900) / 365.25
            years_ago  = CURRENT_YEAR - epoch_year

            if years_ago < MIN_YEARS_SINCE_LAST_EPOCH:
                return (
                    False,
                    f"Last epoch ~{epoch_year:.0f} ({years_ago:.0f} yr ago) — "
                    f"within the {MIN_YEARS_SINCE_LAST_EPOCH}-year threshold; not neglected.",
                )
            else:
                return (
                    True,
                    f"Last epoch ~{epoch_year:.0f} ({years_ago:.0f} yr ago) — "
                    f"exceeds the {MIN_YEARS_SINCE_LAST_EPOCH}-year threshold; qualifies as neglected.",
                )
        except (ValueError, TypeError):
            pass  # fall through to the unparseable-epoch case below

    return True, "VSX epoch could not be parsed — recency unverifiable; treating as neglected."


def check_tess_coverage(star_name: str) -> tuple[Optional[object], list[str]]:
    """
    Search the MAST archive for TESS Target Pixel Files (TPFs) for a star.

    lightkurve resolves the star name via SIMBAD, then performs a cone search
    against the TESS Input Catalog hosted at MAST. The returned SearchResult
    object is passed directly to Script 02 for light-curve download.

    Parameters
    ----------
    star_name:
        Standard variable star designation.

    Returns
    -------
    search_result:
        The lightkurve SearchResult object, or None if no TESS data exists.
    sector_list:
        Deduplicated list of TESS sector numbers as strings, e.g. ["14","40"].
    """
    try:
        search_result = lk.search_targetpixelfile(star_name, mission="TESS")
        if len(search_result) > 0:
            # "sequence_number" is the TESS sector index in the MAST metadata table.
            sector_list = list({str(s) for s in search_result.table["sequence_number"]})
            return search_result, sector_list
        return None, []
    except Exception:
        return None, []


# =============================================================================
# SECTION 4 — MAIN SELECTION LOOP
# Three criteria are applied in the order that minimises total API round-trips:
#   (b) TESS first  — fastest to fail; skips the slower VSX and GCVS queries.
#   (a) VSX second  — single REST call; eliminates recently-timed systems.
#   (c) GCVS last   — positional VizieR query; confirms morphological type.
# =============================================================================

print("\nSTEP 4: Applying selection criteria and building target list...")

selected_targets:  list[dict] = []
rejected_targets: list[dict] = []

for candidate in candidates:

    name   = candidate["name"]
    period = candidate["period"]
    print(f"\n  Evaluating: {name}  (P = {period:.4f} d)")

    # ── CRITERION (b): TESS COVERAGE ─────────────────────────────────────────
    tess_result, observed_sectors = check_tess_coverage(name)

    if tess_result is None:
        rejection_note = (
            "REJECTED — Criterion (b) TESS coverage: "
            "No Target Pixel Files found in the MAST archive. "
            "Script 02 cannot proceed without TESS data."
        )
        print(f"    ✗ {rejection_note}")
        rejected_targets.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (b) — TESS coverage",
            "rejection_reason": rejection_note,
        })
        time.sleep(0.5)
        continue

    print(f"    ✓ Criterion (b) — TESS sectors: {', '.join(observed_sectors)}")

    # ── CRITERION (a): NEGLECTED STATUS ──────────────────────────────────────
    vsx_data = query_vsx(name)
    is_neglected, neglect_reason = classify_neglected_status(name, vsx_data)
    print(f"    VSX: {neglect_reason}")

    if not is_neglected:
        rejection_note = (
            f"REJECTED — Criterion (a) neglected status: {neglect_reason} "
            f"Only systems unupdated for ≥{MIN_YEARS_SINCE_LAST_EPOCH} yr qualify."
        )
        print(f"    ✗ {rejection_note}")
        rejected_targets.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (a) — Neglected status",
            "rejection_reason": rejection_note,
        })
        time.sleep(1.0)
        continue

    print("    ✓ Criterion (a) — system qualifies as neglected.")

    # ── CRITERION (c): GCVS MORPHOLOGICAL TYPE ────────────────────────────────
    gcvs_row = query_gcvs(name)

    if gcvs_row is None:
        rejection_note = (
            "REJECTED — Criterion (c) morphological type: "
            "No GCVS entry within 5 arcseconds. "
            "Cannot confirm EW/EB classification; excluded conservatively."
        )
        print(f"    ✗ {rejection_note}")
        rejected_targets.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (c) — GCVS morphological type",
            "rejection_reason": rejection_note,
        })
        time.sleep(1.0)
        continue

    gcvs_type = (
        str(gcvs_row["VarType"]).strip()
        if "VarType" in gcvs_row.colnames
        else "UNKNOWN"
    )
    print(f"    GCVS type: '{gcvs_type}'")

    # Accept composite type strings such as "EW+EA" by checking for substring.
    type_is_accepted = any(t in gcvs_type for t in ACCEPTED_VARIABLE_TYPES)

    if not type_is_accepted:
        rejection_note = (
            f"REJECTED — Criterion (c) morphological type: "
            f"GCVS reports '{gcvs_type}', not in accepted list {ACCEPTED_VARIABLE_TYPES}."
        )
        print(f"    ✗ {rejection_note}")
        rejected_targets.append({
            "name":             name,
            "period_days":      period,
            "rejection_stage":  "Criterion (c) — GCVS morphological type",
            "rejection_reason": rejection_note,
        })
        time.sleep(1.0)
        continue

    print(f"    ✓ Criterion (c) — GCVS type '{gcvs_type}' accepted.")

    # ── ALL CRITERIA PASSED ───────────────────────────────────────────────────
    selected_targets.append({
        "name":    name,
        "type":    gcvs_type,
        "period":  period,
        "sectors": ";".join(observed_sectors),
    })
    print(f"    ✔✔ SELECTED: {name}")
    time.sleep(1.0)


# =============================================================================
# SECTION 5 — OUTPUT FILES
# =============================================================================

SELECTED_OUTPUT_FILE = "neglected_targets.csv"
REJECTED_OUTPUT_FILE = "rejected_targets.csv"

with open(SELECTED_OUTPUT_FILE, "w", newline="") as selected_file:
    writer = csv.DictWriter(selected_file, fieldnames=["name", "type", "period", "sectors"])
    writer.writeheader()
    writer.writerows(selected_targets)

with open(REJECTED_OUTPUT_FILE, "w", newline="") as rejected_file:
    writer = csv.DictWriter(
        rejected_file,
        fieldnames=["name", "period_days", "rejection_stage", "rejection_reason"],
    )
    writer.writeheader()
    writer.writerows(rejected_targets)

print("\n" + "=" * 70)
print("TARGET SELECTION COMPLETE")
print("=" * 70)

print(f"\n  SELECTED ({len(selected_targets)} systems) → {SELECTED_OUTPUT_FILE}")
for entry in selected_targets:
    print(f"    {entry['name']:<20}  Type: {entry['type']:<10}  "
          f"P = {entry['period']:.4f} d   Sectors: {entry['sectors']}")

print(f"\n  REJECTED ({len(rejected_targets)} systems) → {REJECTED_OUTPUT_FILE}")
for entry in rejected_targets:
    print(f"    {entry['name']:<20}  Failed at: {entry['rejection_stage']}")

print("\n  The rejection log is formatted for direct inclusion as a thesis addendum.")
print("=" * 70)
