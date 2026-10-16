"""
morphology.py — the single source of truth for each star's binary morphology.

Scripts 04 (bundle construction), 05 (optimisation) and 06 (visualisation) must
agree on whether a star is modelled as a CONTACT, DETACHED or SEMI-DETACHED
binary, or a bundle would be built under one geometry and fitted or rendered
under another. Until 2026-10 each script carried its own copy of the classifier
and of MORPHOLOGY_OVERRIDE ("must stay byte-identical"), 06's copy ignored the
override table, and all three read only the GCVS base type before the slash.
That routed GCVS contact classes (EB/KE, EB/K) as detached, gave EB/SD stars no
semi-detached model, and built EW/DW ("not in contact") as contact
(Pipeline_Log Entry 76). Every script now imports this module instead.

GCVS type strings are "<light-curve type>[/<physical class>]" with ":" marking
an uncertain value, e.g. "EW/KW", "EB/SD:", "EB/KE", "EW/DW". Classification:

    physical class KE, KW, K  → contact        (GCVS "contact systems")
    physical class SD         → semidetached   (one star fills its Roche lobe)
    physical class DW         → detached       (W UMa-like but NOT in contact)
    physical class DM, DS, D  → detached
    no class: EW → contact; EA, EB → detached  (light-curve type only)

A plain EB (β Lyrae light curve) defaults to detached; when a detached fit
rails at the Roche lobe (V358 Cep, Entry 75-76) the star is moved to the
semi-detached model through MORPHOLOGY_OVERRIDE.
"""

CONTACT = "contact"
DETACHED = "detached"
SEMIDETACHED = "semidetached"
KINDS = (CONTACT, DETACHED, SEMIDETACHED)

# Per-star override, consulted BEFORE the GCVS classifier. Value: one of KINDS.
# Workflow: add here → re-run Script 04 (rebuilds the bundle) → clear
# results_<star>.csv (or let 05's staleness check refit it) → re-run 05.
MORPHOLOGY_OVERRIDE: dict[str, str] = {
    # Detached fit railed: r2 = 1.0015 × Eggleton lobe, i at 90°, χ²ν 24, eclipses
    # far wider than the model (Pipeline_Log Entries 75-76). Cool, lobe-filling
    # secondary beside a hotter primary — the semi-detached (Algol) configuration.
    "V358 Cep": SEMIDETACHED,
    # Same signature (Entry 84): live walkers pinned at r2 = 1.002 × Eggleton lobe
    # with q driven to 0.988 (prior edge 1.0) to enlarge the lobe; the pre-audit run
    # also ended lobe-filling (r2/a 0.3598 at q 0.795). T2 ≈ 4340 K vs T1 6060 K,
    # r1/a ≈ 0.14, i ≈ 88.5° — cool lobe-filling secondary, Algol configuration.
    "OT Cep": SEMIDETACHED,
    # "FI Boo": DETACHED,   # DETECTOR FALSE POSITIVE — light curve is clearly EW/contact
    #                       # (continuous variation + O'Connell); kept CONTACT. See log.
}

# Upper prior bound on the contact fill-out factor. 0.50 until Entry 90; published
# fill-outs reach 0.84-0.88 (II UMa, FN Cam, V857 Her). A mesh probe at 2000
# triangles computed every model up to 0.95 for q from 0.08 to 10.
CONTACT_FILLOUT_CEIL: float = 0.95


def read_q_adopted(qsearch_dir: str, star_name: str) -> dict:
    """
    The mass ratio adopted for a contact star by ContactQSearch.py, from
    <qsearch_dir>/q_adopted_<star>.txt ("key = value" lines). {} if there is none.

    Scripts 04 (seed), 05 (preflight, provenance) and the thesis table read it
    through this one function. q_adopted is the MODEL ratio m2/m1 with star 1 the
    hotter star eclipsed at phase 0, so a W-type system has q_adopted > 1.
    """
    import os
    path = os.path.join(qsearch_dir, f"q_adopted_{star_name.replace(' ', '_')}.txt")
    out: dict = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                out[key.strip()] = value.strip()
    except OSError:
        return {}
    for key in ("q_adopted", "q_search_min", "q_search_lo", "q_search_hi",
                "incl_at_adopted", "fillout_at_adopted", "T2_over_T1_at_adopted"):
        if key in out:
            try:
                out[key] = float(out[key])
            except ValueError:
                out.pop(key)
    out["q_weak"] = str(out.get("q_weak", "")).lower() == "true"
    return out if "q_adopted" in out else {}


def read_third_light(targets_dir: str) -> dict:
    """
    Stars modelled with third light, from <targets_dir>/third_light.csv
    (Entry 91): {star: {"mode": "free"|"fixed", "prior_max": float,
    "l3_published": float|None, "reference": str}}. {} if the file is absent.

    A third star's constant light dilutes the eclipses; a model without it
    compensates with a lower inclination. `free` fits the fraction with a
    uniform prior [0, prior_max]; `fixed` holds it at the published value.
    """
    import csv
    import os
    path = os.path.join(targets_dir, "third_light.csv")
    out: dict = {}
    try:
        with open(path, encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                mode = (row.get("mode") or "").strip().lower()
                if mode not in ("free", "fixed"):
                    continue
                try:
                    published = float(row["l3_published"])
                except (KeyError, TypeError, ValueError):
                    published = None
                if mode == "fixed" and published is None:
                    continue
                out[row["name"]] = {"mode": mode, "prior_max": float(row.get("prior_max") or 0.6),
                                    "l3_published": published,
                                    "reference": row.get("reference", "")}
    except OSError:
        return {}
    return out


_CONTACT_CLASSES = {"KE", "KW", "K"}
_DETACHED_CLASSES = {"DW", "DM", "DS", "D"}


def classify_gcvs(gcvs_type: str) -> str:
    """Morphology implied by a GCVS type string alone (see module docstring)."""
    text = (gcvs_type or "").strip().upper().replace(":", "")
    base, _, phys = text.partition("/")
    phys = phys.strip()
    if phys in _CONTACT_CLASSES:
        return CONTACT
    if phys == "SD":
        return SEMIDETACHED
    if phys in _DETACHED_CLASSES:
        return DETACHED
    return CONTACT if base.strip() == "EW" else DETACHED


def resolve_morphology_kind(star_name: str, gcvs_type: str) -> str:
    """CONTACT / DETACHED / SEMIDETACHED for a star: override first, then GCVS."""
    override = MORPHOLOGY_OVERRIDE.get(star_name)
    if override is not None:
        kind = override.strip().lower()
        if kind not in KINDS:
            raise ValueError(f"MORPHOLOGY_OVERRIDE[{star_name!r}] = {override!r}; expected one of {KINDS}")
        return kind
    return classify_gcvs(gcvs_type)


def resolve_morphology(star_name: str, gcvs_type: str) -> bool:
    """True if the star is modelled as a contact binary (back-compatible boolean)."""
    return resolve_morphology_kind(star_name, gcvs_type) == CONTACT


def is_contact_morphology(gcvs_type: str) -> bool:
    """True if the GCVS type alone implies a contact binary (no per-star override)."""
    return classify_gcvs(gcvs_type) == CONTACT


def morphology_source(star_name: str) -> str:
    """Why the morphology was chosen: 'GCVS' or 'OVERRIDE_<KIND>'."""
    override = MORPHOLOGY_OVERRIDE.get(star_name)
    return "GCVS" if override is None else "OVERRIDE_" + override.strip().upper()
