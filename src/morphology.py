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
    # "FI Boo": DETACHED,   # DETECTOR FALSE POSITIVE — light curve is clearly EW/contact
    #                       # (continuous variation + O'Connell); kept CONTACT. See log.
}

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
