"""
gravdark.py — gravity-darkening exponent and reflection albedo by temperature.

Stars with convective envelopes are given beta = 0.32 (Lucy 1967) and a
bolometric albedo of 0.6; stars with radiative envelopes beta = 1 (von Zeipel
1924) and albedo 1. Until Entry 91 the pipeline switched between the two at
8000 K, so every star between 6600 and 8000 K was treated as fully convective.

That range is a transition, not a regime:
  * PHOEBE's own check calls 6600-8000 K intermediate and suggests values
    between the two extremes there (phoebe/frontend/bundle.py);
  * Lucy's 0.32 was derived for 5800-6500 K;
  * Claret (1998, 2000) finds a smooth change of the exponent with temperature.

`envelope_coeffs` offers the old step, a linear ramp across the transition, and
the opposite end member (a step at 6600 K), so the three can be compared
(Pipeline_Log Entry 93). The ramp is a stated choice: no published interpolation
formula was found.
"""

CONVECTIVE_BELOW_K = 6600.0
RADIATIVE_ABOVE_K = 8000.0
BETA_CONVECTIVE, BETA_RADIATIVE = 0.32, 1.0
ALBEDO_CONVECTIVE, ALBEDO_RADIATIVE = 0.6, 1.0

MODES = ("step8000", "ramp", "step6600")


def envelope_coeffs(teff_k: float, mode: str = "ramp") -> tuple:
    """(beta, albedo) for a star of temperature `teff_k`.

    step8000: convective below 8000 K, radiative from 8000 K (the old rule).
    step6600: convective up to 6600 K, radiative above (the other end member).
    ramp:     convective up to 6600 K, radiative from 8000 K, linear between.
    """
    t = float(teff_k)
    if mode == "step8000":
        x = 1.0 if t >= RADIATIVE_ABOVE_K else 0.0
    elif mode == "step6600":
        x = 1.0 if t > CONVECTIVE_BELOW_K else 0.0
    elif mode == "ramp":
        x = min(1.0, max(0.0, (t - CONVECTIVE_BELOW_K) / (RADIATIVE_ABOVE_K - CONVECTIVE_BELOW_K)))
    else:
        raise ValueError(f"unknown mode {mode!r}; one of {MODES}")
    return (BETA_CONVECTIVE + x * (BETA_RADIATIVE - BETA_CONVECTIVE),
            ALBEDO_CONVECTIVE + x * (ALBEDO_RADIATIVE - ALBEDO_CONVECTIVE))


def rule_for(kind: str) -> str:
    """The pipeline's rule by model type (user decision 13 Oct 2026, Entry 93).

    Test round B, chi-squared with the convective setting / ramp / radiative:
      hot semi-detached: RZ Dra 117.6 / 94.9 / 108.7, GW Gem 589.5 / 554.8 / 554.3
      hot contact:       ES UMa  80.8 / 133.4 / 318.3, DN Cam 159.8 / 285 / 562
    Contact systems share a convective envelope and stay with the old rule;
    every other star follows the ramp at its own temperature.
    """
    return "step8000" if kind == "contact" else "ramp"


def coeffs_for(kind: str, teff_k: float) -> tuple:
    """(beta, albedo) under the pipeline's rule for a star of model type `kind`
    ("contact", "semidetached" or "detached")."""
    return envelope_coeffs(teff_k, rule_for(kind))


if __name__ == "__main__":
    for temp in (5000, 6600, 7200, 7329, 8000, 9000):
        print(temp, {m: tuple(round(v, 3) for v in envelope_coeffs(temp, m)) for m in MODES})
