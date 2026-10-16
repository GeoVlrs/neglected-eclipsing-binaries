"""
limbdark.py — linear limb-darkening coefficient for the TESS band by temperature.

Until Entry 91 every star, hot or cool, used u = 0.6. The published TESS
coefficients fall steadily with temperature: 0.63 at 4250 K, 0.49 at 6000 K,
0.42 at 7000 K, 0.33 at 10500 K. This module holds that relation so the
pipeline can enter a per-star coefficient by hand (PHOEBE's own lookup needs
atmosphere tables that the offline TESS passband file does not carry, and the
models here use blackbody atmospheres).

Source: Claret (2017, A&A 600, A30), table 24 — linear law, ATLAS models,
least-squares method, log g = 4.5, solar metallicity, microturbulence 2 km/s.
Retrieved from VizieR (J/A+A/600/A30/table24) on 2026-10-11.

The dependence on log g is weak next to the dependence on temperature (main
sequence to subgiant changes u by a few 0.01), so one log g is used.
"""
import numpy as np

# (Teff [K], u_linear) — Claret (2017) table 24, log g 4.5, [M/H] 0, xi 2 km/s, LSM.
_TESS_LINEAR = np.array([
    (3500, 0.5681), (3750, 0.5639), (4000, 0.5975), (4250, 0.6330), (4500, 0.6256),
    (4750, 0.6049), (5000, 0.5837), (5250, 0.5602), (5500, 0.5357), (5750, 0.5122),
    (6000, 0.4890), (6250, 0.4683), (6500, 0.4499), (6750, 0.4346), (7000, 0.4214),
    (7250, 0.4092), (7500, 0.3987), (7750, 0.3895), (8000, 0.3831), (8250, 0.3875),
    (8500, 0.3863), (8750, 0.3740), (9000, 0.3619), (9250, 0.3526), (9500, 0.3454),
    (9750, 0.3398), (10000, 0.3352), (10250, 0.3311), (10500, 0.3271), (10750, 0.3232),
    (11000, 0.3191), (11250, 0.3150), (11500, 0.3106), (11750, 0.3061), (12000, 0.3014),
])

LEGACY_LINEAR_LD = 0.6        # the single value used for every star before Entry 91


def tess_linear_ld(teff_k: float) -> float:
    """Linear limb-darkening coefficient in the TESS band at `teff_k` (clamped to
    the 3500-12000 K range of the table), by linear interpolation."""
    return float(np.interp(float(teff_k), _TESS_LINEAR[:, 0], _TESS_LINEAR[:, 1]))


# (Teff [K], a, b) - Claret (2017) table 25: QUADRATIC law
# I(mu)/I(1) = 1 - a (1 - mu) - b (1 - mu)^2, same grid and method as the linear
# table (ATLAS, least squares, log g 4.5, [M/H] 0, xi 2 km/s). Retrieved from VizieR
# (J/A+A/600/A30/table25) on 2026-10-13; raw copy in Runs/ldgd/. The two-parameter
# law follows the model atmospheres far better than the linear one (Claret's own
# fit quality: 0.08-0.13 against 0.19-0.40), mostly because it is darker at the limb.
_TESS_QUADRATIC = np.array([
    (3500, 0.3109, 0.3446), (3750, 0.3165, 0.3315), (4000, 0.4090, 0.2526),
    (4250, 0.5096, 0.1654), (4500, 0.5126, 0.1514), (4750, 0.4757, 0.1731),
    (5000, 0.4338, 0.2008), (5250, 0.3891, 0.2293), (5500, 0.3464, 0.2538),
    (5750, 0.3075, 0.2743), (6000, 0.2708, 0.2925), (6250, 0.2411, 0.3045),
    (6500, 0.2179, 0.3109), (6750, 0.1982, 0.3168), (7000, 0.1812, 0.3220),
    (7250, 0.1649, 0.3274), (7500, 0.1515, 0.3313), (7750, 0.1413, 0.3326),
    (8000, 0.1391, 0.3270), (8250, 0.1605, 0.3043), (8500, 0.1789, 0.2779),
    (8750, 0.1776, 0.2632), (9000, 0.1698, 0.2574), (9250, 0.1628, 0.2543),
    (9500, 0.1572, 0.2522), (9750, 0.1527, 0.2507), (10000, 0.1493, 0.2491),
    (10250, 0.1462, 0.2479), (10500, 0.1432, 0.2464), (10750, 0.1402, 0.2453),
    (11000, 0.1369, 0.2442), (11250, 0.1337, 0.2429), (11500, 0.1308, 0.2410),
    (11750, 0.1275, 0.2393), (12000, 0.1244, 0.2373),
])


def tess_quadratic_ld(teff_k: float) -> tuple:
    """Quadratic limb-darkening coefficients (a, b) in the TESS band at `teff_k`
    (clamped to the 3500-12000 K range of the table), by linear interpolation."""
    t = float(teff_k)
    return (float(np.interp(t, _TESS_QUADRATIC[:, 0], _TESS_QUADRATIC[:, 1])),
            float(np.interp(t, _TESS_QUADRATIC[:, 0], _TESS_QUADRATIC[:, 2])))
