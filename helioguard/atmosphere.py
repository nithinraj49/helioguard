"""Upper-atmosphere density model.

Phase 1 uses a piecewise-exponential reference atmosphere (Vallado-style table)
with a storm multiplier. Phase 2 swaps this for `pymsis` (NRLMSIS 2.0), driven by
real F10.7 / Ap space-weather indices.
"""
import math

# (base altitude km, base density kg/m^3, scale height km)
_TABLE = [
    (150, 2.070e-09, 22.523),
    (180, 5.464e-10, 29.740),
    (200, 2.789e-10, 37.105),
    (250, 7.248e-11, 45.546),
    (300, 2.418e-11, 53.628),
    (350, 9.518e-12, 53.298),
    (400, 3.725e-12, 58.515),
    (450, 1.585e-12, 60.828),
    (500, 6.967e-13, 63.822),
    (600, 1.454e-13, 71.835),
]


def density(alt_km: float, storm_multiplier: float = 1.0) -> float:
    """Neutral density at altitude, scaled by a storm multiplier (1.0 = quiet)."""
    base = _TABLE[0]
    for row in _TABLE:
        if alt_km >= row[0]:
            base = row
    h0, rho0, H = base
    return rho0 * math.exp(-(alt_km - h0) / H) * storm_multiplier


# NOAA geomagnetic storm scale: Kp -> G-level
def kp_to_g_scale(kp: float) -> str:
    if kp < 5:
        return "G0"
    return f"G{min(int(kp) - 4, 5)}"
