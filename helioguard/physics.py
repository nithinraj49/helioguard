"""Orbit decay physics for a circular low-Earth orbit under drag (+ optional thrust).

Energy balance: dE/dt = F_t * v  ->  da/dt = 2 F_t a^1.5 / (m sqrt(mu))
Phase 2 replaces this with full propagation (Orekit) for elliptical orbits.
"""
import math
from dataclasses import dataclass
from typing import Callable

from .atmosphere import density

MU = 3.986004418e14      # m^3/s^2
R_EARTH = 6378.137e3     # m
REENTRY_ALT_KM = 150.0   # below this, decay is effectively unrecoverable


@dataclass
class Spacecraft:
    name: str
    mass_kg: float
    cd: float
    area_m2: dict          # attitude -> cross-section area (m^2)
    thrust_n: float        # available thrust (N)


# density_fn(alt_km, hours_since_start) -> kg/m^3 replaces the reference atmosphere when given.
DensityFn = Callable[[float, float], float]


def drag_force(alt_km: float, area_m2: float, cd: float, storm_mult: float,
               density_fn: DensityFn | None = None, t_h: float = 0.0) -> float:
    a = R_EARTH + alt_km * 1e3
    v = math.sqrt(MU / a)
    rho = density_fn(alt_km, t_h) if density_fn else density(alt_km, storm_mult)
    return 0.5 * rho * v * v * cd * area_m2


def simulate(sc: Spacecraft, alt_km: float, attitude: str, storm_mult: float,
             hours: float, thrusting: bool = False, dt_s: float = 60.0,
             density_fn: DensityFn | None = None) -> dict:
    """Propagate altitude. Returns final altitude and whether/when reentry occurs."""
    area = sc.area_m2[attitude]
    a = R_EARTH + alt_km * 1e3
    t, t_end = 0.0, hours * 3600.0
    while t < t_end:
        alt = (a - R_EARTH) / 1e3
        if alt < REENTRY_ALT_KM:
            return {"final_alt_km": alt, "reentered": True, "reentry_hour": t / 3600}
        f = -drag_force(alt, area, sc.cd, storm_mult, density_fn, t / 3600)
        if thrusting:
            f += sc.thrust_n
        a += 2 * f * a ** 1.5 / (sc.mass_kg * math.sqrt(MU)) * dt_s
        t += dt_s
    return {"final_alt_km": (a - R_EARTH) / 1e3, "reentered": False, "reentry_hour": None}


def can_raise(sc: Spacecraft, alt_km: float, attitude: str, storm_mult: float,
              density_fn: DensityFn | None = None) -> bool:
    """Orbit raising is possible only if thrust exceeds drag in that attitude (at the start time)."""
    return sc.thrust_n > drag_force(alt_km, sc.area_m2[attitude], sc.cd, storm_mult, density_fn)
