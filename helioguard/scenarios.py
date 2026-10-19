"""Historical storm replays.

Feb 2022 Starlink event (Group 4-7): 49 satellites deployed Feb 3 at ~210 km;
a geomagnetic storm on Feb 4 raised drag up to 50% (per SpaceX); up to 40 lost.
Spacecraft parameters below are ILLUSTRATIVE (tuned to reproduce the event's
qualitative physics), not SpaceX specifications.
"""
from .physics import Spacecraft

STARLINK_LIKE = Spacecraft(
    name="Starlink-like (illustrative)",
    mass_kg=260.0,
    cd=2.2,
    area_m2={"raise": 4.0, "edge_on": 1.5},
    thrust_n=0.07,
)

FEB_2022 = {
    "id": "starlink_feb_2022",
    "alt_km": 210.0,
    "kp": 5,                 # minor (G1) storm
    "storm_multiplier": 1.5, # SpaceX: drag up to 50% higher
    "horizon_h": 48,
}

FEB_2022_QUIET = {**FEB_2022, "id": "same_orbit_no_storm", "kp": 2, "storm_multiplier": 1.0}
