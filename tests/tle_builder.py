"""Build TLE lines (SYNTHETIC test data) from a mean motion, with valid checksums."""
import math
from datetime import datetime, timedelta

from helioguard.atmosphere import density
from helioguard.data.celestrak import parse_tle_history
from helioguard.physics import MU, R_EARTH

B_TRUE = 0.0338                         # Cd*A/m, m^2/kg  (2.2*4/260, the illustrative craft)


def checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in line[:68]) % 10


def make_tle(norad: int, year2: int, epoch_day: float, n_rev_day: float,
             inclination: float = 53.2) -> tuple[str, str]:
    l1 = f"1 {norad:05d}U 22010F   {year2:02d}{epoch_day:012.8f}  .00010000  00000+0  50000-3 0  999"
    l2 = (f"2 {norad:05d} {inclination:8.4f} 341.0000 0001500  95.0000 265.0000 "
          f"{n_rev_day:11.8f}    1")
    l1, l2 = l1.ljust(68)[:68], l2.ljust(68)[:68]
    return l1 + str(checksum(l1)), l2 + str(checksum(l2))


def n_from_alt(alt_km: float) -> float:
    a = R_EARTH + alt_km * 1e3
    return math.sqrt(MU / a ** 3) * 86400 / (2 * math.pi)


def synth_history(norad, start: datetime, k_by_hour, start_alt=330.0, hours=72, step_h=8):
    """SYNTHETIC history: integrate da/dt = -rho*B*sqrt(mu*a) with rho = k(t) * reference
    atmosphere and emit a TLE every `step_h` hours starting at `start` (UTC)."""
    a, t, text, dt = R_EARTH + start_alt * 1e3, 0.0, [], 60.0
    while t <= hours * 3600 + 1:
        if abs(t / 3600 - round(t / 3600 / step_h) * step_h) < 1e-9:
            when = start + timedelta(seconds=t)
            day = when.timetuple().tm_yday + (when.hour + when.minute / 60) / 24
            l1, l2 = make_tle(norad, when.year % 100, day, n_from_alt((a - R_EARTH) / 1e3))
            text += [f"SYNTH-{norad}", l1, l2]
        alt = (a - R_EARTH) / 1e3
        a -= k_by_hour(t / 3600) * density(alt, 1.0) * B_TRUE * math.sqrt(MU * a) * dt
        t += dt
    return parse_tle_history(chr(10).join(text) + chr(10))
