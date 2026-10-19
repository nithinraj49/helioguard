"""Generate the SYNTHETIC Space-Track-style 3LE fixture used by the offline client tests.

Not real data: two fake objects (NORAD 99001, 99002) whose mean motion rises as an orbit
decays. It only exercises parsing, grouping and the client plumbing. Real Starlink Group 4-7
histories need Space-Track access.
"""
from pathlib import Path


def checksum(line: str) -> int:
    return sum(int(c) if c.isdigit() else 1 if c == "-" else 0 for c in line[:68]) % 10


def tle(norad: int, epoch_day: float, n_rev_day: float, bstar: str) -> tuple[str, str]:
    l1 = (f"1 {norad:05d}U 22010F   22{epoch_day:012.8f}  .00010000  00000+0 "
          f"{bstar} 0  999")
    l2 = (f"2 {norad:05d}  53.2000 341.0000 0001500  95.0000 265.0000 {n_rev_day:11.8f}    1")
    l1, l2 = l1.ljust(68)[:68], l2.ljust(68)[:68]
    return l1 + str(checksum(l1)), l2 + str(checksum(l2))


out = []
for norad, n0, rate in ((99001, 16.0, 0.35), (99002, 15.8, 0.20)):
    for k in range(6):                                  # six element sets, 8 h apart
        l1, l2 = tle(norad, 34.75 + k / 3.0, n0 + rate * (k * k / 8 + k / 4), " 50000-3")
        out += [f"SYNTHETIC-TEST-{norad}", l1, l2]
Path(__file__).with_name("spacetrack_synthetic_3le.txt").write_text("\n".join(out) + "\n")
