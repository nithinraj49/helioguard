"""Pull Starlink Group 4-7 (launched 3 Feb 2022, intl. designator 2022-010) TLE histories.

Needs Space-Track credentials in `.env` (see .env.example). Output goes to the git-ignored
`.cache/spacetrack/`; do not commit Space-Track data.
"""
import sys
from datetime import date

from helioguard.data.celestrak import group_by_norad
from helioguard.data.spacetrack import (MissingCredentials, SpaceTrackClient, SpaceTrackError,
                                        credentials_from_env)
from helioguard.replays import CACHE_DIR

try:
    client = SpaceTrackClient(credentials_from_env(), CACHE_DIR / "spacetrack")
    objects = client.launch_objects("2022-010")
    ids = [int(o["NORAD_CAT_ID"]) for o in objects if o.get("OBJECT_TYPE") == "PAYLOAD"]
    print(f"{len(objects)} catalogue objects, {len(ids)} payloads for launch 2022-010")
    history = client.gp_history(ids, date(2022, 2, 3), date(2022, 2, 12))
except (MissingCredentials, SpaceTrackError) as exc:
    sys.exit(f"Group 4-7 histories NOT pulled: {exc}")

for norad, tles in sorted(group_by_norad(history).items()):
    print(f"{norad}  {tles[0].name:16s} {len(tles):3d} element sets  "
          f"{tles[0].epoch:%m-%d %H:%M} -> {tles[-1].epoch:%m-%d %H:%M}")
