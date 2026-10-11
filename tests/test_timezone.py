"""hour_local must follow Europe/Kyiv with DST, not a fixed +3 offset."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from features import build_features  # noqa: E402

# one old alert, so the feature code has history to work with
EVENTS = pd.DataFrame({"started_at": pd.to_datetime(["2022-03-15 16:00"], utc=True),
                       "finished_at": pd.to_datetime(["2022-03-15 16:30"], utc=True)})


def hours(ts):
    return build_features(EVENTS, pd.DatetimeIndex(ts, tz="UTC")).hour_local.tolist()


def test_spring_forward_2026():
    # 2026-03-29: 03:00 EET -> 04:00 EEST (at 01:00 UTC). Local 03:00 does not exist.
    assert hours(["2026-03-29 00:00", "2026-03-29 01:00", "2026-03-29 02:00"]) == [2, 4, 5]


def test_fall_back_2025():
    # 2025-10-26: 04:00 EEST -> 03:00 EET (at 01:00 UTC). Local 03:00 happens twice.
    assert hours(["2025-10-26 00:00", "2025-10-26 01:00", "2025-10-26 02:00"]) == [3, 3, 4]


def test_offsets_winter_summer():
    assert hours(["2026-01-15 12:00"]) == [14]   # UTC+2 in winter
    assert hours(["2026-07-15 12:00"]) == [15]   # UTC+3 in summer


if __name__ == "__main__":
    test_spring_forward_2026(); test_fall_back_2025(); test_offsets_winter_summer()
    print("PASS timezone: DST transitions handled via Europe/Kyiv")
