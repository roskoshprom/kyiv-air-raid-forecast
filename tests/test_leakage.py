"""Leakage tests: features at t must depend only on what is known at t.

Run with `pytest` or plainly: `python tests/test_leakage.py`.
"""
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import load_clean_events                           # noqa: E402
from features import FEATURES, build_features, build_target  # noqa: E402

FAKE_FUTURE = pd.Timedelta("3650D")


@lru_cache(maxsize=1)
def events():
    ev, _, _ = load_clean_events()
    return ev


def close(a, b):
    if isinstance(b, float) and math.isnan(b):
        return isinstance(a, float) and math.isnan(a)
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)


def known_at(ev, t):
    """What an observer sees at t: alerts started before t; all-clears after t replaced
    by an absurd fake value. If features use only knowable info, this changes nothing."""
    ev = ev[ev.started_at < t].copy()
    ev.loc[ev.finished_at > t, "finished_at"] = t + FAKE_FUTURE
    return ev


def sample_times(ev, n_random=300, seed=42):
    """Random times, hourly grid times, and edge cases taken from real alerts."""
    rng = np.random.default_rng(seed)
    lo, hi = ev.started_at.min().value, ev.finished_at.max().value
    rand = pd.to_datetime(rng.integers(lo, hi, n_random), utc=True)
    idx = rng.choice(len(ev), 30, replace=False)
    s, e = ev.started_at.iloc[idx], ev.finished_at.iloc[idx]
    edge = [*s, *(s + pd.Timedelta("1ns")), *(s + (e - s) / 2), *e, *(e + pd.Timedelta("1ns"))]
    return pd.DatetimeIndex([*rand, *rand.floor("h"), *edge]).unique().sort_values()


def test_features_unchanged_when_future_removed():
    ev = events()
    ts = sample_times(ev)
    full = build_features(ev, ts)
    ongoing_cases = 0
    for t in ts:
        trunc = build_features(known_at(ev, t), [t])
        pd.testing.assert_frame_equal(full.loc[[t], FEATURES], trunc[FEATURES],
                                      check_dtype=False, obj=f"features at {t}")
        ongoing_cases += int(full.at[t, "active_now"])
    assert ongoing_cases >= 20, "test must cover t inside ongoing alerts"
    return len(ts), ongoing_cases


def naive_features(ev, t):
    """Slow, obvious reference implementation for a single t."""
    past = ev[ev.started_at < t]
    closed = past[past.finished_at <= t]
    ongoing = past[past.finished_at > t]
    r = {"active_now": int(len(ongoing) > 0),
         "active_elapsed_min": (t - ongoing.started_at.iloc[-1]).total_seconds() / 60
         if len(ongoing) else 0.0,
         "since_last_start_h": (t - past.started_at.max()).total_seconds() / 3600
         if len(past) else float("nan"),
         "since_last_end_h": (t - closed.finished_at.max()).total_seconds() / 3600
         if len(closed) else float("nan")}
    for name, w in [("starts_1h", "1h"), ("starts_3h", "3h"), ("starts_6h", "6h"),
                    ("starts_24h", "24h"), ("starts_7d", "7D"), ("starts_30d", "30D")]:
        r[name] = int((past.started_at >= t - pd.Timedelta(w)).sum())
    return r


def test_matches_naive_reference():
    ev = events()
    ts = sample_times(ev, n_random=150, seed=7)
    fast = build_features(ev, ts)
    for t in ts:
        for k, v in naive_features(ev, t).items():
            assert close(fast.at[t, k], v), (t, k, fast.at[t, k], v)
    return len(ts)


def test_alert_starting_exactly_at_t():
    """An alert starting exactly at t is NOT a feature at t, but IS the target at t."""
    t = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    ev = pd.DataFrame({"started_at": [t - pd.Timedelta("3h"), t],
                       "finished_at": [t - pd.Timedelta("2h"), t + pd.Timedelta("30min")]})
    f = build_features(ev, [t]).iloc[0]
    assert f.starts_3h == 1 and f.starts_1h == 0 and f.active_now == 0
    assert close(f.since_last_start_h, 3.0)
    assert build_target(ev, [t]).iloc[0].y == 1
    assert build_target(ev, [t - pd.Timedelta("1h")]).iloc[0].y == 0


def test_ongoing_alert_known_but_end_unknown():
    t = pd.Timestamp("2026-01-01 10:00", tz="UTC")
    ev = pd.DataFrame({"started_at": [t - pd.Timedelta("40min")],
                       "finished_at": [t + pd.Timedelta("5h")]})
    f = build_features(ev, [t]).iloc[0]
    assert f.active_now == 1 and close(f.active_elapsed_min, 40.0)
    assert math.isnan(f.since_last_end_h)        # no all-clear observed yet


if __name__ == "__main__":
    n, ongoing = test_features_unchanged_when_future_removed()
    print(f"PASS truncation invariance: {n} times checked, {ongoing} inside ongoing alerts")
    print(f"PASS matches naive reference: {test_matches_naive_reference()} times checked")
    test_alert_starting_exactly_at_t(); print("PASS alert starting exactly at t")
    test_ongoing_alert_known_but_end_unknown(); print("PASS ongoing alert: end unknown")
