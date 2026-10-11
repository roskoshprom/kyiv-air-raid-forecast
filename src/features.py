"""Hourly grid, target and leakage-safe features.

Forecast is made at time t (UTC, top of the hour). Information set at t:
  * alerts that STARTED strictly before t (started_at < t);
  * for those, the all-clear only if it already happened (finished_at <= t).
    If finished_at > t the alert is still ongoing at t: we know THAT it is on and
    for how long, but not when it will end. finished_at > t is only ever used through
    the comparison `finished_at <= t`, so its actual value cannot leak.

Target: y[t] = 1 if an alert starts in [t, t + 1h).
"""
import numpy as np
import pandas as pd

from data import LOCAL_TZ

HOUR = pd.Timedelta("1h")
COUNT_WINDOWS = {"starts_1h": "1h", "starts_3h": "3h", "starts_6h": "6h",
                 "starts_24h": "24h", "starts_7d": "7D",
                 "starts_30d": "30D"}
FEATURES = ["active_now", "active_elapsed_min", "since_last_start_h",
            "since_last_end_h", *COUNT_WINDOWS, "hour_local", "dow_local"]


def _ns(x):
    """Timestamps (UTC) -> int64 ns, for searchsorted."""
    return pd.DatetimeIndex(x).as_unit("ns").asi8


def build_features(events: pd.DataFrame, ts) -> pd.DataFrame:
    """Features at each forecast time in `ts`, using only what is known at that time."""
    ts = pd.DatetimeIndex(ts)
    t = _ns(ts)
    starts = np.sort(_ns(events.started_at))
    ends = np.sort(_ns(events.finished_at))

    # --- the only two ways the function touches event times -------------------
    n_started = np.searchsorted(starts, t, side="left")   # # alerts with start <  t
    n_ended = np.searchsorted(ends, t, side="right")      # # alerts with end   <= t
    # An end <= t always belongs to an alert that started < t, so the difference
    # is the number of alerts that started before t and have no all-clear yet.
    n_ongoing = n_started - n_ended

    out = pd.DataFrame(index=ts)
    out.index.name = "t"
    out["active_now"] = (n_ongoing > 0).astype(int)

    has_start = n_started > 0
    last_start = np.where(has_start, starts[np.maximum(n_started - 1, 0)], 0)
    out["since_last_start_h"] = np.where(has_start, (t - last_start) / 3.6e12, np.nan)
    # No overlapping alerts (asserted in data.region_events), so the ongoing alert,
    # if any, is the last one that started before t.
    out["active_elapsed_min"] = np.where(out.active_now == 1, (t - last_start) / 6e10, 0.0)

    has_end = n_ended > 0
    last_end = np.where(has_end, ends[np.maximum(n_ended - 1, 0)], 0)
    out["since_last_end_h"] = np.where(has_end, (t - last_end) / 3.6e12, np.nan)

    for name, w in COUNT_WINDOWS.items():
        lo = np.searchsorted(starts, t - pd.Timedelta(w).value, side="left")
        out[name] = n_started - lo                          # starts in [t - w, t)

    local = ts.tz_convert(LOCAL_TZ)
    out["hour_local"] = local.hour
    out["dow_local"] = local.dayofweek
    return out


def build_target(events: pd.DataFrame, ts) -> pd.DataFrame:
    """Labels. These columns use the future relative to t and must NEVER be features."""
    ts = pd.DatetimeIndex(ts)
    t = _ns(ts)
    h = HOUR.value
    starts = np.sort(_ns(events.started_at))
    n_new = (np.searchsorted(starts, t + h, side="left")
             - np.searchsorted(starts, t, side="left"))     # starts in [t, t + 1h)
    s_ev, e_ev = _ns(events.started_at), _ns(events.finished_at)
    # EDA only: alert active at any moment of [t, t + 1h)
    active = np.array([((s_ev < x + h) & (e_ev > x)).any() for x in t])
    return pd.DataFrame({"y": (n_new > 0).astype(int), "n_starts_next_1h": n_new,
                         "eda_active_next_1h": active.astype(int)}, index=ts)


def gap_flags(ts, gaps: pd.DataFrame) -> pd.DataFrame:
    """Flags for hours affected by country-wide data-collection gaps."""
    ts = pd.DatetimeIndex(ts)
    label_gap = np.zeros(len(ts), bool)
    hist_gap_24h = np.zeros(len(ts), bool)
    hist_gap_7d = np.zeros(len(ts), bool)
    hist_gap_30d = np.zeros(len(ts), bool)
    for g0, g1 in zip(gaps.gap_start, gaps.gap_end):
        label_gap |= (ts < g1) & (ts + HOUR > g0)                     # label window
        hist_gap_24h |= (ts - pd.Timedelta("24h") < g1) & (ts > g0)  # feature window
        hist_gap_7d |= (ts - pd.Timedelta("7D") < g1) & (ts > g0)
        hist_gap_30d |= (ts - pd.Timedelta("30D") < g1) & (ts > g0)
    return pd.DataFrame({"label_in_gap": label_gap.astype(int),
                         "history_in_gap_24h": hist_gap_24h.astype(int),
                         "history_in_gap_7d": hist_gap_7d.astype(int),
                         "history_in_gap_30d": hist_gap_30d.astype(int)}, index=ts)


def build_grid(events, gaps, start, end) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    grid = build_features(events, ts).join(build_target(events, ts)).join(gap_flags(ts, gaps))
    grid["warmup_30d"] = (ts < ts[0] + pd.Timedelta("30D")).astype(int)
    return grid
