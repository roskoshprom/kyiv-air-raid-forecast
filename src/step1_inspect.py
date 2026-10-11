"""Step 1: inspect the official air-raid sirens dataset.

Prints structure, date range, missing values, Kyiv City breakdown by month
and level, and duration anomalies. No transformations are saved here.
"""
import pandas as pd

from data import OFFICIAL_CSV as DATA

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", 20)


def section(title):
    print("\n" + "=" * 10 + f" {title} " + "=" * 10)


df = pd.read_csv(DATA)

section("STRUCTURE")
print(df.shape)
print(df.dtypes)
print(df.head(5).to_string())

# Parse timestamps as UTC; count values that fail to parse
for c in ["started_at", "finished_at"]:
    raw_na = df[c].isna().sum()
    df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    print(f"{c}: raw NaN={raw_na}, unparseable={df[c].isna().sum() - raw_na}")

section("DATE RANGE")
print("started_at :", df.started_at.min(), "->", df.started_at.max())
print("finished_at:", df.finished_at.min(), "->", df.finished_at.max())

section("MISSING VALUES")
print(df.isna().sum())

section("CATEGORICALS")
for c in ["level", "source"]:
    print(df[c].value_counts(dropna=False), "\n")

section("ROWS PER MONTH (ALL REGIONS) — last 14 months")
m = df.started_at.dt.tz_convert(None).dt.to_period("M")
print(df.groupby([m, "level"]).size().unstack(fill_value=0).tail(14))

section("KYIV CITY: MATCHING OBLAST NAMES")
print(df.oblast[df.oblast.str.contains("Kyiv", na=False)].value_counts())

kyiv = df[df.oblast == "Kyiv City"].copy()
section("KYIV CITY: LEVEL / RAION / HROMADA")
print(kyiv.level.value_counts(dropna=False))
print("raion values:", kyiv.raion.value_counts(dropna=False).head(10).to_dict())
print("hromada values:", kyiv.hromada.value_counts(dropna=False).head(10).to_dict())

section("KYIV CITY: ALERTS PER MONTH x LEVEL")
km = kyiv.started_at.dt.tz_convert(None).dt.to_period("M")
tab = kyiv.groupby([km, "level"]).size().unstack(fill_value=0)
tab["total"] = tab.sum(axis=1)
print(tab.to_string())

section("KYIV CITY: DATE RANGE")
print(kyiv.started_at.min(), "->", kyiv.started_at.max(), "| rows:", len(kyiv))

section("DURATION ANOMALIES (ALL)")
dur = (df.finished_at - df.started_at).dt.total_seconds() / 60
df["dur_min"] = dur
print(dur.describe(percentiles=[.01, .5, .99]))
print("<=0 min:", (dur <= 0).sum(), "| <1 min:", (dur < 1).sum(),
      "| >24h:", (dur > 1440).sum(), "| finished_at NaN:", df.finished_at.isna().sum())

section("DURATION ANOMALIES (KYIV CITY)")
kyiv["dur_min"] = (kyiv.finished_at - kyiv.started_at).dt.total_seconds() / 60
print(kyiv.dur_min.describe(percentiles=[.01, .5, .9, .99]))
print("<=0 min:", (kyiv.dur_min <= 0).sum(), "| >6h:", (kyiv.dur_min > 360).sum(),
      "| >24h:", (kyiv.dur_min > 1440).sum())
cols = ["level", "started_at", "finished_at", "dur_min", "source"]
print("\nShortest Kyiv City:\n", kyiv.nsmallest(5, "dur_min")[cols].to_string())
print("\nLongest Kyiv City:\n", kyiv.nlargest(5, "dur_min")[cols].to_string())

section("DATA COVERAGE NEAR THE END (is Kyiv silence on 2026-09-06 real?)")
# If other oblasts have alerts on a day, that day was collected; Kyiv silence is then real.
dd = df.drop_duplicates()
last = dd[dd.started_at >= "2026-09-01"]
day = last.started_at.dt.tz_convert("Europe/Kyiv").dt.date
cov = pd.DataFrame({
    "all_alerts": last.groupby(day).size(),
    "oblasts_with_alerts": last.groupby(day).oblast.nunique(),
    "kyiv_city_alerts": last[last.oblast == "Kyiv City"].groupby(day).size(),
}).fillna(0).astype(int)
print("(days in Kyiv local time)")
print(cov.to_string())
hrs = last.started_at.dt.floor("h")
print("UTC hours with >=1 alert start anywhere, Sep 5-7:",
      hrs[hrs >= "2026-09-05"].nunique(), "of",
      int((last.started_at.max().floor("h") - pd.Timestamp("2026-09-05", tz="UTC")) / pd.Timedelta("1h")) + 1)

section("KYIV CITY: OVERLAPS / DUPLICATES")
k = kyiv.sort_values("started_at")
print("exact duplicate rows:", k.duplicated().sum())
print("duplicate started_at:", k.started_at.duplicated().sum())
overlap = (k.started_at < k.finished_at.shift()).sum()
print("alerts starting before previous one finished:", overlap)
