"""Stage 2: join hourly weather from Tampa International Airport to each crash.

Input:  data/interim/crashes_clean.parquet (stage 1)
        data/raw/ghcnh_USW00012842_<year>.parquet (downloaded here if missing)
Output: data/interim/weather_hourly.parquet   one row per UTC hour
        data/interim/crashes_joined.parquet   one row per crash, same row count
        outputs/tables/weather_match.csv      match rate per weather field
        outputs/tables/timezone_check.csv     officer-reported rain vs. NOAA rain
                                              at different hour offsets

Weather source: NOAA GHCNh (https://www.ncei.noaa.gov/products/global-historical-climatology-network-hourly),
public AWS bucket noaa-ghcnh-pds. Times in GHCNh are UTC.
"""

import numpy as np
import pandas as pd
import requests

import config
from src.utils import ensure_dirs, log

# GHCNh column -> our column name. Units: deg C, mm (last hour), km, m/s, %.
WEATHER_COLUMNS = {
    "temperature": "wx_temp_c",
    "precipitation": "wx_precip_mm",
    "visibility": "wx_visibility_km",
    "wind_speed": "wx_wind_ms",
    "relative_humidity": "wx_rel_humidity",
}
OFFICER_RAIN_CODE = "03"   # EVNT_WTHR_COND_CD: 03 = rain


def download_weather() -> None:
    """Download each year's station file from NOAA if it is not cached in data/raw/."""
    ensure_dirs(config.DATA_RAW)
    for year in config.YEARS:
        path = config.DATA_RAW / config.WEATHER_RAW_PATTERN.format(station=config.WEATHER_STATION, year=year)
        if path.exists():
            continue
        url = config.WEATHER_URL.format(station=config.WEATHER_STATION, year=year)
        log(f"  downloading weather {year} from NOAA")
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        path.write_bytes(resp.content)


def load_hourly_weather() -> pd.DataFrame:
    """Reduce the station's observations to one row per UTC hour.

    The airport sends a routine report at about :53 each hour (report type
    FM15, rainfall = the previous hour) plus extra special reports (FM16).
    Each observation is assigned to its nearest whole hour; within an hour the
    routine report wins, then the one closest to the top of the hour.
    """
    frames = []
    for year in config.YEARS:
        path = config.DATA_RAW / config.WEATHER_RAW_PATTERN.format(station=config.WEATHER_STATION, year=year)
        frames.append(pd.read_parquet(path, columns=["DATE", "precipitation_Report_Type",
                                                     "temperature_Report_Type", *WEATHER_COLUMNS]))
    w = pd.concat(frames, ignore_index=True)
    w["obs_utc"] = pd.to_datetime(w["DATE"], utc=True)
    for col in WEATHER_COLUMNS:
        w[col] = pd.to_numeric(w[col], errors="coerce")

    w["hour_utc"] = w["obs_utc"].dt.round("h")
    report_type = w["temperature_Report_Type"].fillna(w["precipitation_Report_Type"])
    w["is_routine"] = (report_type == "FM15").astype(int)
    w["minutes_off"] = (w["obs_utc"] - w["hour_utc"]).abs().dt.total_seconds() / 60
    w = (w.sort_values(["hour_utc", "is_routine", "minutes_off"], ascending=[True, False, True])
          .drop_duplicates("hour_utc"))

    hourly = w.set_index("hour_utc")[list(WEATHER_COLUMNS)].rename(columns=WEATHER_COLUMNS)
    # Full hourly index so the 3-hour rain window is correct even across gaps.
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", tz="UTC")
    hourly = hourly.reindex(full)
    hourly.index.name = "hour_utc"
    # Rain over the last 3 hours: a wet-road proxy (plan, Phase 3).
    hourly["wx_precip_3h_mm"] = hourly["wx_precip_mm"].rolling(3, min_periods=1).sum()
    log(f"  weather: {len(hourly):,} hours, {hourly['wx_temp_c'].notna().mean():.1%} with a temperature")
    return hourly.reset_index()


def join_weather(crashes: pd.DataFrame, hourly: pd.DataFrame, offset_hours: int = 0) -> pd.DataFrame:
    """Attach the weather of each crash's nearest UTC hour (optionally shifted)."""
    key = crashes["crash_utc"].dt.round("h") + pd.Timedelta(hours=offset_hours)
    out = crashes.assign(hour_utc=key).merge(hourly, on="hour_utc", how="left")
    assert len(out) == len(crashes), "weather join changed the row count"
    return out


def timezone_check(crashes: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """Test the time zone choice (KNOWLEDGE.md D6) against the weather itself.

    For crashes where the officer reported rain, the share with NOAA rain in the
    matched hour should peak at offset 0 if timestamps are right. Run separately
    for daylight-saving months and standard-time months, since a DST mistake
    would only shift the summer crashes.
    """
    rows = []
    rain = crashes[crashes["EVNT_WTHR_COND_CD"] == OFFICER_RAIN_CODE]
    is_dst = rain["crash_utc"].dt.tz_convert(config.LOCAL_TZ).map(lambda t: bool(t.dst()))
    for label, subset in [("daylight time", rain[is_dst]), ("standard time", rain[~is_dst])]:
        for offset in range(-3, 4):
            j = join_weather(subset, hourly, offset)
            rows.append({"period": label, "offset_hours": offset, "crashes": len(j),
                         "pct_with_noaa_rain": round(100 * (j["wx_precip_mm"] > 0).mean(), 1)})
    table = pd.DataFrame(rows)
    for label, part in table.groupby("period"):
        best = part.loc[part["pct_with_noaa_rain"].idxmax(), "offset_hours"]
        log(f"  time zone check ({label}): NOAA rain best matches officer rain at offset {best:+d} h")
    return table


def irma_check(joined: pd.DataFrame) -> None:
    """Spot check: crashes during Hurricane Irma (Sept 10-11, 2017) should see rain and wind."""
    irma = joined[(joined["crash_local"] >= "2017-09-10") & (joined["crash_local"] < "2017-09-12")]
    if len(irma):
        log(f"  Irma check: {len(irma)} crashes, mean wind {irma['wx_wind_ms'].mean():.1f} m/s, "
            f"max hourly rain {irma['wx_precip_mm'].max():.1f} mm")


def run() -> pd.DataFrame:
    """Run stage 2 and return the crash table with weather attached."""
    ensure_dirs(config.DATA_INTERIM, config.OUT_TABLES)
    download_weather()
    hourly = load_hourly_weather()
    hourly.to_parquet(config.WEATHER_HOURLY, index=False)

    crashes = pd.read_parquet(config.CRASHES_CLEAN)
    joined = join_weather(crashes, hourly)
    wx_cols = [*WEATHER_COLUMNS.values(), "wx_precip_3h_mm"]
    joined["wx_missing"] = joined[list(WEATHER_COLUMNS.values())].isna().any(axis=1).astype("int8")

    match = pd.DataFrame({"field": wx_cols,
                          "pct_matched": [round(100 * joined[c].notna().mean(), 1) for c in wx_cols]})
    match.to_csv(config.OUT_TABLES / "weather_match.csv", index=False)
    log("  weather match rates: " + ", ".join(f"{r.field} {r.pct_matched}%" for r in match.itertuples()))

    timezone_check(crashes, hourly).to_csv(config.OUT_TABLES / "timezone_check.csv", index=False)
    irma_check(joined)

    joined.to_parquet(config.CRASHES_JOINED, index=False)
    log(f"  wrote {len(joined):,} rows to {config.CRASHES_JOINED.relative_to(config.ROOT)}")
    return joined


if __name__ == "__main__":
    run()
