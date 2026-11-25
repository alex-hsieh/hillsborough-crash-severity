"""Stage 1: load the raw FDOT crash files into one table.

Input:  data/raw/fdot_crashes_hillsborough_<year>.json (one per year), made by
        tools/fdot_browser_download.js, plus fdot_crashes_layer_metadata.json.
Output: data/interim/crashes_clean.parquet
        outputs/tables/table1_crash_missing.csv (percent missing per field)

gis.fdot.gov blocks scripted downloads, so this stage never downloads crash
data itself. It only reads files that already exist in data/raw/.
"""

import json

import pandas as pd

import config
from src.utils import ensure_dirs, log


def load_metadata() -> dict:
    """Read the layer metadata saved by the browser download script."""
    path = config.DATA_RAW / config.CRASH_METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run tools/fdot_browser_download.js first "
            "(instructions at the top of that file)."
        )
    return json.loads(path.read_text())


def load_year(year: int) -> pd.DataFrame:
    """Read one year's crash file into a DataFrame."""
    path = config.DATA_RAW / config.CRASH_RAW_PATTERN.format(year=year)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run tools/fdot_browser_download.js.")
    payload = json.loads(path.read_text())
    df = pd.DataFrame(payload["rows"], columns=payload["fields"])
    log(f"  {year}: {len(df):,} rows from {path.name}")
    return df


def convert_dates(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """Turn ArcGIS date fields (milliseconds since 1970, UTC) into pandas datetimes."""
    date_fields = [f["name"] for f in meta["fields"] if f["type"] == "esriFieldTypeDate"]
    for col in date_fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def build_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Add crash_local (wall-clock) and crash_utc columns. See config D6 notes.

    CRASH_DATE is midnight fixed-EST expressed in UTC, so shifting it by
    -5 hours gives the calendar date. CRASH_TIME ("HHMM") is local clock time.
    """
    day = (df[config.CRASH_DATE_FIELD] + pd.Timedelta(hours=config.CRASH_DATE_UTC_OFFSET_HOURS)).dt.normalize()
    hhmm = df[config.CRASH_TIME_FIELD].fillna("").str.zfill(4)
    hours = pd.to_numeric(hhmm.str[:2], errors="coerce")
    minutes = pd.to_numeric(hhmm.str[2:], errors="coerce")
    valid = hours.between(0, 23) & minutes.between(0, 59)
    df["time_missing"] = (~valid).astype("int8")

    local = day + pd.to_timedelta(hours.where(valid), unit="h") + pd.to_timedelta(minutes.where(valid), unit="m")
    df["crash_local"] = local
    df["crash_utc"] = (
        local.dt.tz_localize(config.LOCAL_TZ, ambiguous=config.DST_AMBIGUOUS_AS_DAYLIGHT,
                             nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )
    log(f"  built timestamps; {int((~valid).sum()):,} crashes have no usable crash time")
    return df


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove repeated crash IDs and crashes with missing or out-of-county coordinates.

    Some crashes appear twice with different OBJECTIDs but the same crash ID
    (XID); only the first copy is kept.
    """
    n0 = len(df)
    df = df.drop_duplicates(subset="XID", keep="first")
    n_dup = n0 - len(df)

    box = config.COUNTY_BBOX
    lat, lon = df["lat_wgs84"], df["lon_wgs84"]
    inside = lat.between(box["lat_min"], box["lat_max"]) & lon.between(box["lon_min"], box["lon_max"])
    n_coord = int((~inside).sum())
    df = df[inside].copy()
    df = df.rename(columns={"lat_wgs84": "lat", "lon_wgs84": "lon"})

    log(f"  dropped {n_dup:,} repeated crash IDs and {n_coord:,} rows with missing/out-of-county coordinates")
    return df


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    """Percent missing per field (empty strings count as missing). Feeds report Table 1."""
    as_missing = df.replace({"": None, " ": None})
    pct = (as_missing.isna().mean() * 100).round(1)
    return pct.rename("pct_missing").rename_axis("field").reset_index().sort_values(
        "pct_missing", ascending=False
    )


def run() -> pd.DataFrame:
    """Run stage 1 and return the cleaned crash table."""
    ensure_dirs(config.DATA_INTERIM, config.OUT_TABLES)
    meta = load_metadata()
    log(f"  layer time reference: {meta.get('dateFieldsTimeReference') or 'not given'}")

    df = pd.concat([load_year(y) for y in config.YEARS], ignore_index=True)
    log(f"  {len(df):,} crashes loaded in total")
    df = convert_dates(df, meta)
    df = drop_bad_rows(df)
    df = build_timestamps(df)

    missing_table(df).to_csv(config.OUT_TABLES / "table1_crash_missing.csv", index=False)
    df.to_parquet(config.CRASHES_CLEAN, index=False)
    log(f"  wrote {len(df):,} rows to {config.CRASHES_CLEAN.relative_to(config.ROOT)}")
    return df


if __name__ == "__main__":
    run()
