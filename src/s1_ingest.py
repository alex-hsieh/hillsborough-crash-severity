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
    """Turn ArcGIS date fields (milliseconds since 1970) into pandas datetimes.

    The values are kept exactly as the server returned them (no time zone
    shift). The EST-versus-UTC decision (KNOWLEDGE.md D6) is applied later,
    once the timestamp convention has been checked against known crashes.
    """
    date_fields = [f["name"] for f in meta["fields"] if f["type"] == "esriFieldTypeDate"]
    for col in date_fields:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
    return df


def drop_bad_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicates and crashes with missing or out-of-county coordinates."""
    n0 = len(df)
    df = df.drop_duplicates()
    n_dup = n0 - len(df)

    box = config.COUNTY_BBOX
    lat, lon = df["lat_wgs84"], df["lon_wgs84"]
    inside = lat.between(box["lat_min"], box["lat_max"]) & lon.between(box["lon_min"], box["lon_max"])
    n_coord = int((~inside).sum())
    df = df[inside].copy()
    df = df.rename(columns={"lat_wgs84": "lat", "lon_wgs84": "lon"})

    log(f"  dropped {n_dup:,} duplicate rows and {n_coord:,} rows with missing/out-of-county coordinates")
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

    missing_table(df).to_csv(config.OUT_TABLES / "table1_crash_missing.csv", index=False)
    df.to_parquet(config.CRASHES_CLEAN, index=False)
    log(f"  wrote {len(df):,} rows to {config.CRASHES_CLEAN.relative_to(config.ROOT)}")
    return df


if __name__ == "__main__":
    run()
