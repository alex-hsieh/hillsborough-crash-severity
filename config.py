"""Project-wide settings: paths, years, seeds, label mapping, feature groups.

Every stage imports from here, so a change in one place (a year range, a
feature group, the leakage deny list) applies to the whole pipeline.
See KNOWLEDGE.md for the reasoning behind each decision.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_MODELS = ROOT / "outputs" / "models"

CRASH_RAW_PATTERN = "fdot_crashes_hillsborough_{year}.json"
CRASH_METADATA_FILE = "fdot_crashes_layer_metadata.json"
CRASHES_CLEAN = DATA_INTERIM / "crashes_clean.parquet"

# --- Study scope (KNOWLEDGE.md decisions D1 and D3) --------------------------
YEARS = [2014, 2015, 2016, 2017, 2018]
TRAIN_YEARS = [2014, 2015, 2016, 2017]
TEST_YEAR = 2018

# Time-ordered tuning folds: (training years, validation year).
CV_FOLDS = [
    ([2014], 2015),
    ([2014, 2015], 2016),
    ([2014, 2015, 2016], 2017),
]

# --- Reproducibility ---------------------------------------------------------
SEED = 42
NOISE_CHECK_SEEDS = [42, 7, 2026]

# --- Time zone (decision D6, revised Nov 24 after checking the real data) ----
# CRASH_DATE is stored as midnight in fixed EST (always 05:00 UTC, all year),
# so the calendar date is recovered by shifting it by -5 hours.
# CRASH_TIME is a separate "HHMM" string: the officer's local wall-clock time,
# which DOES follow daylight saving. So date + time is a local wall-clock time,
# converted to UTC with America/New_York.
CRASH_DATE_UTC_OFFSET_HOURS = -5
LOCAL_TZ = "America/New_York"
# Daylight-saving edge cases: the repeated 1 am hour in November is read as the
# first (daylight) occurrence; the skipped 2 am hour in March is shifted forward.
DST_AMBIGUOUS_AS_DAYLIGHT = True

# --- Label (decision D2) -----------------------------------------------------
# INJSEVER code -> class. Codes not listed (0 unknown, 6 non-traffic) are dropped.
LABEL_FIELD = "INJSEVER"
# INJSEVER is a one-character string field on the FDOT layer.
LABEL_MAP = {"1": 0, "2": 1, "3": 1, "4": 2, "5": 2}
CLASS_NAMES = ["No injury", "Injury", "Severe"]

# --- Leakage deny list -------------------------------------------------------
# These are only known after the crash is investigated, or they encode the
# outcome directly. Training asserts none of them is used as a feature.
DENY_EXACT = {
    "INJSEVER",
    "NUMBER_OF_INJURED",
    "NUMBER_OF_KILLED",
    "NUMBER_OF_SERIOUS_INJURIES",
    "TOTAL_PERSONS",
}
# Any column containing one of these substrings is also denied. This catches
# outcome fields plus FDOT's pre-built behaviour flags (seat belt, impairment,
# speeding...), which are coded after the investigation.
DENY_SUBSTRINGS = (
    "INJ", "FATAL", "KILL", "DEATH",
    "NO_BELT", "IMPAIRED", "SPEEDING", "AGGRESSIVE", "DISTRACTED",
    "ALCINV", "WRONGWAY", "LANE_DEPARTURE", "CRASH_CLASS",
)

# --- Feature groups (also the ablation steps, in order) ----------------------
# Names refer to columns produced by s3_features.py.
FEATURE_GROUPS = {
    "roadway": [
        "SPEED_LIMIT", "FUNCLASS", "CNTOFLANES", "GEO_URBAN_RURAL_IND",
        "INTERSECTION_IND", "IMPCT_TYP_CD", "lat", "lon",
        # Extra roadway fields found on the real layer (Nov 24):
        "RDWY_ALIGN_CD", "RDWY_GRDE_CD", "TRAF_WAY_CD", "JCT_CD",
        "V1TRAFCTL", "RCI_MEDIAN_WIDTH_FT", "RCI_SURFACE_WIDTH_FT",
        # Road system (interstate/US/state/county/local). Always filled, and it
        # explains why speed limit, lanes and ADT are missing on ~half the rows:
        # FDOT inventories only state-maintained roads.
        "DHSRDSYS",
    ],
    "time": ["hour", "day_of_week", "month", "is_weekend", "LGHT_COND_CD"],
    "weather": [
        "EVNT_WTHR_COND_CD", "RD_SRFC_COND_CD",
        "wx_temp_c", "wx_precip_mm", "wx_visibility_km", "wx_wind_ms",
        "wx_rel_humidity", "wx_missing",
    ],
    "traffic": ["AVERAGE_DAILY_TRAFFIC", "adt_missing"],
}
ABLATION_ORDER = ["roadway", "time", "weather", "traffic"]

# Columns treated as categorical (gradient boosting reads them natively).
CATEGORICAL_FEATURES = [
    "FUNCLASS", "GEO_URBAN_RURAL_IND", "INTERSECTION_IND", "IMPCT_TYP_CD",
    "LGHT_COND_CD", "EVNT_WTHR_COND_CD", "RD_SRFC_COND_CD",
    "RDWY_ALIGN_CD", "RDWY_GRDE_CD", "TRAF_WAY_CD", "JCT_CD", "V1TRAFCTL",
    "DHSRDSYS",
]

# Crash time is a separate 4-character "HHMM" string next to CRASH_DATE.
CRASH_DATE_FIELD = "CRASH_DATE"
CRASH_TIME_FIELD = "CRASH_TIME"

# --- Hillsborough County bounding box (rough, for dropping bad coordinates) --
COUNTY_BBOX = {"lat_min": 27.55, "lat_max": 28.20, "lon_min": -82.85, "lon_max": -82.05}


def is_denied(column: str) -> bool:
    """Return True if a column must never be used as a model feature."""
    upper = column.upper()
    return upper in DENY_EXACT or any(s in upper for s in DENY_SUBSTRINGS)
