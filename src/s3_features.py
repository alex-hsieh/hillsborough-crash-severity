"""Stage 3: severity label, model features, and leakage controls.

Input:  data/interim/crashes_joined.parquet (stage 2)
Output: data/processed/model_table.parquet  label, year and allowed features only
        outputs/tables/table2_class_shares.csv  class shares by year (report Table 2)

Category lists are learned from the training years only (2014-2017), so the
2018 test year cannot influence the encoding.
"""

import pandas as pd

import config
from src.utils import ensure_dirs, log


def all_features() -> list[str]:
    """Every feature name across the ablation groups, in group order."""
    return [f for group in config.ABLATION_ORDER for f in config.FEATURE_GROUPS[group]]


def assert_no_leakage(features: list[str]) -> None:
    """Stop the run if any feature is on the leakage deny list."""
    denied = [f for f in features if config.is_denied(f)]
    assert not denied, f"Leakage: denied columns used as features: {denied}"


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Map INJSEVER to 0/1/2 and drop crashes with unknown or non-traffic severity."""
    df["label"] = df[config.LABEL_FIELD].map(config.LABEL_MAP)
    n_drop = int(df["label"].isna().sum())
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype("int8")
    log(f"  dropped {n_drop:,} crashes with unknown (0) or non-traffic (6) severity")
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour, weekday, month and weekend flag from the local crash time."""
    t = df["crash_local"]
    df["hour"] = t.dt.hour.astype("int8")
    df["day_of_week"] = t.dt.day_of_week.astype("int8")   # 0 = Monday
    df["month"] = t.dt.month.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    return df


def clean_types(df: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Numeric fields to numbers, coded fields to pandas categories.

    Blank strings become missing. Codes never seen in the training years
    become missing too, so the test year cannot add new categories.
    """
    df["adt_missing"] = df["AVERAGE_DAILY_TRAFFIC"].isna().astype("int8")
    for col in all_features():
        if col in config.CATEGORICAL_FEATURES:
            s = df[col].astype("string").str.strip().replace("", pd.NA)
            categories = sorted(s[train_mask].dropna().unique())
            df[col] = pd.Categorical(s, categories=categories)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def class_share_table(df: pd.DataFrame) -> pd.DataFrame:
    """Crash counts and class shares (%) per year, for report Table 2."""
    counts = pd.crosstab(df["CALENDAR_YEAR"], df["label"]).rename(columns=dict(enumerate(config.CLASS_NAMES)))
    shares = (counts.div(counts.sum(axis=1), axis=0) * 100).round(1).add_suffix(" %")
    table = pd.concat([counts.assign(Total=counts.sum(axis=1)), shares], axis=1)
    table.index.name = "year"
    return table.reset_index()


def run() -> pd.DataFrame:
    """Run stage 3 and return the model-ready table."""
    ensure_dirs(config.DATA_PROCESSED, config.OUT_TABLES)
    features = all_features()
    assert_no_leakage(features)

    df = pd.read_parquet(config.CRASHES_JOINED)
    df = add_label(df)
    df = add_time_features(df)
    train_mask = df["CALENDAR_YEAR"].isin(config.TRAIN_YEARS)
    df = clean_types(df, train_mask)

    missing = [f for f in features if f not in df.columns]
    assert not missing, f"features not found in the data: {missing}"
    table = df[["XID", "CALENDAR_YEAR", "label", *features]].rename(columns={"CALENDAR_YEAR": "year"})
    table.to_parquet(config.MODEL_TABLE, index=False)
    class_share_table(df).to_csv(config.OUT_TABLES / "table2_class_shares.csv", index=False)
    log(f"  model table: {len(table):,} crashes, {len(features)} features in "
        f"{len(config.ABLATION_ORDER)} groups")
    return table


if __name__ == "__main__":
    run()
