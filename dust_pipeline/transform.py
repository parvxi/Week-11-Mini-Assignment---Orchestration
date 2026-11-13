"""Pure transformation functions (no Airflow, no database) so they can be unit-tested.

hourly ISD  ──clean──▶ hourly QC'd ──daily──▶ one row per station-day (weather + visibility)
hourly CAMS ──clean──▶ hourly QC'd ──daily──▶ one row per station-day (dust / PM / AOD)
                                    merge on (station_id, obs_date) ──▶ features + next-day target
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dust_pipeline.schema import (
    DRY_AIR_MAX_RH,
    LOCAL_UTC_OFFSET_HOURS,
    LOW_VIS_METERS,
    MIN_HOURLY_OBS_PER_DAY,
    MIN_LOW_VIS_HOURS,
    TARGET,
)

# ISD encodes "missing" as sentinel numbers. Anything at or above these is not a measurement.
ISD_SENTINELS = {
    "visibility_distance": 999_999,
    "wind_speed_rate": 999.9,
    "wind_direction_angle": 999,
    "air_temperature": 999.9,
    "air_temperature_dew_point": 999.9,
    "atmospheric_sea_level_pressure": 9999.9,
}
# Physically plausible ranges for Saudi Arabia; values outside are treated as bad readings.
ISD_RANGES = {
    "visibility_distance": (0, 200_000),
    "wind_speed_rate": (0, 60),
    "air_temperature": (-15, 60),
    "air_temperature_dew_point": (-45, 40),
    "atmospheric_sea_level_pressure": (950, 1060),
}
MAX_MISSING_SHARE = 0.5
VISIBILITY_CAP_M = 10_000  # stations report "10 km or more" inconsistently (9999, 10000, 16000…)
CAMS_VARIABLES = ["pm10", "pm2_5", "dust", "aerosol_optical_depth"]


def relative_humidity(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    """Relative humidity (%) from temperature and dew point (Magnus formula)."""
    a, b = 17.625, 243.04
    rh = 100 * np.exp(a * dewpoint_c / (b + dewpoint_c)) / np.exp(a * temp_c / (b + temp_c))
    return rh.clip(0, 100)


def _local_date(ts_utc: pd.Series) -> pd.Series:
    return (ts_utc + pd.Timedelta(hours=LOCAL_UTC_OFFSET_HOURS)).dt.tz_localize(None).dt.normalize()


# ---------------------------------------------------------------------------
# Source 1 — NOAA ISD
# ---------------------------------------------------------------------------
def clean_isd(raw: pd.DataFrame, station_ids: list[int]) -> tuple[pd.DataFrame, dict]:
    """Sentinels → NaN, range checks, duplicate reports removed, local calendar date added."""
    df = raw[raw["station_id"].isin(station_ids)].copy()
    stats = {"rows_in": int(len(raw)), "rows_sparse_stations_dropped": int(len(raw) - len(df))}
    df["observation_date"] = pd.to_datetime(df["observation_date"], utc=True)

    before = len(df)
    df = df.drop_duplicates(subset=["station_id", "observation_date"])
    stats["duplicate_reports_dropped"] = int(before - len(df))

    for col, sentinel in ISD_SENTINELS.items():
        missing = df[col] >= sentinel
        stats[f"{col}_sentinels"] = int(missing.sum())
        df[col] = df[col].astype(float).mask(missing)
    for col, (lo, hi) in ISD_RANGES.items():
        bad = df[col].notna() & ~df[col].between(lo, hi)
        stats[f"{col}_out_of_range"] = int(bad.sum())
        df[col] = df[col].mask(bad)

    df["visibility_m"] = df["visibility_distance"].clip(upper=VISIBILITY_CAP_M)
    df["rh_pct"] = relative_humidity(df["air_temperature"], df["air_temperature_dew_point"])
    # A dust-type low-visibility report: poor visibility in dry air (humid cases are fog / rain).
    df["low_vis_dry"] = ((df["visibility_m"] < LOW_VIS_METERS) & (df["rh_pct"] < DRY_AIR_MAX_RH)).astype(int)
    radians = np.deg2rad(df["wind_direction_angle"])
    df["wind_u"] = -df["wind_speed_rate"] * np.sin(radians)
    df["wind_v"] = -df["wind_speed_rate"] * np.cos(radians)
    df["obs_date"] = _local_date(df["observation_date"])
    stats["rows_out"] = int(len(df))
    return df, stats


def daily_isd(hourly: pd.DataFrame) -> pd.DataFrame:
    """One row per station and local day. Days with too few reports are dropped."""
    # Stations also file extra "special" reports when conditions change, so a bad hour can
    # contain several reports. Count distinct clock hours, not reports, so the unit is hours.
    hourly = hourly.assign(clock_hour=pd.to_datetime(hourly["observation_date"]).dt.floor("h"))
    low_hours = (
        hourly[hourly["low_vis_dry"] == 1]
        .groupby(["station_id", "obs_date"])["clock_hour"]
        .nunique()
        .rename("low_vis_hours")
    )
    g = hourly.groupby(["station_id", "obs_date"])
    daily = g.agg(
        n_hourly_obs=("clock_hour", "nunique"),
        vis_min_m=("visibility_m", "min"),
        vis_mean_m=("visibility_m", "mean"),
        wind_mean_ms=("wind_speed_rate", "mean"),
        wind_max_ms=("wind_speed_rate", "max"),
        wind_u_mean=("wind_u", "mean"),
        wind_v_mean=("wind_v", "mean"),
        temp_mean_c=("air_temperature", "mean"),
        temp_max_c=("air_temperature", "max"),
        temp_min_c=("air_temperature", "min"),
        rh_mean_pct=("rh_pct", "mean"),
        rh_min_pct=("rh_pct", "min"),
        pressure_mean_hpa=("atmospheric_sea_level_pressure", "mean"),
    ).reset_index()
    daily = daily.merge(low_hours, on=["station_id", "obs_date"], how="left")
    daily["low_vis_hours"] = daily["low_vis_hours"].fillna(0).astype(int)
    daily = daily[(daily["n_hourly_obs"] >= MIN_HOURLY_OBS_PER_DAY) & daily["vis_min_m"].notna()]
    daily["temp_range_c"] = daily["temp_max_c"] - daily["temp_min_c"]
    daily["low_vis_today"] = (daily["low_vis_hours"] >= MIN_LOW_VIS_HOURS).astype(int)
    return daily.round(3).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Source 2 — CAMS
# ---------------------------------------------------------------------------
def clean_cams(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Parse timestamps, drop duplicates, null out negative concentrations, add local date."""
    df = raw.copy()
    stats = {"rows_in": int(len(df))}
    df["time"] = pd.to_datetime(df["time"], utc=True)
    before = len(df)
    df = df.drop_duplicates(subset=["station_id", "time"])
    stats["duplicate_hours_dropped"] = int(before - len(df))
    for col in CAMS_VARIABLES:
        bad = df[col] < 0
        stats[f"{col}_negative"] = int(bad.sum())
        stats[f"{col}_null"] = int(df[col].isna().sum())
        df[col] = df[col].mask(bad)
    df["obs_date"] = _local_date(df["time"])
    stats["rows_out"] = int(len(df))
    return df, stats


def daily_cams(hourly: pd.DataFrame) -> pd.DataFrame:
    g = hourly.groupby(["station_id", "obs_date"])
    daily = g.agg(
        cams_hours=("time", "count"),
        dust_mean_ugm3=("dust", "mean"),
        dust_max_ugm3=("dust", "max"),
        pm10_mean_ugm3=("pm10", "mean"),
        pm10_max_ugm3=("pm10", "max"),
        pm25_mean_ugm3=("pm2_5", "mean"),
        aod_mean=("aerosol_optical_depth", "mean"),
        aod_max=("aerosol_optical_depth", "max"),
    ).reset_index()
    daily = daily[daily["cams_hours"] >= 18]  # need most of the day
    return daily.round(3).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Merge + features + target
# ---------------------------------------------------------------------------
def merge_station_days(isd_daily: pd.DataFrame, cams_daily: pd.DataFrame, stations: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Join the two sources per station-day, add history features and the next-day target."""
    isd_daily = isd_daily.assign(obs_date=pd.to_datetime(isd_daily["obs_date"]))
    cams_daily = cams_daily.assign(obs_date=pd.to_datetime(cams_daily["obs_date"]))
    merged = isd_daily.merge(cams_daily, on=["station_id", "obs_date"], how="inner", validate="one_to_one")
    stats = {
        "isd_station_days": int(len(isd_daily)),
        "cams_station_days": int(len(cams_daily)),
        "matched_station_days": int(len(merged)),
        "isd_days_without_cams": int(len(isd_daily) - len(merged)),
    }
    static = stations[["station_id", "station_name", "latitude", "longitude", "elevation_m"]]
    merged = merged.merge(static, on="station_id", how="left", validate="many_to_one")
    merged = merged.sort_values(["station_id", "obs_date"]).reset_index(drop=True)

    by_station = merged.groupby("station_id", sort=False)
    prev_date = by_station["obs_date"].shift(1)
    next_date = by_station["obs_date"].shift(-1)
    has_yesterday = (merged["obs_date"] - prev_date) == pd.Timedelta(days=1)
    has_tomorrow = (next_date - merged["obs_date"]) == pd.Timedelta(days=1)

    # Yesterday's values (only when yesterday is actually the previous calendar day).
    for col in ("vis_min_m", "low_vis_hours", "dust_max_ugm3", "wind_max_ms", "pressure_mean_hpa"):
        merged[f"{col}_yesterday"] = by_station[col].shift(1).where(has_yesterday)
    merged["pressure_change_hpa"] = merged["pressure_mean_hpa"] - merged["pressure_mean_hpa_yesterday"]
    merged = merged.drop(columns=["pressure_mean_hpa_yesterday"])
    # 3-day history ending today.
    for col in ("dust_mean_ugm3", "wind_mean_ms", "low_vis_hours"):
        merged[f"{col}_3d_mean"] = by_station[col].transform(lambda s: s.rolling(3, min_periods=2).mean())
    # Season as a smooth cycle (day 365 is next to day 1).
    doy = merged["obs_date"].dt.dayofyear
    merged["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    merged["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    merged["month"] = merged["obs_date"].dt.month

    # Tomorrow's outcome — kept for analysis, never a feature.
    merged["next_day_min_visibility_m"] = by_station["vis_min_m"].shift(-1).where(has_tomorrow)
    merged["next_day_low_vis_hours"] = by_station["low_vis_hours"].shift(-1).where(has_tomorrow)
    stats["rows_without_next_day"] = int((~has_tomorrow).sum())
    merged = merged[has_tomorrow].copy()
    merged[TARGET] = (merged["next_day_low_vis_hours"] >= MIN_LOW_VIS_HOURS).astype(int)
    merged["next_day_low_vis_hours"] = merged["next_day_low_vis_hours"].astype(int)

    # A feature that is missing for most station-days cannot be learned from (sea-level
    # pressure is not reported by most of these stations). Rule: drop above 50% missing.
    protected = {TARGET, "next_day_min_visibility_m", "next_day_low_vis_hours"}
    sparse = [c for c in merged.columns if c not in protected and merged[c].isna().mean() > MAX_MISSING_SHARE]
    stats["sparse_columns_dropped"] = {c: round(float(merged[c].isna().mean()), 3) for c in sparse}
    merged = merged.drop(columns=sparse)

    merged["obs_date"] = merged["obs_date"].dt.date
    stats.update(rows=int(len(merged)), columns=int(merged.shape[1]), positive_rate=round(float(merged[TARGET].mean()), 4))
    return merged.round(4).reset_index(drop=True), stats
