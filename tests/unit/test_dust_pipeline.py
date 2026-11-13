"""Part 1 transformation logic: cleaning, daily aggregation, merge and target construction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dust_pipeline import transform
from dust_pipeline.schema import MIN_LOW_VIS_HOURS, TARGET


def _isd(hours: int = 24 * 4) -> pd.DataFrame:
    ts = pd.date_range("2024-03-01 21:00", periods=hours, freq="h", tz="UTC")  # 21:00 UTC = local midnight
    return pd.DataFrame({
        "station_id": 1, "station_name": "TEST", "observation_date": ts, "elevation": 600.0,
        "wind_direction_angle": 90, "wind_speed_rate": 4.0, "wind_type": "Normal", "visibility_distance": 10000,
        "air_temperature": 30.0, "air_temperature_dew_point": 5.0, "atmospheric_sea_level_pressure": 1010.0,
        "geopoint": "24.9, 46.7",
    })


def test_relative_humidity_formula() -> None:
    rh = transform.relative_humidity(pd.Series([20.0, 30.0]), pd.Series([20.0, 5.0]))
    assert abs(rh[0] - 100) < 1e-6 and 19 < rh[1] < 22


def test_clean_isd_removes_sentinels_duplicates_and_sparse_stations() -> None:
    raw = _isd()
    raw.loc[0, "visibility_distance"] = 999999
    raw.loc[1, "air_temperature"] = 999.9
    raw.loc[2, "atmospheric_sea_level_pressure"] = 9999.9
    raw = pd.concat([raw, raw.iloc[[5]], raw.assign(station_id=2).head(3)], ignore_index=True)
    hourly, stats = transform.clean_isd(raw, station_ids=[1])
    assert stats["duplicate_reports_dropped"] == 1 and stats["rows_sparse_stations_dropped"] == 3
    assert stats["visibility_distance_sentinels"] == 1 and hourly["visibility_m"].isna().sum() == 1
    assert hourly["air_temperature"].isna().sum() == 1 and hourly["visibility_m"].max() <= 10000


def test_low_visibility_counts_distinct_dry_hours_not_reports() -> None:
    raw = _isd(24)
    raw.loc[3:5, "visibility_distance"] = 2000  # three dusty hours in dry air
    extra = raw.iloc[[3, 3, 3]].assign(observation_date=lambda d: d["observation_date"] + pd.to_timedelta([10, 20, 30], "min"))
    foggy = raw.iloc[[10]].index
    raw.loc[foggy, ["visibility_distance", "air_temperature_dew_point"]] = [500, 29.5]  # humid → fog, not dust
    hourly, _ = transform.clean_isd(pd.concat([raw, extra], ignore_index=True), [1])
    daily = transform.daily_isd(hourly)
    assert len(daily) == 1 and daily.loc[0, "low_vis_hours"] == 3 and daily.loc[0, "n_hourly_obs"] == 24
    assert daily.loc[0, "low_vis_today"] == int(3 >= MIN_LOW_VIS_HOURS)


def test_merge_builds_next_day_target_only_for_consecutive_days() -> None:
    days = pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03", "2024-03-05"])  # 4 March is missing
    isd = pd.DataFrame({
        "station_id": 1, "obs_date": days, "n_hourly_obs": 24, "vis_min_m": [9000, 3000, 9000, 9000],
        "vis_mean_m": 9000.0, "low_vis_hours": [0, 4, 0, 0], "wind_mean_ms": 3.0, "wind_max_ms": 6.0, "wind_u_mean": 0.0,
        "wind_v_mean": 0.0, "temp_mean_c": 30.0, "temp_max_c": 36.0, "temp_min_c": 24.0, "rh_mean_pct": 20.0,
        "rh_min_pct": 10.0, "pressure_mean_hpa": 1010.0, "temp_range_c": 12.0, "low_vis_today": [0, 1, 0, 0],
    })
    cams = pd.DataFrame({"station_id": 1, "obs_date": days, "cams_hours": 24, "dust_mean_ugm3": 100.0, "dust_max_ugm3": 200.0,
                         "pm10_mean_ugm3": 80.0, "pm10_max_ugm3": 150.0, "pm25_mean_ugm3": 20.0, "aod_mean": 0.3, "aod_max": 0.4})
    stations = pd.DataFrame({"station_id": [1], "station_name": ["TEST"], "latitude": [24.9], "longitude": [46.7], "elevation_m": [600.0]})
    merged, stats = transform.merge_station_days(isd, cams, stations)
    # 3 March has no next day (the 4th is missing) and 5 March is the last day → both dropped
    assert [str(d) for d in merged["obs_date"]] == ["2024-03-01", "2024-03-02"]
    assert merged[TARGET].tolist() == [1, 0] and merged["next_day_low_vis_hours"].tolist() == [4, 0]
    assert np.isnan(merged.loc[0, "vis_min_m_yesterday"]) and merged.loc[1, "vis_min_m_yesterday"] == 9000
    assert stats["matched_station_days"] == 4 and stats["rows_without_next_day"] == 2
