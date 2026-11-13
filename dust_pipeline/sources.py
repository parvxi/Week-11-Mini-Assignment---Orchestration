"""Download the two raw sources. Used by the Airflow DAG and runnable on its own.

Source 1 — NOAA Integrated Surface Database (ISD), hourly observations at Saudi stations,
           served by the KAPSARC Data Portal (Opendatasoft API). Public domain.
Source 2 — Copernicus Atmosphere Monitoring Service (CAMS) global atmospheric composition,
           hourly dust / PM10 / PM2.5 / aerosol optical depth at each station's coordinates,
           served by the Open-Meteo Air Quality API (CC BY 4.0). Available from August 2022.

    python -m dust_pipeline.sources            # download both into datasets/raw/
"""

from __future__ import annotations

import gzip
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).resolve().parent.parent
RAW = REPO / "datasets" / "raw"
ISD_FILE = RAW / "noaa_isd_saudi_hourly.csv.gz"
CAMS_FILE = RAW / "cams_dust_hourly.csv.gz"
MANIFEST = RAW / "manifest.json"

START_DATE = "2022-08-01"  # first day CAMS global data is available from Open-Meteo
MIN_STATION_OBS = 15_000  # stations with fewer hourly rows in the window are too sparse to use

ISD_DATASET = "saudi-hourly-weather-data"
ISD_EXPORT = f"https://datasource.kapsarc.org/api/explore/v2.1/catalog/datasets/{ISD_DATASET}/exports/csv"
ISD_COLUMNS = [
    "station_id", "station_name", "observation_date", "elevation", "wind_direction_angle",
    "wind_speed_rate", "wind_type", "visibility_distance", "air_temperature",
    "air_temperature_dew_point", "atmospheric_sea_level_pressure", "geopoint",
]
CAMS_API = "https://air-quality-api.open-meteo.com/v1/air-quality"
CAMS_VARIABLES = ["pm10", "pm2_5", "dust", "aerosol_optical_depth"]


def _update_manifest(key: str, entry: dict) -> None:
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    manifest[key] = {**entry, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def download_isd(dest: Path = ISD_FILE, start: str = START_DATE) -> dict:
    """Stream the hourly ISD export (selected columns, from `start`) to a gzipped CSV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    params = {
        "select": ",".join(ISD_COLUMNS),
        "where": f"observation_date >= date'{start}'",
        "delimiter": ",",
        "limit": -1,
    }
    tmp = dest.with_suffix(".part")
    rows = -1  # header line
    with requests.get(ISD_EXPORT, params=params, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with gzip.open(tmp, "wb") as out:
            for line in resp.iter_lines():
                out.write(line + b"\n")
                rows += 1
    if rows < 100_000:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ISD export returned only {rows} rows — refusing to replace the cached file")
    tmp.replace(dest)
    info = {"file": dest.name, "rows": rows, "source": ISD_EXPORT, "where": params["where"]}
    _update_manifest("noaa_isd", info)
    return info


def read_isd(path: Path = ISD_FILE) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["observation_date"] = pd.to_datetime(df["observation_date"], utc=True)
    return df


def station_table(isd: pd.DataFrame) -> pd.DataFrame:
    """One row per usable station with coordinates parsed from the ISD geopoint column."""
    geo = isd["geopoint"].str.split(",", expand=True).astype(float)
    tmp = isd.assign(latitude=geo[0], longitude=geo[1])
    stations = (
        tmp.groupby("station_id")
        .agg(
            station_name=("station_name", "first"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median"),
            elevation_m=("elevation", "median"),
            n_hourly_obs=("observation_date", "count"),
            first_obs=("observation_date", "min"),
            last_obs=("observation_date", "max"),
        )
        .reset_index()
    )
    return stations[stations["n_hourly_obs"] >= MIN_STATION_OBS].sort_values("station_id").reset_index(drop=True)


def download_cams(stations: pd.DataFrame, end: str, dest: Path = CAMS_FILE, start: str = START_DATE) -> dict:
    """Fetch hourly CAMS variables (UTC) for every station; one long gzipped CSV."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for row in stations.itertuples():
        params = {
            "latitude": round(row.latitude, 4), "longitude": round(row.longitude, 4),
            "hourly": ",".join(CAMS_VARIABLES), "start_date": start, "end_date": end, "timezone": "GMT",
        }
        for attempt in range(5):
            resp = requests.get(CAMS_API, params=params, timeout=120)
            if resp.status_code == 200:
                break
            time.sleep(15 * (attempt + 1))  # rate limit or transient error → back off
        else:
            raise RuntimeError(f"CAMS request failed for station {row.station_id}: HTTP {resp.status_code} {resp.text[:200]}")
        body = resp.json()
        hourly = pd.DataFrame(body["hourly"])
        hourly.insert(0, "station_id", row.station_id)
        hourly["grid_latitude"], hourly["grid_longitude"] = body["latitude"], body["longitude"]
        frames.append(hourly)
        time.sleep(1.0)
    cams = pd.concat(frames, ignore_index=True)
    buf = io.BytesIO()
    cams.to_csv(buf, index=False)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(gzip.compress(buf.getvalue()))
    tmp.replace(dest)
    info = {"file": dest.name, "rows": int(len(cams)), "stations": int(len(stations)), "source": CAMS_API,
            "variables": CAMS_VARIABLES, "start": start, "end": end}
    _update_manifest("cams", info)
    return info


def main() -> None:
    print("ISD :", download_isd())
    isd = read_isd()
    stations = station_table(isd)
    end = isd["observation_date"].max().strftime("%Y-%m-%d")
    print(f"{len(stations)} usable stations, window ends {end}")
    print("CAMS:", download_cams(stations, end))


if __name__ == "__main__":
    main()
