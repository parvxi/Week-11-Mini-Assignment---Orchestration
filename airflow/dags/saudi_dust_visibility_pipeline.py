"""saudi_dust_visibility_pipeline — Week 11, Part 1.

Question: can tomorrow's dust-type low-visibility day at a Saudi airport be forecast from
today's weather observations and today's atmospheric dust load?

Two independent sources are ingested, cleaned and aggregated to one row per station-day,
merged, loaded into PostgreSQL, checked with SQL, and analysed from the database.

  noaa_observations.extract ─┬─ validate → clean → aggregate_daily ───────────────┐
                             └─ build_station_dim ─┬─ cams_dust.extract → validate │
                                                   │        → clean → aggregate_daily ─┤
                                                   │                                   ▼
                                                   │                           merge_station_days
                                                   │                                   │
                                                   └──────── load.station_dim   load.station_daily
                                                                      └──────┬──────────┘
                                                                       quality_checks
                                                   ┌─────────────────────────┼──────────────────────┐
                                        analysis.train_model   analysis.seasonality_plot   analysis.export_for_agents
                                                   └─────────────────────────┼──────────────────────┘
                                                                        write_report

Tasks exchange small dicts over XCom (paths, counts, QC stats); data moves as files staged
under artifacts/airflow/runs/<run>/staging/.
"""

from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow.decorators import dag, task, task_group
from airflow.models.param import Param
from airflow.providers.postgres.hooks.postgres import PostgresHook

from dust_pipeline import sources, transform
from dust_pipeline.schema import OUTCOME_COLUMNS, PERSISTENCE_COLUMN, TABLE, TARGET, TIME_COLUMN

REPO = Path("/repo")
RUNS = REPO / "artifacts" / "airflow" / "runs"
LATEST = REPO / "artifacts" / "airflow" / "latest"
PUBLISH = REPO / "datasets" / "airflow" / "saudi_dust_daily_latest.csv"  # hand-off to the agents
WAREHOUSE_CONN_ID = "warehouse_db"
STATION_TABLE = "stations"
RAW_MAX_AGE_DAYS = 30  # re-download a raw file when the cached copy is older than this

ISD_REQUIRED = ["station_id", "station_name", "observation_date", "visibility_distance", "wind_speed_rate",
                "air_temperature", "air_temperature_dew_point", "geopoint"]
CAMS_REQUIRED = ["station_id", "time", "pm10", "pm2_5", "dust", "aerosol_optical_depth"]
# Not model inputs: keys, dates, and tomorrow's raw outcome.
NON_FEATURES = ["station_id", "station_name", TIME_COLUMN, *OUTCOME_COLUMNS]


def _stage(df: pd.DataFrame, run_id: str, name: str) -> dict:
    out = RUNS / run_id / "staging" / f"{name}.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return {"path": str(out), "n_rows": int(len(df)), "n_cols": int(df.shape[1])}


def _is_fresh(path: Path) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < RAW_MAX_AGE_DAYS * 86400


def _engine():
    return PostgresHook(postgres_conn_id=WAREHOUSE_CONN_ID).get_sqlalchemy_engine()


@dag(
    dag_id="saudi_dust_visibility_pipeline",
    description="NOAA station observations + CAMS dust → station-day table in Postgres → next-day low-visibility analysis",
    start_date=datetime(2026, 11, 1),
    schedule="0 3 * * *",  # daily at 03:00 UTC = 06:00 in Riyadh, before the day's flights
    catchup=False,
    max_active_runs=1,  # each run replaces the tables, so runs must not overlap
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},  # the sources are public APIs
    params={"force_refresh": Param(False, type="boolean", description="Re-download both raw sources even if cached")},
    tags=["week11", "saudi", "dust", "visibility", "postgres"],
)
def saudi_dust_visibility_pipeline():
    # =======================================================================
    # Source 1 — NOAA ISD hourly observations
    # =======================================================================
    @task_group(group_id="noaa_observations")
    def noaa_group():
        @task
        def extract(**context):
            """Download the hourly export (or reuse a recent cached copy)."""
            if context["params"]["force_refresh"] or not _is_fresh(sources.ISD_FILE):
                info = sources.download_isd()
                return {"path": str(sources.ISD_FILE), "downloaded": True, **info}
            return {"path": str(sources.ISD_FILE), "downloaded": False, "note": f"cache newer than {RAW_MAX_AGE_DAYS} days"}

        @task
        def validate(meta: dict):
            """Schema, volume and time-range checks on the raw file."""
            raw = sources.read_isd(Path(meta["path"]))
            missing = [c for c in ISD_REQUIRED if c not in raw.columns]
            if missing:
                raise ValueError(f"ISD: missing columns {missing}")
            if len(raw) < 500_000:
                raise ValueError(f"ISD: only {len(raw)} rows")
            first, last = raw["observation_date"].min(), raw["observation_date"].max()
            if (last - first).days < 365:
                raise ValueError(f"ISD: time range too short ({first} → {last})")
            return {
                "path": meta["path"], "rows": int(len(raw)), "stations": int(raw["station_id"].nunique()),
                "first_obs": str(first), "last_obs": str(last),
                "null_visibility": int(raw["visibility_distance"].isna().sum()),
                "duplicate_reports": int(raw.duplicated(["station_id", "observation_date"]).sum()),
            }

        @task
        def build_station_dim(meta: dict, **context):
            """Station dimension: name, coordinates, elevation, coverage. Sparse stations are excluded."""
            raw = sources.read_isd(Path(meta["path"]))
            stations = sources.station_table(raw)
            if not (stations["latitude"].between(15, 33).all() and stations["longitude"].between(34, 56).all()):
                raise ValueError("a station lies outside Saudi Arabia's bounding box")
            out = _stage(stations, context["ts_nodash"], "stations")
            return {**out, "window_end": raw["observation_date"].max().strftime("%Y-%m-%d")}

        @task
        def clean(validated: dict, stations: dict, **context):
            raw = sources.read_isd(Path(validated["path"]))
            ids = pd.read_csv(stations["path"])["station_id"].tolist()
            hourly, stats = transform.clean_isd(raw, ids)
            keep = ["station_id", "observation_date", "obs_date", "visibility_m", "rh_pct", "low_vis_dry",
                    "wind_speed_rate", "wind_u", "wind_v", "air_temperature", "atmospheric_sea_level_pressure"]
            return {**_stage(hourly[keep], context["ts_nodash"], "isd_hourly_clean"), "qc": stats}

        @task
        def aggregate_daily(cleaned: dict, **context):
            hourly = pd.read_csv(cleaned["path"], parse_dates=["observation_date", "obs_date"])
            daily = transform.daily_isd(hourly)
            return {**_stage(daily, context["ts_nodash"], "isd_daily"), "station_days": int(len(daily))}

        raw = extract()
        validated = validate(raw)
        stations = build_station_dim(raw)
        return {"daily": aggregate_daily(clean(validated, stations)), "stations": stations, "validated": validated}

    # =======================================================================
    # Source 2 — CAMS dust / particulates at the same coordinates
    # =======================================================================
    @task_group(group_id="cams_dust")
    def cams_group(stations: dict):
        @task
        def extract(stations: dict, **context):
            """One API call per station (or reuse a recent cached copy)."""
            if context["params"]["force_refresh"] or not _is_fresh(sources.CAMS_FILE):
                info = sources.download_cams(pd.read_csv(stations["path"]), end=stations["window_end"])
                return {"path": str(sources.CAMS_FILE), "downloaded": True, **info}
            return {"path": str(sources.CAMS_FILE), "downloaded": False, "note": f"cache newer than {RAW_MAX_AGE_DAYS} days"}

        @task
        def validate(meta: dict, stations: dict):
            raw = pd.read_csv(meta["path"])
            missing = [c for c in CAMS_REQUIRED if c not in raw.columns]
            if missing:
                raise ValueError(f"CAMS: missing columns {missing}")
            expected = set(pd.read_csv(stations["path"])["station_id"])
            absent = expected - set(raw["station_id"].unique())
            if absent:
                raise ValueError(f"CAMS: no data for stations {sorted(absent)}")
            null_share = float(raw["dust"].isna().mean())
            if null_share > 0.05:
                raise ValueError(f"CAMS: {null_share:.1%} of dust values are null")
            return {"path": meta["path"], "rows": int(len(raw)), "stations": int(raw["station_id"].nunique()),
                    "first_hour": str(raw["time"].min()), "last_hour": str(raw["time"].max()), "dust_null_share": null_share}

        @task
        def clean(validated: dict, **context):
            hourly, stats = transform.clean_cams(pd.read_csv(validated["path"]))
            return {**_stage(hourly, context["ts_nodash"], "cams_hourly_clean"), "qc": stats}

        @task
        def aggregate_daily(cleaned: dict, **context):
            hourly = pd.read_csv(cleaned["path"], parse_dates=["time", "obs_date"])
            daily = transform.daily_cams(hourly)
            return {**_stage(daily, context["ts_nodash"], "cams_daily"), "station_days": int(len(daily))}

        return aggregate_daily(clean(validate(extract(stations), stations)))

    # =======================================================================
    # Merge → load → quality gate
    # =======================================================================
    @task
    def merge_station_days(isd: dict, cams: dict, stations: dict, **context):
        """Inner join on (station_id, obs_date), history features, next-day target."""
        merged, stats = transform.merge_station_days(
            pd.read_csv(isd["path"], parse_dates=["obs_date"]),
            pd.read_csv(cams["path"], parse_dates=["obs_date"]),
            pd.read_csv(stations["path"]),
        )
        if stats["matched_station_days"] < 0.9 * stats["isd_station_days"]:
            raise ValueError(f"join lost too many station-days: {stats}")
        return {**_stage(merged, context["ts_nodash"], "station_daily"), "stats": stats}

    @task_group(group_id="load")
    def load_group(merged: dict, stations: dict):
        @task
        def station_daily(merged: dict, **context):
            """Full refresh in one transaction: a rerun replaces the table, it never appends duplicates."""
            df = pd.read_csv(merged["path"], parse_dates=[TIME_COLUMN])
            df[TIME_COLUMN] = df[TIME_COLUMN].dt.date
            with _engine().begin() as conn:
                df.to_sql(TABLE, conn, if_exists="replace", index=False, chunksize=1000, method="multi")
                conn.execute(f"ALTER TABLE {TABLE} ADD PRIMARY KEY (station_id, {TIME_COLUMN})")
                conn.execute(f"CREATE INDEX ix_{TABLE}_date ON {TABLE} ({TIME_COLUMN})")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS pipeline_runs (run_id text PRIMARY KEY, "
                    "loaded_at timestamptz DEFAULT now(), n_rows int, n_cols int)"
                )
                conn.execute(
                    "INSERT INTO pipeline_runs (run_id, n_rows, n_cols) VALUES (%s, %s, %s) "
                    "ON CONFLICT (run_id) DO UPDATE SET loaded_at = now(), n_rows = EXCLUDED.n_rows, n_cols = EXCLUDED.n_cols",
                    (context["ts_nodash"], len(df), df.shape[1]),
                )
            return {"table": TABLE, "n_rows": int(len(df)), "n_cols": int(df.shape[1])}

        @task
        def station_dim(stations: dict):
            df = pd.read_csv(stations["path"])
            with _engine().begin() as conn:
                df.to_sql(STATION_TABLE, conn, if_exists="replace", index=False)
                conn.execute(f"ALTER TABLE {STATION_TABLE} ADD PRIMARY KEY (station_id)")
            return {"table": STATION_TABLE, "n_rows": int(len(df))}

        return {"fact": station_daily(merged), "dim": station_dim(stations)}

    @task
    def quality_checks(fact: dict, dim: dict):
        """SQL checks on what was actually loaded. A failure blocks the analysis tasks."""
        hook = PostgresHook(postgres_conn_id=WAREHOUSE_CONN_ID)
        n, n_keys, n_null_target, rate, first, last, n_stations, orphans, bad_vis, future_gap = hook.get_first(
            f"""
            SELECT count(*),
                   count(DISTINCT (station_id, {TIME_COLUMN})),
                   count(*) FILTER (WHERE {TARGET} IS NULL),
                   avg({TARGET}::float),
                   min({TIME_COLUMN}), max({TIME_COLUMN}),
                   count(DISTINCT station_id),
                   count(*) FILTER (WHERE station_id NOT IN (SELECT station_id FROM {STATION_TABLE})),
                   count(*) FILTER (WHERE vis_min_m < 0 OR vis_min_m > 10000 OR rh_mean_pct < 0 OR rh_mean_pct > 100
                                       OR dust_mean_ugm3 < 0 OR low_vis_hours > 24),
                   count(*) FILTER (WHERE ({TARGET} = 1) <> (next_day_low_vis_hours >= 2))
            FROM {TABLE}
            """
        )
        checks = {
            "row_count_matches_load": n == fact["n_rows"],
            "station_day_key_unique": n_keys == n,
            "target_never_null": n_null_target == 0,
            "event_rate_plausible": 0.005 < rate < 0.5,
            "covers_at_least_two_years": (last - first).days >= 730,
            "every_row_has_a_known_station": orphans == 0 and n_stations == dim["n_rows"],
            "values_in_physical_range": bad_vis == 0,
            "target_consistent_with_outcome": future_gap == 0,
        }
        failed = [k for k, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"data quality checks failed: {failed}")
        return {"checks": checks, "n_rows": int(n), "n_stations": int(n_stations), "event_rate": float(rate),
                "first_date": str(first), "last_date": str(last)}

    # =======================================================================
    # Analysis — three independent tasks that read the cleaned data from Postgres
    # =======================================================================
    @task_group(group_id="analysis")
    def analysis_group(quality: dict):
        @task
        def train_model(quality: dict, **context):
            """One simple model vs. the persistence baseline, on a chronological split."""
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.impute import SimpleImputer
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import (
                ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score, average_precision_score,
                f1_score, precision_score, recall_score, roc_auc_score,
            )
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            df = pd.read_sql(f"SELECT * FROM {TABLE} ORDER BY {TIME_COLUMN}, station_id", _engine())
            dates = sorted(df[TIME_COLUMN].unique())
            cutoff = dates[int(len(dates) * 0.8)]  # train on the first 80% of days, test on the rest
            train, test = df[df[TIME_COLUMN] < cutoff], df[df[TIME_COLUMN] >= cutoff]
            features = [c for c in df.columns if c not in {TARGET, *NON_FEATURES}]
            model = Pipeline([
                ("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])
            model.fit(train[features], train[TARGET])
            proba = model.predict_proba(test[features])[:, 1]
            pred = (proba >= 0.5).astype(int)
            y = test[TARGET]
            naive = test[PERSISTENCE_COLUMN]

            def scores(p) -> dict:
                return {"precision": float(precision_score(y, p, zero_division=0)), "recall": float(recall_score(y, p, zero_division=0)),
                        "f1": float(f1_score(y, p, zero_division=0)), "accuracy": float(accuracy_score(y, p))}

            metrics = {
                "model": "logistic_regression (class_weight=balanced)",
                "split": f"chronological: train < {cutoff}, test ≥ {cutoff}",
                "n_train": int(len(train)), "n_test": int(len(test)), "n_features": len(features),
                "test_event_rate": float(y.mean()),
                **scores(pred),
                "roc_auc": float(roc_auc_score(y, proba)), "pr_auc": float(average_precision_score(y, proba)),
                "persistence_baseline": scores(naive),
                "always_no_event_accuracy": float(1 - y.mean()),
            }

            fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
            PrecisionRecallDisplay.from_predictions(y, proba, ax=axes[0], name="logistic regression")
            p0 = metrics["persistence_baseline"]
            axes[0].scatter([p0["recall"]], [p0["precision"]], color="#c8553d", zorder=5, label="persistence (tomorrow = today)")
            axes[0].axhline(y.mean(), linestyle=":", color="grey", label=f"event rate {y.mean():.1%}")
            axes[0].set_title("Precision–recall on the last 20% of days")
            axes[0].legend(loc="upper right")
            ConfusionMatrixDisplay.from_predictions(y, pred, ax=axes[1], display_labels=["clear", "low-vis"], colorbar=False, cmap="Blues")
            axes[1].set_title("Confusion matrix (threshold 0.5)")
            fig.tight_layout()
            out_dir = RUNS / context["ts_nodash"]
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / "model_eval.png", dpi=130)
            plt.close(fig)

            flat = {k: v for k, v in metrics.items() if not isinstance(v, dict)}
            flat.update({f"persistence_{k}": v for k, v in p0.items()})
            pd.DataFrame([{"run_id": context["ts_nodash"], **flat}]).to_sql("model_metrics", _engine(), if_exists="append", index=False)
            return {"metrics": metrics, "plot": str(out_dir / "model_eval.png")}

        @task
        def seasonality_plot(quality: dict, **context):
            """Charts built from SQL aggregates computed in Postgres."""
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            engine = _engine()
            overall = quality["event_rate"]
            by_month = pd.read_sql(f"SELECT month, avg({TARGET}::float) AS rate FROM {TABLE} GROUP BY month ORDER BY month", engine)
            by_station = pd.read_sql(
                f"SELECT station_name, avg({TARGET}::float) AS rate, count(*) AS days FROM {TABLE} "
                "GROUP BY station_name ORDER BY rate DESC", engine)
            by_dust = pd.read_sql(
                f"SELECT ntile(10) OVER (ORDER BY dust_mean_ugm3) AS decile, dust_mean_ugm3, {TARGET} FROM {TABLE} "
                "WHERE dust_mean_ugm3 IS NOT NULL", engine).groupby("decile").agg(rate=(TARGET, "mean"), dust=("dust_mean_ugm3", "median"))
            by_today = pd.read_sql(f"SELECT {PERSISTENCE_COLUMN} AS today, avg({TARGET}::float) AS rate FROM {TABLE} GROUP BY 1 ORDER BY 1", engine)

            fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
            names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            axes[0, 0].bar([names[m - 1] for m in by_month["month"]], by_month["rate"] * 100, color="#b7791f")
            axes[0, 0].axhline(overall * 100, linestyle="--", color="grey", label=f"overall {overall:.1%}")
            axes[0, 0].set_title("Share of days followed by a low-visibility day, by month")
            axes[0, 0].set_ylabel("%")
            axes[0, 0].legend()
            axes[0, 1].barh(by_station["station_name"].str.title().str.slice(0, 28)[::-1], by_station["rate"][::-1] * 100, color="#3b6ea5")
            axes[0, 1].axvline(overall * 100, linestyle="--", color="grey")
            axes[0, 1].set_title("By station")
            axes[0, 1].set_xlabel("%")
            axes[0, 1].tick_params(axis="y", labelsize=7)
            axes[1, 0].plot(by_dust["dust"], by_dust["rate"] * 100, marker="o", color="#b7791f")
            axes[1, 0].set_xscale("log")
            axes[1, 0].set_title("Next-day event rate by today's CAMS dust (deciles)")
            axes[1, 0].set_xlabel("median dust concentration in decile, µg/m³ (log)")
            axes[1, 0].set_ylabel("%")
            axes[1, 1].bar(["clear today", "low-vis today"], by_today["rate"] * 100, color=["#3b6ea5", "#c8553d"])
            axes[1, 1].set_title("Persistence: next-day event rate given today")
            axes[1, 1].set_ylabel("%")
            fig.suptitle(
                f"{TABLE}: {quality['n_rows']:,} station-days, {quality['n_stations']} Saudi stations, "
                f"{quality['first_date']} to {quality['last_date']} (read from Postgres)", fontsize=12)
            fig.tight_layout()
            out_dir = RUNS / context["ts_nodash"]
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / "dust_overview.png", dpi=130)
            plt.close(fig)
            return {
                "plot": str(out_dir / "dust_overview.png"),
                "peak_month": names[int(by_month.loc[by_month["rate"].idxmax(), "month"]) - 1],
                "peak_month_rate": float(by_month["rate"].max()),
                "highest_station": str(by_station.iloc[0]["station_name"]), "highest_station_rate": float(by_station.iloc[0]["rate"]),
                "rate_after_clear_day": float(by_today.iloc[0]["rate"]), "rate_after_low_vis_day": float(by_today.iloc[-1]["rate"]),
            }

        @task
        def export_for_agents(quality: dict):
            """Publish the warehouse table as the CSV the Part 2 / Part 3 agents read."""
            df = pd.read_sql(f"SELECT * FROM {TABLE} ORDER BY {TIME_COLUMN}, station_id", _engine())
            PUBLISH.parent.mkdir(parents=True, exist_ok=True)
            # Write to a temp file and rename: a reader never sees a half-written CSV.
            tmp = PUBLISH.with_suffix(".csv.part")
            df.to_csv(tmp, index=False)
            tmp.replace(PUBLISH)
            return {"published": str(PUBLISH), "n_rows": int(len(df)), "n_cols": int(df.shape[1])}

        return {"model": train_model(quality), "plot": seasonality_plot(quality), "export": export_for_agents(quality)}

    @task
    def write_report(isd: dict, merged: dict, quality: dict, model: dict, plot: dict, export: dict, **context):
        report = {
            "run_id": context["ts_nodash"], "logical_date": context["ds"],
            "sources": {"noaa_isd": isd, "merge": merged["stats"]},
            "warehouse": {"table": TABLE, "rows": quality["n_rows"], "stations": quality["n_stations"],
                          "first_date": quality["first_date"], "last_date": quality["last_date"], "event_rate": quality["event_rate"]},
            "quality_checks": quality["checks"],
            "model_metrics": model["metrics"],
            "findings": {k: v for k, v in plot.items() if k != "plot"},
            "published_for_agents": export["published"],
        }
        LATEST.mkdir(parents=True, exist_ok=True)
        (LATEST / "report.json").write_text(json.dumps(report, indent=2, default=str))
        shutil.copyfile(model["plot"], LATEST / "model_eval.png")
        shutil.copyfile(plot["plot"], LATEST / "dust_overview.png")
        return report

    # -----------------------------------------------------------------------
    noaa = noaa_group()
    merged = merge_station_days(noaa["daily"], cams_group(noaa["stations"]), noaa["stations"])
    loaded = load_group(merged, noaa["stations"])
    quality = quality_checks(loaded["fact"], loaded["dim"])
    results = analysis_group(quality)
    write_report(noaa["validated"], merged, quality, results["model"], results["plot"], results["export"])


saudi_dust_visibility_pipeline()
