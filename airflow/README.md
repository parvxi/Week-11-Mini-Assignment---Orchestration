# Airflow (Part 1)

One DAG: [`dags/saudi_dust_visibility_pipeline.py`](dags/saudi_dust_visibility_pipeline.py). Its download and
transformation logic lives in [`../dust_pipeline/`](../dust_pipeline/) so it can be unit-tested without Airflow.

Airflow runs only in Docker (`make airflow`, UI on http://localhost:8088). The repo is mounted at `/repo`
inside the containers, and the warehouse Postgres is reachable through the Airflow connection `warehouse_db`
(set by `AIRFLOW_CONN_WAREHOUSE_DB` in [`infra/docker-compose.yml`](../infra/docker-compose.yml)).

See the repo [README](../README.md) for the full description.
