# Requirement audit

Verified on 12 November 2026 by running each item. Status is one of PASS, FAIL, NOT VERIFIED.
The instructor decides grades; this table only records what was built and what was checked.

## Part 1 — Airflow data pipeline

| Requirement | Implemented where | Actual evidence | Status |
| --- | --- | --- | --- |
| Deploy Airflow | `infra/docker-compose.yml`, `infra/Dockerfile.airflow` | Airflow 2.10.4 running in Docker on :8088; `airflow dags list-import-errors` returned none | PASS |
| Scheduled DAG | `saudi_dust_visibility_pipeline.py`: `schedule="0 3 * * *"`, `catchup=False` | Scheduled run `scheduled__2026-11-11T03:00:00` succeeded; `airflow dags list-runs` shows it | PASS |
| Orchestrate ingestion through analysis | 17 tasks: extract → validate → clean → aggregate → merge → load → quality gate → analysis → report | 4 runs, all 17 tasks success (`airflow_dag_graph_success.png`) | PASS |
| Independent tasks in parallel | NOAA branch ∥ station table + CAMS branch; 2 load tasks; 3 analysis tasks | `airflow_dag_gantt_parallel.png` | PASS |
| At least two related datasets | NOAA ISD (KAPSARC) and CAMS (Open-Meteo) | 948,244 and 752,640 rows downloaded inside Airflow (`force_refresh` run `manual__2026-11-12T19:36:30`), `datasets/raw/manifest.json` | PASS |
| Clean and transform each dataset | `dust_pipeline/transform.py`: `clean_isd`, `daily_isd`, `clean_cams`, `daily_cams` | 216,109 duplicates removed, sentinels nulled; unit tests in `tests/unit/test_dust_pipeline.py` | PASS |
| TaskGroups | `noaa_observations`, `cams_dust`, `load`, `analysis` | Graph screenshot with groups expanded | PASS |
| Merge the datasets | `merge_station_days` (one-to-one on station and day) | 29,796 of 29,796 station-days matched | PASS |
| Load into a database | `load.station_daily`, `load.station_dim` → PostgreSQL `warehouse` | Query: 29,486 rows, 29,486 distinct keys, 44 columns; `stations` 28 rows | PASS |
| Read cleaned data back from the database | `quality_checks`, `analysis.*` use `PostgresHook` / `pd.read_sql` | 8 SQL checks pass in every successful run | PASS |
| Analysis (visualization or simple ML) | `analysis.train_model`, `analysis.seasonality_plot` | `docs/images/model_eval.png`, `docs/images/dust_overview.png`, `model_metrics` table (4 rows) | PASS |
| Document sources, transformations, structure, results, scheduling, parallelism | `README.md`, `docs/PROJECT_SELECTION.md` | Files present | PASS |

## Part 2 — Agent-based ML workflow

| Requirement | Implemented where | Actual evidence | Status |
| --- | --- | --- | --- |
| Use the in-class workflow as the template | `student_loop/graph.py` on the class gateway, UI and `loop_runtime.py` | Runs through the class `/student-loop` endpoints | PASS |
| Train and evaluate on the cleaned data from Part 1 | Agents read `saudi_dust_daily_latest.csv`, exported from Postgres by the DAG | CSV and table both 29,486 × 44; UI run completed | PASS |
| Add, remove or modify agents | +Leakage Auditor, +Baseline Benchmark, +Threshold Tuner, +training router; ML Engineer, Judge, Memory writer, Final holdout modified; Context builder removed | `get_status()` returns 9 nodes; unit test `test_graph_has_the_modified_agents` | PASS |
| Update the live agent graph | `frontend/src/graph/studentLoopConfig.ts`, drawn from the compiled graph | `agent_graph_after_run.png` | PASS |
| Report accuracy or another appropriate metric | Holdout F1, precision, recall, ROC-AUC, PR-AUC, accuracy, confusion matrix, persistence baseline | `agent_evaluation_results.png`: F1 0.269, ROC-AUC 0.802 | PASS |

## Submission requirements

| Requirement | Implemented where | Actual evidence | Status |
| --- | --- | --- | --- |
| Repository with DAG and modified agent code | This folder | Files present | PASS |
| Pushed to a remote repository | — | Not done: I was told not to commit or push | NOT VERIFIED |
| Instructions for running both workflows | `README.md` "Running it" | `make dag-run`, `make agents-cli`, `make adaptive-run`, `make test-unit`, `make frontend-build` each run successfully | PASS |
| Clean-clone setup from scratch (`cp .env.example .env`, `make airflow`) | `Makefile`, compose | Stack was built and started from this folder, but with an existing `.env`, not from a fresh clone | NOT VERIFIED |
| Containerized setup | Dockerfiles, compose, `.devcontainer/devcontainer.json` | Compose stack built and healthy; `docker compose config` valid | PASS |
| Dev container opens | `.devcontainer/devcontainer.json` | Opened in VS Code; packages installed; `make test-unit` (23 passed) and `make adaptive-run` run inside it | PASS |
| Stack started from inside the dev container | — | Not supported: `make airflow` must be run from a host terminal (stated in the README) | NOT VERIFIED |
| Screenshot: successful DAG execution | `docs/screenshots/airflow_dag_graph_success.png`, `airflow_dag_run_details.png` | Present, from a real run | PASS |
| Screenshot: DAG graph view | `docs/screenshots/airflow_dag_graph_success.png` | Present | PASS |
| Screenshot: agent graph | `docs/screenshots/agent_graph_after_run.png` | Present | PASS |
| Screenshot: model evaluation results | `docs/screenshots/agent_evaluation_results.png` | Present | PASS |
| README with pipeline, agent modifications, findings | `README.md` | Present | PASS |

## Kedar's Slack clarification

| Requirement | Implemented where | Actual evidence | Status |
| --- | --- | --- | --- |
| Find your own datasets | NOAA ISD via KAPSARC, CAMS via Open-Meteo | Neither is a class dataset; selection record in `docs/PROJECT_SELECTION.md` | PASS |
| At least two distinct datasets | Two producers, two APIs, two licences | `datasets/raw/manifest.json` | PASS |
| Pipeline created from scratch | `saudi_dust_visibility_pipeline.py`, `dust_pipeline/` | No class DAG is in this repo | PASS |
| Unique concept that solves a problem | Next-day dust / low-visibility forecast at Saudi airports | README problem statement | PASS |
| Class LangGraph UI may be reused | Gateway and UI shell reused, stated in the README | — | PASS |

## Part 3 — bonus

| Requirement | Implemented where | Actual evidence | Status |
| --- | --- | --- | --- |
| Multi-agent system with distinct responsibilities | `adaptive_loop/agents.py`: 12 registry agents + planner | `plan_integrity` case passes | PASS |
| Workflow represented as a graph, created by an agent | `planner` → JSON plan → `compile_plan` → LangGraph | `adaptive_workflow_graph_planned.png`; Mermaid in the README | PASS |
| Adaptive: selecting agents | Planner includes probe, tuner, benchmark only when the data warrants | `agent_selection` case 1.00 | PASS |
| Adaptive: routing on intermediate results | `supervise()` in `adaptive_loop/graph.py` | `repair_routing`, `efficiency_no_signal`, `plateau_stop`, `invalid_data_stops` pass | PASS |
| Adaptive: retrying failed steps, bounded | Up to 2 retries per trial, then switch family | `fault_transient`, `fault_family_outage`, `retry_limit` pass; `adaptive_run_retry.png` | PASS |
| Evaluation suite: intermediate quality | 4 cases | `evals/results/results.json` | PASS |
| Evaluation suite: final quality | 4 cases | Same | PASS |
| Evaluation suite: coordination | 11 cases | Same | PASS |
| Each case has input, expected behaviour, scoring, threshold | `evals/cases.py` `Case` dataclass | Unit test `test_every_case_is_fully_specified` | PASS |
| Adaptive system passes its own suite | — | 18 of 19. `forecast_skill` fails: F1 is 89% of persistence, bar is 90% | FAIL (1 case) |
| Compare with fixed Airflow workflow | `evals/systems.py::run_fixed_pipeline` (class `profile_light` + `train_baseline`) | 3 of 19; table in README | PASS |
| Baseline 1 executed through the Airflow scheduler | — | The class task functions are called directly in fixed order, not via Airflow | NOT VERIFIED |
| Compare with the simple class agentic system | `evals/systems.py::run_simple_loop` (`mini_loop/graph.py`, unmodified) | 2 of 19; table in README | PASS |
| Source, setup instructions, workflow graph, evaluation suite, summary | `adaptive_loop/`, `evals/`, README Part 3 | Present | PASS |
| Dashboard | `evals/results/dashboard.html` | File opens in a browser; `eval_suite_results.png` | PASS |
| Adaptive behaviour shown in a *live* graph | Dashboard draws each real run's path on the planned graph after the run | Rendered from recorded runs, not streamed while running | FAIL |
| LangSmith evaluation | — | No key available; local suite used instead | NOT VERIFIED |
| Adaptive system runs with LLM planner / strategist | `--llm` flag | One full run with the free OpenRouter model: the LLM wrote the plan and chose three model families; holdout F1 0.241, ROC-AUC 0.771 | PASS |
| Evaluation suite run with the LLM on | — | The suite runs with the LLM off so results are repeatable | NOT VERIFIED |

## Other checks

| Check | Result | Status |
| --- | --- | --- |
| Unit tests | 23 passed, 4 skipped (class tests that need files not in this repo) | PASS |
| Frontend type-check and build | `tsc -b && vite build` succeeds | PASS |
| No secrets in tracked files | `.env` is gitignored. A key that had been pasted into `.env.example` (inherited from the local class copy) was removed | PASS |
| Paid API usage | None: the only LLM used is a free OpenRouter model | PASS |
| `~/automl-research-lab` and the fraud project untouched | No files deleted. In the fraud project only `.env.example` was edited, to remove the same pasted key | PASS |
