COMPOSE := docker compose -f infra/docker-compose.yml --env-file .env
VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: ensure-env up down down-mac down-windows logs test install health venv \
	demo-sse rebuild-gateway frontend start start-mac start-windows airflow \
	dag-run agents-cli test-unit adaptive-run eval fetch-data frontend-build

ensure-env:
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

# ---------------------------------------------------------------------------
# Web app — pick your OS (start is the same; down kills Vite differently)
# ---------------------------------------------------------------------------

start:
	@echo ""
	@echo "  Use one of:"
	@echo "    make start-mac        # macOS"
	@echo "    make start-windows    # Windows (run from Git Bash)"
	@echo ""
	@exit 1

down:
	@echo ""
	@echo "  Use one of:"
	@echo "    make down-mac         # macOS (kills Vite via lsof + Docker down)"
	@echo "    make down-windows     # Windows (kills Vite via PowerShell + Docker down)"
	@echo ""
	@exit 1

# Docker stack + Vite UI (blocks on npm run dev). Same on both OSes.
start-mac: ensure-env
	$(COMPOSE) up --build -d
	@echo "Waiting for gateway on :8000..."
	@i=0; while [ $$i -lt 36 ]; do \
	  if curl -sf http://127.0.0.1:8000/healthz >/dev/null; then echo "gateway ok"; break; fi; \
	  i=$$((i+1)); sleep 5; \
	  if [ $$i -eq 36 ]; then echo "gateway not ready — try: make health / make logs"; exit 1; fi; \
	done
	@echo "Open http://127.0.0.1:5173"
	@echo "Stop with:  make down-mac  (macOS)  or  make down-windows  (Windows)"
	cd frontend && npm install && npm run dev

# Same as start-mac. On Windows: run from Git Bash (needs curl + Make).
start-windows: start-mac

frontend:
	cd frontend && npm run dev

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -e ".[dev]"

up: ensure-env
	$(COMPOSE) up --build -d
	@echo "Waiting for service health..."
	@$(COMPOSE) ps

airflow: ensure-env
	$(COMPOSE) --profile airflow up --build -d
	@echo "Airflow starting (often 1–2 min). Check: $(COMPOSE) --profile airflow ps"
	@echo "UI: http://localhost:8088  (admin / admin unless .env overrides)"

# Trigger the Part 1 DAG from the CLI (same as pressing ▶ in the UI).
dag-run: ensure-env
	$(COMPOSE) --profile airflow exec -T airflow-scheduler airflow dags unpause saudi_dust_visibility_pipeline
	$(COMPOSE) --profile airflow exec -T airflow-scheduler airflow dags trigger saudi_dust_visibility_pipeline

# Part 2 without the browser: run the agent graph once, print JSON.
agents-cli: ensure-env
	$(COMPOSE) exec -T gateway python /student_loop/run_cli.py \
	  --dataset /bundled/airflow/saudi_dust_daily_latest.csv --target low_vis_next_day --max-cycles 3

# The four targets below need Python with the project's packages.
# - On your own machine they run inside the already-built Docker images (no local Python setup needed).
# - Inside the dev container (.devcontainer/) the packages are already installed, so they run directly.
ifneq ($(wildcard /.dockerenv),)
PY_GATEWAY := PYTHONPATH=. python
PY_AIRFLOW := PYTHONPATH=. python
else
PY_GATEWAY := docker run --rm -v "$(CURDIR):/w" -w /w -e PYTHONPATH=/w saudi-dust-forecast-lab-gateway python
PY_AIRFLOW := docker run --rm -v "$(CURDIR):/repo" -w /repo -e PYTHONPATH=/repo --entrypoint python saudi-dust-forecast-lab-airflow-scheduler
endif

# Unit tests.
test-unit:
ifneq ($(wildcard /.dockerenv),)
	PYTHONPATH=. python -m pytest tests/unit -q -W ignore
else
	docker run --rm -v "$(CURDIR):/w" -w /w -e PYTHONPATH=/w saudi-dust-forecast-lab-gateway \
	  sh -c "pip install -q pytest 2>/dev/null && python -m pytest tests/unit -q -W ignore"
endif

# Download both raw sources again (the DAG does this itself when the cache is older than 30 days).
fetch-data:
	$(PY_AIRFLOW) -m dust_pipeline.sources

# Part 3: run the adaptive multi-agent system once on the Airflow-built table.
adaptive-run:
	$(PY_GATEWAY) -W ignore adaptive_loop/graph.py --max-trials 6

# Part 3: evaluation suite, all systems → evals/results/ (results.md, results.json, dashboard.html).
eval:
	$(PY_GATEWAY) -W ignore -m evals.run_evals

# Type-check and production-build the UI.
frontend-build:
	cd frontend && npm install && npm run build

down-mac: ensure-env
	@-pids=$$(lsof -tiTCP:5173 -sTCP:LISTEN 2>/dev/null); \
	if [ -n "$$pids" ]; then echo "Stopping Vite on :5173 ($$pids)"; kill $$pids 2>/dev/null || true; fi
	$(COMPOSE) --profile airflow down

down-windows: ensure-env
	@powershell.exe -NoProfile -Command "Get-NetTCPConnection -LocalPort 5173 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('Stopping Vite on :5173 (PID ' + $$_.OwningProcess + ')'); Stop-Process -Id $$_.OwningProcess -Force -ErrorAction SilentlyContinue }"
	$(COMPOSE) --profile airflow down

logs: ensure-env
	$(COMPOSE) logs -f --tail=100

test: install
	$(PYTHON) -m pytest tests/contract -q

rebuild-gateway: ensure-env
	$(COMPOSE) up --build -d gateway

health:
	@for port in 8000 8011 8002 8003 8004 8005 8006; do \
	  printf "localhost:%s -> " "$$port"; \
	  curl -sf "http://127.0.0.1:$$port/healthz" || printf "FAIL"; \
	  printf "\n"; \
	done

# Stage A1: create a mock session and stream SSE until done.
# Usage: make demo-sse SCENARIO=happy
SCENARIO ?= happy
demo-sse:
	@SID=$$(curl -sf -X POST http://127.0.0.1:8000/sessions \
	  -H 'Content-Type: application/json' \
	  -d '{"target_column":"churned","scenario":"$(SCENARIO)"}' | $(PYTHON) -c 'import sys,json; print(json.load(sys.stdin)["session_id"])'); \
	echo "session=$$SID scenario=$(SCENARIO)"; \
	curl -sf -N "http://127.0.0.1:8000/sessions/$$SID/events"
