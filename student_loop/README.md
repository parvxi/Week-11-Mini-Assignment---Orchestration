# Forecast agent workflow (Part 2)

- [`graph.py`](graph.py) — the LangGraph: agents, routers, wiring, and the event stream the UI listens to.
- [`training.py`](training.py) — deterministic helpers the agents call: chronological splits, leakage audit,
  persistence benchmark, fit, threshold tuning, holdout scoring.
- [`run_cli.py`](run_cli.py) — run the graph once from a terminal (`make agents-cli`).

See the repo [README](../README.md) for what changed from the class template and why.
