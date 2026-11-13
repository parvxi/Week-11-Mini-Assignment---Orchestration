"""The systems under test, wrapped so every one returns the same result shape.

  fixed_pipeline  Baseline 1 — the class Airflow workflow's task logic (profile_light →
                  train_baseline from mini_loop/baseline.py) executed in its fixed order.
                  It is called directly here instead of through the Airflow scheduler so
                  the suite can run it on many datasets; the steps are the same functions.
  simple_loop     Baseline 2 — the in-class LangGraph loop (mini_loop/graph.py), unmodified.
  part2_loop      The Part 2 workflow from this repo (student_loop/graph.py), for reference.
  adaptive        The Part 3 system (adaptive_loop/).

Every system gets the same table, trial budget and target. The two class baselines are run
exactly as written; they were built for random splits and name-based leak rules, which is
part of what is being compared.

All systems run without an LLM here (their deterministic paths), so results are reproducible
and differences come from the workflow design, not from model sampling.

The REFEREE re-fits each system's final model spec with one shared procedure on the shared
chronological holdout (the most recent 20% of days), so predictive quality is measured
identically for every system, separately from how the system was orchestrated.
"""

from __future__ import annotations

import os
import time
import warnings
from contextlib import contextmanager

import pandas as pd
from sklearn.pipeline import Pipeline

from student_loop.training import fit_holdout

BUDGET = 3
TARGET_SCORE = 0.35  # validation F1 a system must reach to stop early (persistence scores about 0.20)
SEED = 42
HOLDOUT_FRAC = 0.2

SYSTEM_LABELS = {
    "fixed_pipeline": "Baseline 1 · fixed Airflow workflow",
    "simple_loop": "Baseline 2 · class agent loop",
    "part2_loop": "Part 2 loop (this repo)",
    "adaptive": "Part 3 adaptive system",
}


# ---------------------------------------------------------------------------
# Fault injection — applied identically to every system
# ---------------------------------------------------------------------------
class InjectedFault(RuntimeError):
    pass


@contextmanager
def inject(fault: dict | None):
    """Make sklearn Pipeline.fit crash according to `fault` while the system runs.

    {"first_n": k}        the first k fits crash (transient outage)
    {"estimator": "X"}    every fit whose final estimator class is X crashes
    {"all": True}         every fit crashes
    """
    counter = {"fits": 0, "crashes": 0}
    if not fault:
        yield counter
        return
    original = Pipeline.fit

    def flaky_fit(self, X, y=None, **kw):
        counter["fits"] += 1
        final = self.steps[-1][1].__class__.__name__
        if (
            fault.get("all")
            or counter["fits"] <= int(fault.get("first_n") or 0)
            or final == fault.get("estimator")
        ):
            counter["crashes"] += 1
            raise InjectedFault(f"injected fault: {final} fit #{counter['fits']} crashed")
        return original(self, X, y, **kw)

    Pipeline.fit = flaky_fit
    try:
        yield counter
    finally:
        Pipeline.fit = original


@contextmanager
def _no_llm():
    saved = {k: os.environ.pop(k, None) for k in ("OPENROUTER_API_KEY", "CURSOR_API_KEY")}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def _empty(system: str) -> dict:
    return {
        "system": system,
        "completed": False,
        "error": None,
        "excluded_columns": None,  # columns kept out of the final model
        "final_spec": None,
        "threshold": 0.5,
        "trials": 0,
        "trial_budget": BUDGET,
        "stop_reason": None,
        "reported_metrics": None,  # what the system itself claims
        "agents_planned": None,
        "agents_run": None,
        "agent_steps": None,
        "plan": None,
        "probe": None,
        "route_log": None,
        "retries": None,
        "validation": None,
        "trial_log": None,
        "repairs": None,
        "steps": None,
    }


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------
def run_fixed_pipeline(path: str, target: str, budget: int = BUDGET) -> dict:
    from mini_loop.baseline import profile_light, train_baseline

    out = _empty("fixed_pipeline")
    profile = profile_light(path, target)  # task 1: name-based profile
    trained = train_baseline(  # task 2: one logistic regression, default threshold
        path, target_column=target, drop_columns=profile["recommended_drops"], seed=SEED, holdout_frac=HOLDOUT_FRAC
    )
    out.update(
        completed=True,
        excluded_columns=trained["baseline_spec"]["drop_columns"],
        final_spec=trained["baseline_spec"],
        trials=1,
        stop_reason="fixed_steps_done",
        reported_metrics=trained["baseline_validation_metrics"],
        agent_steps=2,
    )
    return out


def _run_class_style_loop(system: str, module: str, path: str, target: str, budget: int) -> dict:
    import importlib

    out = _empty(system)
    graph = importlib.import_module(module)
    res = graph.run_simple_loop(
        dataset_path=path, target_column=target, target_score=TARGET_SCORE, max_cycles=budget, seed=SEED, holdout_frac=HOLDOUT_FRAC
    )
    spec = res.get("best_spec")
    steps = res.get("steps") or []
    out.update(
        completed=bool(res.get("holdout_metrics")),
        excluded_columns=(spec or {}).get("drop_columns"),
        final_spec=spec,
        threshold=float(res.get("best_threshold") or 0.5),
        trials=int(res.get("cycles") or 0),
        stop_reason=res.get("stop_reason"),
        reported_metrics=res.get("holdout_metrics"),
        agents_run=list(dict.fromkeys(s.get("node") for s in steps)),
        agent_steps=len(steps),
        retries=sum(1 for s in steps if s.get("repaired")),
    )
    return out


def run_simple_loop(path: str, target: str, budget: int = BUDGET) -> dict:
    return _run_class_style_loop("simple_loop", "mini_loop.graph", path, target, budget)


def run_part2_loop(path: str, target: str, budget: int = BUDGET) -> dict:
    return _run_class_style_loop("part2_loop", "student_loop.graph", path, target, budget)


def run_adaptive(path: str, target: str, budget: int = BUDGET) -> dict:
    from adaptive_loop.graph import run_adaptive as _run

    out = _empty("adaptive")
    res = _run(dataset_path=path, target_column=target, target_score=TARGET_SCORE, max_trials=budget, seed=SEED, holdout_frac=HOLDOUT_FRAC)
    best = res.get("best") or {}
    report = res.get("report") or {}
    out.update(
        completed=report.get("stop_reason") is not None,
        excluded_columns=(res.get("audit") or {}).get("drop_columns"),
        final_spec=best.get("spec"),
        threshold=float(best.get("threshold") or 0.5),
        trials=report.get("trials", 0),
        stop_reason=res.get("stop_reason"),
        reported_metrics=res.get("holdout"),
        agents_planned=res["plan"]["agents"],
        agents_run=report.get("agents_run"),
        agent_steps=report.get("agent_steps"),
        plan=res["plan"],
        probe=res.get("probe"),
        validation=res.get("validation"),
        trial_log=res.get("trials"),
        repairs=report.get("repairs"),
        steps=res.get("steps"),
        route_log=res.get("route_log"),
        retries=report.get("retries"),
    )
    return out


SYSTEMS = {
    "fixed_pipeline": run_fixed_pipeline,
    "simple_loop": run_simple_loop,
    "part2_loop": run_part2_loop,
    "adaptive": run_adaptive,
}


# ---------------------------------------------------------------------------
def referee(path: str, target: str, result: dict) -> dict | None:
    """Score a system's final model on the shared holdout with the shared procedure."""
    spec = result.get("final_spec")
    if not spec:
        return None
    df = pd.read_csv(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fit_holdout(df, target, spec, float(result.get("threshold") or 0.5), SEED, HOLDOUT_FRAC)
    out = {k: m[k] for k in ("f1", "precision", "recall", "accuracy", "roc_auc", "pr_auc", "threshold", "n_holdout")}
    out["persistence_f1"] = (m.get("persistence_baseline") or {}).get("f1")
    return out


def execute(system: str, path: str, target: str, fault: dict | None = None, budget: int = BUDGET,
            referee_path: str | None = None) -> dict:
    """Run one system on one dataset; never raises. The referee runs with no fault active.

    `referee_path` lets the referee score the final spec on a clean copy of the table when the
    system was deliberately given a corrupted one.
    """
    t0 = time.time()
    with _no_llm(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with inject(fault) as counter:
            try:
                result = SYSTEMS[system](path, target, budget)
            except Exception as exc:  # noqa: BLE001 — a crash is a result, not a harness error
                result = {**_empty(system), "error": f"{type(exc).__name__}: {exc}"[:300]}
    result["seconds"] = round(time.time() - t0, 2)
    result["fits_attempted"] = counter["fits"] if fault else None
    result["trial_budget"] = budget
    try:
        result["referee"] = referee(referee_path or path, target, result) if result["completed"] else None
    except Exception as exc:  # noqa: BLE001 — e.g. the final spec cannot be fitted at all
        result["referee"] = None
        result["referee_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return result
