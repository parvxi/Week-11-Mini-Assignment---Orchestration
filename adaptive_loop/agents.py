"""The agents. Each one has a single responsibility and returns a state patch.

LLM use is optional (planner, strategist). Everything that decides *what is true*
about the data or the model — the audit, the probe, training, scoring — is code.
"""

from __future__ import annotations

import json
import operator
import tempfile
import warnings
from pathlib import Path
from typing import Annotated, TypedDict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline

from dust_pipeline.schema import PERSISTENCE_COLUMN, TIME_COLUMN
from loop_runtime import (
    FAMILIES,
    _coerce_experiment_spec,
    _repair_heuristic,
    _resolve_model,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)
from student_loop.training import (
    _features,
    _pipeline,
    audit_leakage,
    fit_holdout,
    persistence_benchmark,
    fit_validate,
    scores_at,
    split_frames,
    tune_threshold,
)

NO_SIGNAL_AUC = 0.55
MAX_RETRIES = 2  # retries per trial after the first failed attempt
MAX_REPAIRS = 1  # data repair passes before the run is declared invalid
# Validation holds roughly 100 positive station-days, so F1 moves by about ±0.01 from noise alone.
PLATEAU_DELTA = 0.01  # a refinement must beat the best validation F1 by more than this to count
MIN_ROWS = 200

DEFAULT_HP = {
    "logistic_regression": {"C": 1.0, "max_iter": 500},
    "random_forest": {"n_estimators": 200, "max_depth": 8},
    "hist_gradient_boosting": {"learning_rate": 0.1, "max_iter": 150},
}


class AdaptiveState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    config: dict  # max_trials, target_score, seed, holdout_frac, use_llm, model
    plan: dict  # written by the planner: agents, edges, rationale
    validation: dict  # data_validator output: issues + verdict
    repairs: int  # repair passes done
    repair_log: list
    profile: dict
    audit: dict
    benchmark: dict  # persistence forecast scored on validation
    probe: dict
    budget: int  # trials the supervisor is willing to spend (probe can shrink it)
    spec: dict
    result: dict
    val_scores: dict
    attempt: int  # attempts for the current trial
    fits: int  # total fit calls
    failed_families: list
    trials: list  # one summary per finished trial
    best: dict
    verdict: dict
    holdout: dict
    stop_reason: str
    report: dict
    steps: Annotated[list, operator.add]  # what each agent did
    route_log: Annotated[list, operator.add]  # every supervisor decision + reason


def _df(state: AdaptiveState) -> pd.DataFrame:
    return pd.read_csv(state["dataset_path"])


def _cfg(state: AdaptiveState) -> dict:
    return state.get("config") or {}


def _step(agent: str, note: str, **extra) -> dict:
    return {"agent": agent, "note": note, **extra}


# ---------------------------------------------------------------------------
def data_validator(state: AdaptiveState) -> dict:
    """Decide whether the table can be modelled: ok, repairable, or fatally invalid."""
    target = state["target_column"]
    try:
        df = _df(state)
    except Exception as exc:  # noqa: BLE001
        v = {"status": "invalid", "fatal": [f"cannot read dataset: {exc}"], "repairable": []}
        return {"validation": v, "steps": [_step("data_validator", f"INVALID — {v['fatal'][0]}", output=v)]}

    fatal, repairable = [], []
    if target not in df.columns:
        fatal.append(f"target column {target!r} is missing")
    else:
        y = pd.to_numeric(df[target], errors="coerce")
        n_null = int(y.isna().sum())
        if n_null:
            repairable.append({"issue": "null_target", "rows": n_null})
        classes = sorted(y.dropna().unique().tolist())
        if len(classes) < 2:
            fatal.append(f"target has a single class {classes} — nothing to learn")
        elif not set(classes) <= {0, 1}:
            fatal.append(f"target is not binary 0/1 (values {classes[:5]})")
        elif int((y == 1).sum()) < 20:
            fatal.append(f"only {int((y == 1).sum())} positive rows")
    if len(df) < MIN_ROWS:
        fatal.append(f"only {len(df)} rows (need ≥ {MIN_ROWS})")
    n_dup = int(df.duplicated().sum())
    if n_dup:
        repairable.append({"issue": "duplicate_rows", "rows": n_dup})
    numeric = df.select_dtypes("number")
    n_inf = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    if n_inf:
        repairable.append({"issue": "infinite_values", "cells": n_inf})
    empty = [c for c in df.columns if c != target and df[c].isna().all()]
    if empty:
        repairable.append({"issue": "empty_columns", "columns": empty})

    status = "invalid" if fatal else ("needs_repair" if repairable else "ok")
    v = {"status": status, "fatal": fatal, "repairable": repairable, "rows": int(len(df)), "columns": int(df.shape[1])}
    note = {"ok": f"ok — {len(df)} rows, {df.shape[1]} columns",
            "needs_repair": "needs repair — " + ", ".join(f"{r['issue']}" for r in repairable),
            "invalid": "INVALID — " + "; ".join(fatal)}[status]
    return {"validation": v, "steps": [_step("data_validator", note, output=v)]}


def data_repairer(state: AdaptiveState) -> dict:
    """Apply the safe repairs the validator listed and hand back a repaired copy of the table."""
    df, target = _df(state), state["target_column"]
    log = []
    for item in (state.get("validation") or {}).get("repairable") or []:
        before = len(df)
        if item["issue"] == "null_target":
            df = df[pd.to_numeric(df[target], errors="coerce").notna()]
            log.append(f"dropped {before - len(df)} rows with a null target")
        elif item["issue"] == "duplicate_rows":
            df = df.drop_duplicates()
            log.append(f"dropped {before - len(df)} duplicate rows")
        elif item["issue"] == "infinite_values":
            num = df.select_dtypes("number").columns
            df[num] = df[num].replace([np.inf, -np.inf], np.nan)
            log.append(f"replaced {item['cells']} infinite values with missing")
        elif item["issue"] == "empty_columns":
            df = df.drop(columns=item["columns"])
            log.append(f"dropped empty columns {item['columns']}")
    df[target] = pd.to_numeric(df[target]).astype(int)
    out = Path(tempfile.mkdtemp(prefix="repaired_")) / "repaired.csv"
    df.to_csv(out, index=False)
    return {
        "dataset_path": str(out),
        "repairs": int(state.get("repairs") or 0) + 1,
        "repair_log": [*(state.get("repair_log") or []), *log],
        "steps": [_step("data_repairer", "; ".join(log) or "nothing to repair", output={"repairs": log})],
    }


def benchmark(state: AdaptiveState) -> dict:
    """Score the no-model persistence forecast on validation: the floor a model must beat."""
    cfg = _cfg(state)
    bench = persistence_benchmark(_df(state), state["target_column"], cfg["seed"], cfg["holdout_frac"])
    note = f"persistence val F1 {bench['f1']:.3f}" if bench.get("available") else f"unavailable: {bench.get('reason')}"
    return {"benchmark": bench, "steps": [_step("benchmark", note, output=bench)]}


def profiler(state: AdaptiveState) -> dict:
    """Describe the table. No modelling decisions."""
    df, target = _df(state), state["target_column"]
    y = df[target]
    rate = float(y.mean()) if pd.api.types.is_numeric_dtype(y) else None
    profile = {
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "target": target,
        "positive_rate": rate,
        "minority_rate": min(rate, 1 - rate) if rate is not None else None,
        "n_numeric": int(df.drop(columns=[target]).select_dtypes("number").shape[1]),
        "n_categorical": int(df.drop(columns=[target]).select_dtypes(exclude="number").shape[1]),
        "missing_cells": int(df.isna().sum().sum()),
    }
    return {
        "profile": profile,
        "steps": [_step("profiler", f"{profile['n_rows']} rows · positive rate {rate:.3f}", output=profile)],
    }


def leakage_auditor(state: AdaptiveState) -> dict:
    """Measure every column; own the list of columns training may not use."""
    cfg = _cfg(state)
    audit = audit_leakage(_df(state), state["target_column"], cfg["seed"], cfg["holdout_frac"])
    leaks = audit["leak_columns"]
    note = f"{len(audit['drop_columns'])} blocked" + (f" · leak: {', '.join(leaks)}" if leaks else " · no leak")
    return {"audit": audit, "steps": [_step("leakage_auditor", note, output=audit)]}


def signal_probe(state: AdaptiveState) -> dict:
    """Cheap 3-fold check: is there anything to learn on the leak-free features?

    The supervisor uses the answer to size the search budget.
    """
    cfg = _cfg(state)
    df, target = _df(state), state["target_column"]
    train, val, _ = split_frames(df, target, cfg["seed"], cfg["holdout_frac"])
    dev = pd.concat([train, val])
    X = _features(dev, target, (state.get("audit") or {}).get("drop_columns") or [])
    pipe: Pipeline = _pipeline(X, {"model_family": "logistic_regression", "hyperparameters": DEFAULT_HP["logistic_regression"]})
    pipe.set_params(clf=LogisticRegression(max_iter=500, class_weight="balanced"))
    max_trials = int(cfg["max_trials"])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if TIME_COLUMN in dev.columns:  # forecasting data: always validate on later days
                order = dev[TIME_COLUMN].argsort(kind="stable")
                X, y, cv = X.iloc[order], dev[target].iloc[order], TimeSeriesSplit(n_splits=3)
            else:
                y, cv = dev[target], 3
            folds = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")  # a crashed fold scores NaN
    except Exception:  # noqa: BLE001 — sklearn raises when every fold crashes
        folds = np.array([np.nan])
    if np.isnan(folds).all():
        # The probe itself failed. Do not conclude "no signal" from a broken measurement.
        probe = {"cv_roc_auc": None, "has_signal": None, "budget": max_trials, "error": "all probe fits failed"}
        note = "probe failed → inconclusive, keeping the full budget"
        return {"probe": probe, "budget": max_trials, "steps": [_step("signal_probe", note, output=probe)]}
    auc = float(np.nanmean(folds))
    has_signal = auc >= NO_SIGNAL_AUC
    budget = max_trials if has_signal else 1
    probe = {"cv_roc_auc": round(auc, 4), "has_signal": has_signal, "budget": budget}
    note = (
        f"3-fold ROC-AUC {auc:.3f} → full search ({budget} trials)"
        if has_signal
        else f"3-fold ROC-AUC {auc:.3f} is chance level → one confirmatory trial instead of {max_trials}"
    )
    return {"probe": probe, "budget": budget, "steps": [_step("signal_probe", note, output=probe)]}


def strategist(state: AdaptiveState) -> dict:
    """Choose the next model family + hyperparameters from what has been tried."""
    cfg = _cfg(state)
    trials = list(state.get("trials") or [])
    failed = set(state.get("failed_families") or [])
    tried = [t["model_family"] for t in trials]
    drops = list((state.get("audit") or {}).get("drop_columns") or [])
    eid = f"trial-{len(trials) + 1}"

    untried = [f for f in FAMILIES if f not in tried and f not in failed]
    if untried:
        family, why = untried[0], "next family not tried yet"
        hp = dict(DEFAULT_HP[family])
    else:
        ok = [t for t in trials if t.get("status") == "succeeded"] or trials
        best = max(ok, key=lambda t: t.get("f1") or 0) if ok else {"model_family": "logistic_regression"}
        family, why = best["model_family"], "all families tried — more capacity for the best one"
        hp = dict(DEFAULT_HP[family])
        k = tried.count(family)  # how many times this family has been tried → next capacity step
        if family == "random_forest":
            hp.update({"n_estimators": 200 + 200 * k, "max_depth": 8 + 6 * k})
        elif family == "hist_gradient_boosting":
            hp.update({"learning_rate": round(0.1 / (k + 1), 3), "max_iter": 150 + 150 * k})
        else:
            hp.update({"C": float(4**k)})
    spec = {"experiment_id": eid, "model_family": family, "hyperparameters": hp, "rationale": why}
    used_llm = False

    model = _resolve_model(cfg)
    if cfg.get("use_llm") and llm_configured(model):
        system = (
            "You are the Strategist in a multi-agent ML system. Pick the next experiment. "
            'Return ONLY JSON: {"model_family": "...", "hyperparameters": {...}, "rationale": "one sentence"}. '
            f"model_family must be one of {list(FAMILIES)} and must not be in unavailable_families."
        )
        user = (
            f"profile={json.dumps(state.get('profile'), default=str)}\n"
            f"probe={json.dumps(state.get('probe'), default=str)}\n"
            f"trials_so_far={json.dumps([{k: t.get(k) for k in ('model_family', 'status', 'f1', 'roc_auc')} for t in trials])}\n"
            f"unavailable_families={sorted(failed)}\n"
            f"last_verdict={json.dumps(state.get('verdict'), default=str)}\n"
        )
        content, _err = openrouter_chat(system, user, model=model, max_tokens=400, temperature=0.2, json_object=True)
        coerced = _coerce_experiment_spec(extract_json_object(content or ""), experiment_id=eid)
        if coerced and coerced["model_family"] not in failed:
            spec, used_llm = {**coerced, "experiment_id": eid}, True

    spec["drop_columns"] = drops  # the feature set is the auditor's decision
    return {
        "spec": spec,
        "attempt": 0,
        "steps": [_step("strategist", f"{spec['model_family']} — {spec.get('rationale')}", used_llm=used_llm, output=spec)],
    }


def trainer(state: AdaptiveState) -> dict:
    """Fit the spec and score validation. One attempt per call; the supervisor decides on retries."""
    cfg = _cfg(state)
    spec = dict(state.get("spec") or {})
    attempt = int(state.get("attempt") or 0) + 1
    fits = int(state.get("fits") or 0) + 1
    if attempt > 1:  # a retry: fall back to known-safe hyperparameters for this family
        drops = spec.get("drop_columns") or []
        spec = {**_repair_heuristic(spec, "retry"), "experiment_id": spec.get("experiment_id"), "drop_columns": drops}

    try:
        result = fit_validate(_df(state), state["target_column"], spec, cfg["seed"], cfg["holdout_frac"])
    except Exception as exc:  # noqa: BLE001
        result = {"status": "failed", "error": str(exc), "validation_metrics": {}}

    val_scores = result.pop("val_scores", {})
    ok = result["status"] == "succeeded"
    m = result.get("validation_metrics") or {}
    note = (
        f"attempt {attempt}: {spec.get('model_family')} · f1@0.5={m.get('f1', 0):.3f} · auc={m.get('roc_auc') or 0:.3f}"
        if ok
        else f"attempt {attempt} FAILED: {result.get('error')}"
    )
    return {
        "spec": spec,
        "result": result,
        "val_scores": val_scores,
        "attempt": attempt,
        "fits": fits,
        "steps": [_step("trainer", note, status=result["status"], attempt=attempt)],
    }


def threshold_tuner(state: AdaptiveState) -> dict:
    """Re-score validation at the F1-optimal decision threshold."""
    result = dict(state.get("result") or {})
    scores = state.get("val_scores") or {}
    before = dict(result.get("validation_metrics") or {})
    tuned = scores_at(scores["y_true"], scores["proba"], tune_threshold(scores["y_true"], scores["proba"])["threshold"])
    result["validation_metrics"] = tuned
    note = f"threshold 0.50 → {tuned['threshold']:.3f} · f1 {before.get('f1', 0):.3f} → {tuned['f1']:.3f}"
    return {"result": result, "val_scores": {}, "steps": [_step("threshold_tuner", note)]}


def critic(state: AdaptiveState) -> dict:
    """Close the trial: record it, update the best model, give a verdict."""
    cfg = _cfg(state)
    spec, result = state.get("spec") or {}, state.get("result") or {}
    m = result.get("validation_metrics") or {}
    ok = result.get("status") == "succeeded"
    trials = [
        *(state.get("trials") or []),
        {
            "experiment_id": spec.get("experiment_id"),
            "model_family": spec.get("model_family"),
            "status": result.get("status"),
            "attempts": int(state.get("attempt") or 1),
            "f1": m.get("f1"),
            "roc_auc": m.get("roc_auc"),
            "threshold": m.get("threshold"),
            "error": result.get("error"),
        },
    ]
    patch: dict = {"trials": trials}
    best = dict(state.get("best") or {})
    if ok and (m.get("f1") or 0) > (best.get("f1") if best.get("f1") is not None else -1):
        best = {"spec": spec, "f1": m["f1"], "roc_auc": m.get("roc_auc"), "threshold": m.get("threshold", 0.5)}
        patch["best"] = best
    if not ok:
        patch["failed_families"] = [*(state.get("failed_families") or []), spec.get("model_family")]

    target = float(cfg["target_score"])
    budget = int(state.get("budget") or cfg["max_trials"])
    if ok and m["f1"] >= target:
        verdict = {"action": "accept", "reason": f"val F1 {m['f1']:.3f} ≥ target {target:.2f}"}
    elif _plateaued(trials):
        verdict = {"action": "stop", "reason": "plateau"}
    elif len(trials) >= budget:
        probe = state.get("probe") or {}
        reason = "no_signal" if probe.get("has_signal") is False else "budget_exhausted"
        verdict = {"action": "stop", "reason": reason}
    elif not ok:
        verdict = {"action": "revise", "reason": f"{spec.get('model_family')} failed after {state.get('attempt')} attempts — switch family"}
    else:
        verdict = {"action": "revise", "reason": f"val F1 {m['f1']:.3f} < target {target:.2f}"}
    bench = state.get("benchmark") or {}
    if ok and bench.get("available"):
        verdict["skill_over_persistence"] = round(m["f1"] - bench["f1"], 4)
    patch["verdict"] = verdict
    skill = f" · skill vs persistence {verdict['skill_over_persistence']:+.3f}" if "skill_over_persistence" in verdict else ""
    patch["steps"] = [_step("critic", f"{verdict['action']} — {verdict['reason']}{skill}", verdict=verdict)]
    return patch


def _plateaued(trials: list) -> bool:
    """True once every model family has been tried and a further refinement brought no real gain."""
    ok = [t for t in trials if t.get("status") == "succeeded"]
    if len({t["model_family"] for t in ok}) < len(FAMILIES) or len(ok) < len(FAMILIES) + 1:
        return False
    best_before = max(t["f1"] or 0 for t in ok[:-1])
    return (ok[-1]["f1"] or 0) <= best_before + PLATEAU_DELTA


def evaluator(state: AdaptiveState) -> dict:
    """Score the best model once on the sealed holdout."""
    cfg = _cfg(state)
    best = state.get("best") or {}
    metrics = fit_holdout(
        _df(state), state["target_column"], best["spec"], float(best.get("threshold") or 0.5), cfg["seed"], cfg["holdout_frac"]
    )
    metrics["model_family"] = best["spec"].get("model_family")
    note = f"holdout f1={metrics['f1']:.3f} · auc={metrics.get('roc_auc') or 0:.3f}"
    return {"holdout": metrics, "steps": [_step("evaluator", note, output=metrics)]}


def reporter(state: AdaptiveState) -> dict:
    """Summarise the run, including how the agents were coordinated."""
    verdict = state.get("verdict") or {}
    trials = state.get("trials") or []
    validation = state.get("validation") or {}
    if validation.get("status") == "invalid":
        stop_reason = "invalid_data"
    elif not state.get("best"):
        stop_reason = "training_failed"
    elif verdict.get("action") == "accept":
        stop_reason = "target_met"
    else:
        stop_reason = verdict.get("reason") or "budget_exhausted"
    steps = state.get("steps") or []
    ran = [s["agent"] for s in steps] + ["reporter"]
    planned = (state.get("plan") or {}).get("agents") or []
    report = {
        "stop_reason": stop_reason,
        "trials": len(trials),
        "trial_budget_requested": int(_cfg(state)["max_trials"]),
        "retries": sum(max(0, int(t.get("attempts") or 1) - 1) for t in trials),
        "agents_planned": planned,
        "agents_run": list(dict.fromkeys(ran)),
        "agents_planned_but_skipped": [a for a in planned if a not in ran],
        "agent_steps": len(ran),
        "leak_columns": (state.get("audit") or {}).get("leak_columns"),
        "data_validation": validation.get("status"),
        "data_problems": validation.get("fatal") or None,
        "repairs": state.get("repair_log") or [],
        "persistence_benchmark": state.get("benchmark"),
        "best": state.get("best"),
        "holdout": state.get("holdout"),
    }
    return {"report": report, "stop_reason": stop_reason, "steps": [_step("reporter", f"stop reason: {stop_reason}")]}


# Registry the planner chooses from: name → (function, one-line responsibility, required?)
REGISTRY = {
    "data_validator": (data_validator, "check the table can be modelled: ok / repairable / invalid", True),
    "data_repairer": (data_repairer, "apply safe repairs (duplicates, null targets, infinities) and re-validate", True),
    "profiler": (profiler, "describe the table (size, class balance, column types)", True),
    "leakage_auditor": (leakage_auditor, "measure every column and block leaks / ids / raw dates", True),
    "signal_probe": (signal_probe, "cheap cross-validated check for learnable signal; sizes the search budget", False),
    "benchmark": (benchmark, "score the persistence forecast (tomorrow = today) as the floor to beat", False),
    "strategist": (strategist, "choose the next model family and hyperparameters", True),
    "trainer": (trainer, "fit one spec and score validation; retried by the supervisor on failure", True),
    "threshold_tuner": (threshold_tuner, "tune the decision threshold when classes are imbalanced", False),
    "critic": (critic, "record the trial, track the best model, accept / revise / stop", True),
    "evaluator": (evaluator, "score the best model once on the sealed holdout", True),
    "reporter": (reporter, "final report including the coordination trace", True),
}
