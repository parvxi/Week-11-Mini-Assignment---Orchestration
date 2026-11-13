"""Week 11, Part 2 — agent-based ML workflow that forecasts next-day low-visibility (dust)
days at Saudi airports from the station-day table the Airflow DAG loads into Postgres.

Adapted from the in-class research loop (mini_loop/graph.py). What changed:

  + leakage_auditor   NEW      measures every column (single-feature ROC-AUC, id / date
                               checks) and makes its drop list mandatory for the trainer
  + baseline_benchmark NEW     scores the persistence forecast (tomorrow = today) on validation;
                               the judge reports every model as skill over that floor
  + threshold_tuner   NEW      picks the decision threshold on validation; events are ~4% of
                               station-days, so the default 0.5 cut-off is the wrong one
  + training router   NEW      a failed training run skips the tuner and goes to the judge
  ~ ml_engineer       MODIFIED picks family + hyperparameters only; the feature set is fixed
                               by the auditor; returns validation scores for the tuner
  ~ judge             MODIFIED sees F1 / precision / recall / ROC-AUC; code guardrails overrule
                               a false "accept", a premature "reject", and stop the loop
                               once three trials in a row are no better than chance
  ~ memory_writer     MODIFIED also writes the brief for the next cycle
  - context_builder   REMOVED  folded into memory_writer
  ~ holdout_eval      MODIFIED full metric set, confusion matrix, majority-class baseline,
                               and a leakage ablation
  ~ threshold note             the tuner leaves 0.5 alone when validation ROC-AUC is at
                               chance level (tuning a threshold on noise just flags everything)

Flow:

  START → data_scientist → leakage_auditor → baseline_benchmark → hypothesis_generator → ml_engineer
                                                   ↑                    │ ok        │ failed
                                                   │              threshold_tuner   │
                                                   │                    └──→ judge ←┘
                                                   └── memory_writer ←── revise ──┤
                                                                 accept / stop ───┴→ holdout_eval → END

LLM / model-building helpers come from loop_runtime.py (class-provided). Every LLM
node has a deterministic fallback, so the loop also runs with no API key.
"""

from __future__ import annotations

import json
import operator
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
for _p in (REPO, Path("/app")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loop_runtime import (  # noqa: E402
    DEFAULT_MODEL,
    FAMILIES,
    _SPEC_SCHEMA_HINT,
    _coerce_experiment_spec,
    _heuristic_plan,
    _heuristic_profile,
    _heuristic_spec,
    _heuristic_verdict,
    _repair_heuristic,
    _resolve_model,
    _validate_spec,
    extract_json_object,
    llm_configured,
    openrouter_chat,
)
from student_loop.training import (  # noqa: E402
    audit_leakage,
    persistence_benchmark,
    fit_holdout,
    fit_validate,
    scores_at,
    tune_threshold,
)

# Stop early when this many trials in a row are no better than a coin flip.
NO_SIGNAL_AUC = 0.55
NO_SIGNAL_MIN_CYCLES = 3


# ---------------------------------------------------------------------------
# STATE — one shared dict; each node returns a patch with the keys it updates.
# ---------------------------------------------------------------------------
class SimpleLoopState(TypedDict, total=False):
    dataset_path: str
    target_column: str
    target_score: float
    max_cycles: int
    cycle: int
    seed: int
    holdout_frac: float
    model: str  # LLM id from the UI / env
    profile: dict
    audit: dict  # leakage_auditor output
    benchmark: dict  # baseline_benchmark output (persistence forecast on validation)
    brief: dict
    hypothesis: dict
    experiment_spec: dict
    execution_result: dict
    val_scores: dict  # validation labels + probabilities, ml_engineer → threshold_tuner
    verdict: dict
    memory: dict
    best_spec: dict
    best_val_f1: float
    best_threshold: float
    holdout_metrics: dict
    stop_reason: str
    steps: Annotated[list, operator.add]


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------
def node_data_scientist(state: SimpleLoopState) -> dict:
    """Profile CSV via OpenRouter; heuristic fallback if key/parse fails."""
    path = Path(state["dataset_path"])
    target = state["target_column"]
    df = pd.read_csv(path)
    if target not in df.columns:
        return {
            "profile": {},
            "steps": [{"node": "data_scientist", "note": f"target {target!r} missing", "status": "error"}],
        }

    # This workflow is a yes/no classifier scored with F1. Stop early, before any LLM call,
    # when the chosen column is not a 0/1 label (for example a wind speed or an hour count).
    labels = set(pd.unique(df[target].dropna()))
    if not labels or not labels <= {0, 1}:
        raise ValueError(
            f"'{target}' is not a yes/no (0/1) column, so it cannot be the target of this classifier. "
            "Choose 'low_vis_next_day'."
        )

    profile = _heuristic_profile(df, target)
    used_llm = False
    note = "heuristic"
    model = _resolve_model(state)

    if llm_configured(model):
        missing = {c: float(df[c].isna().mean()) for c in df.columns}
        nunique = {c: int(df[c].nunique(dropna=True)) for c in df.columns}
        sample = df.head(6).astype(str).replace("nan", "").to_dict(orient="records")
        system = (
            "You are the Data Scientist only — not the hypothesizer and not the ML engineer. "
            "Your job is dataset understanding: profile the CSV and flag risk. "
            "Do NOT propose model families, hyperparameters, or experiment configs. "
            "Return ONLY one JSON object with keys: "
            "n_rows (int), n_columns (int), target (string), target_rate (number|null), "
            "numeric_features (string[]), categorical_features (string[]), "
            "leakage_suspects (string[]), recommended_drops (string[]), "
            "notes (string[], short bullets about data risks — leakage, imbalance, sparsity). "
            "Flag PassengerId/Name/Ticket/Cabin-style identifiers and near-unique text as leakage. "
            "Do not invent columns that are not present."
        )
        user = (
            f"dataset={path.name}\ntarget={target}\nn_rows={len(df)}\n"
            f"columns={list(df.columns)}\n"
            f"dtypes={{{', '.join(f'{c}:{df[c].dtype}' for c in df.columns)}}}\n"
            f"missing_rate={missing}\nnunique={nunique}\n"
            f"sample_rows={sample}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=900, temperature=0.15)
        if content:
            parsed = extract_json_object(content)
            if parsed and (
                "recommended_drops" in parsed
                or "leakage_suspects" in parsed
                or "numeric_features" in parsed
            ):
                profile = parsed
                profile.setdefault("n_rows", int(len(df)))
                profile.setdefault("n_columns", int(len(df.columns)))
                profile.setdefault("target", target)
                profile.setdefault("recommended_drops", profile.get("leakage_suspects") or [])
                used_llm = True
                note = f"llm profile · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "profile": profile,
        "steps": [
            {
                "node": "data_scientist",
                "note": note,
                "used_llm": used_llm,
                "model": model if used_llm else None,
                "output": profile,
            }
        ],
    }


def node_leakage_auditor(state: SimpleLoopState) -> dict:
    """NEW (code only). Test every column instead of trusting names or the LLM."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    profile = dict(state.get("profile") or {})
    audit = audit_leakage(df, target, int(state.get("seed") or 42), float(state.get("holdout_frac") or 0.2))

    # One agent checks another: the Data Scientist (LLM) only *suspects* columns.
    # A suspect is dropped only if the measurement confirms it; the rest stay in.
    suspects = [
        c
        for c in [*(profile.get("recommended_drops") or []), *(profile.get("leakage_suspects") or [])]
        if c in df.columns and c != target
    ]
    audit["unconfirmed_suspects"] = [c for c in dict.fromkeys(suspects) if c not in audit["drop_columns"]]
    profile["recommended_drops"] = list(audit["drop_columns"])
    profile["leakage_suspects"] = list(audit["leak_columns"])

    leaks = audit["leak_columns"]
    note = (
        f"{len(audit['drop_columns'])} columns blocked"
        + (f" · target leak: {', '.join(leaks)}" if leaks else " · no target leak found")
    )
    return {
        "audit": audit,
        "profile": profile,
        "steps": [{"node": "leakage_auditor", "note": note, "used_llm": False, "output": audit}],
    }


def node_hypothesis(state: SimpleLoopState) -> dict:
    """Conceptual next experiment from DS profile + Judge feedback — no HPT numbers."""
    profile = state.get("profile") or {}
    brief = state.get("brief") or {}
    verdict = state.get("verdict") or {}
    history = list(brief.get("history") or (state.get("memory") or {}).get("history") or [])
    model = _resolve_model(state)

    plan = _heuristic_plan(profile, history, verdict)
    used_llm = False
    note = "heuristic plan"

    if llm_configured(model):
        system = (
            "You are the Hypothesis Generator only — not the Data Scientist and not the ML Engineer. "
            "Read the data-scientist profile and the latest judge verdict / lessons. "
            "Propose ONE conceptual research direction for the next trial. "
            "Do NOT invent concrete hyperparameter values, C/learning_rate/max_depth numbers, "
            "or a full ExperimentSpec — the ML Engineer owns HPT config. "
            "Return ONLY one JSON object with keys: "
            "direction (1-2 sentences: what should change conceptually), "
            "why (cite profile signals and/or judge observation), "
            "focus (one of: baseline_then_iterate|capacity_or_regularization|model_family_switch|feature_drops), "
            "respect_drops (string[] of columns to keep dropped — copy from profile when relevant), "
            "avoid_repeats (string[] of model families or ideas already tried). "
            "Stay conceptual: e.g. 'try a tree ensemble with less regularization' — never 'C=0.5'."
        )
        user = (
            f"profile={json.dumps(profile, default=str)}\n"
            f"latest_verdict={json.dumps(verdict, default=str)}\n"
            f"brief={json.dumps(brief, default=str)}\n"
            f"recent_history={json.dumps(history[-5:], default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.4)
        if content:
            parsed = extract_json_object(content)
            if parsed and parsed.get("direction"):
                plan = {
                    "direction": parsed.get("direction"),
                    "why": parsed.get("why") or plan.get("why"),
                    "focus": parsed.get("focus") or plan.get("focus"),
                    "respect_drops": parsed.get("respect_drops")
                    if isinstance(parsed.get("respect_drops"), list)
                    else plan.get("respect_drops"),
                    "avoid_repeats": parsed.get("avoid_repeats")
                    if isinstance(parsed.get("avoid_repeats"), list)
                    else plan.get("avoid_repeats"),
                }
                used_llm = True
                note = f"live conceptual hypothesis · {model}"
            else:
                note = f"llm unparsable; heuristic ({err or 'ok'})"
        else:
            note = f"{err or 'no llm'}; heuristic"

    return {
        "hypothesis": plan,
        "steps": [
            {
                "node": "hypothesis_generator",
                "note": note,
                "hypothesis": plan,
                "output": plan,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_ml_engineer(state: SimpleLoopState) -> dict:
    """MODIFIED. Hypothesis → concrete spec → train → validation scores.

    The LLM chooses the model family and hyperparameters; the feature set is fixed
    by the auditor's drop list.
    Picking the best model is left to threshold_tuner, after the cut-off is tuned.
    """
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    profile = state.get("profile") or {}
    plan = dict(state.get("hypothesis") or {})
    history = list((state.get("memory") or {}).get("history") or [])
    prev_spec = dict(state.get("experiment_spec") or {})
    mandatory_drops = list((state.get("audit") or {}).get("drop_columns") or [])
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    model_id = _resolve_model(state)
    next_eid = f"exp-{len(history) + 1}"

    # Fallback spec. If the last trial could not rank event days above chance, turning the
    # same model's knobs again is pointless — move to a family not tried yet.
    fallback_plan = dict(plan)
    tried = {h.get("model_family") for h in history}
    if history and (len(tried) < len(FAMILIES) or (history[-1].get("roc_auc") or 0) < NO_SIGNAL_AUC):
        fallback_plan["focus"] = "model_family_switch"  # compare every family before fine-tuning one
    spec = _heuristic_spec(profile, history, fallback_plan)
    used_llm = False
    repaired = False
    note = "heuristic HPT"

    if llm_configured(model_id):
        system = (
            "You are the ML Engineer only — not the Hypothesis Generator and not the Judge. "
            "Translate the conceptual plan into a concrete training config. "
            f"Return ONLY one JSON object matching this schema exactly: {_SPEC_SCHEMA_HINT} "
            "model_family MUST be one of logistic_regression|random_forest|hist_gradient_boosting. "
            "hyperparameters MUST hold concrete numbers for that family "
            "(C/max_iter OR n_estimators/max_depth OR learning_rate/max_iter). "
            "The positive class is rare; a separate agent tunes the decision threshold. "
            "No markdown fences, no commentary."
        )
        user = (
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"blocked_columns={mandatory_drops}\n"
            f"n_rows={profile.get('n_rows')} target_rate={profile.get('target_rate')}\n"
            f"prior_spec={json.dumps(prev_spec, default=str)}\n"
            f"recent_history={json.dumps(history[-3:], default=str)}\n"
            f"allowed_families={list(FAMILIES)}\n"
            f"next_experiment_id={next_eid}\n"
        )
        content, err = openrouter_chat(
            system, user, model=model_id, max_tokens=700, temperature=0.2, json_object=True
        )
        coerced = _coerce_experiment_spec(
            extract_json_object(content or ""), experiment_id=next_eid, default_drops=mandatory_drops
        )
        if coerced:
            spec, used_llm, note = coerced, True, f"live HPT config · {model_id}"
        else:
            note = f"llm unparsable HPT; heuristic ({err or 'ok'})"

    err = _validate_spec(spec)
    if err:
        spec, repaired, note = _repair_heuristic(spec, err), True, f"heuristic repair after: {err}"

    # Guardrail: the feature set is the auditor's decision, not the LLM's. The spec
    # can neither restore a blocked column nor quietly drop a legitimate one.
    spec["experiment_id"] = next_eid
    spec["drop_columns"] = list(mandatory_drops)

    try:
        result = fit_validate(df, target, spec, seed, holdout_frac)
    except Exception as exc:  # noqa: BLE001
        # Self-heal once: fall back to safe hyperparameters for the same family.
        spec = _repair_heuristic(spec, str(exc))
        spec["experiment_id"] = next_eid
        spec["drop_columns"] = list(mandatory_drops)
        repaired = True
        note = f"heuristic repair after run error: {exc}"
        try:
            result = fit_validate(df, target, spec, seed, holdout_frac)
        except Exception as exc2:  # noqa: BLE001
            result = {"status": "failed", "error": str(exc2), "validation_metrics": {}}

    val_scores = result.pop("val_scores", {})
    ok = result.get("status") == "succeeded"
    metrics = result.get("validation_metrics") or {}
    f1 = float(metrics.get("f1") or 0)
    return {
        "execution_result": result,
        "experiment_spec": spec,
        "val_scores": val_scores,
        "cycle": int(state.get("cycle") or 0) + 1,
        "steps": [
            {
                "node": "ml_engineer",
                "note": f"{note} · f1@0.5={f1:.3f} · auc={metrics.get('roc_auc') or 0:.3f}" if ok else note,
                "f1": f1 if ok else None,
                "status": result.get("status"),
                "repaired": repaired,
                "used_llm": used_llm,
                "experiment_id": next_eid,
                "model_family": spec.get("model_family"),
                "validation_metrics": _flat(metrics),
                "duration_seconds": result.get("duration_seconds"),
                "spec": spec,
                "output": {"experiment_spec": spec, "execution_result": {k: v for k, v in result.items()}},
            }
        ],
    }


def node_baseline_benchmark(state: SimpleLoopState) -> dict:
    """NEW (code only). How good is the no-model forecast "tomorrow = today"?"""
    df = pd.read_csv(state["dataset_path"])
    bench = persistence_benchmark(
        df, state["target_column"], int(state.get("seed") or 42), float(state.get("holdout_frac") or 0.2)
    )
    note = (
        f"persistence on validation: F1 {bench['f1']:.3f} (precision {bench['precision']:.3f}, recall {bench['recall']:.3f})"
        if bench.get("available")
        else f"no persistence baseline ({bench.get('reason')})"
    )
    return {"benchmark": bench, "steps": [{"node": "baseline_benchmark", "note": note, "used_llm": False, "output": bench}]}


def route_after_training(state: SimpleLoopState) -> str:
    """NEW router (plain code). Only a successful run has scores worth tuning."""
    return "tune" if (state.get("execution_result") or {}).get("status") == "succeeded" else "skip"


def node_threshold_tuner(state: SimpleLoopState) -> dict:
    """NEW (code only). Re-score validation at the F1-optimal threshold, track the best model."""
    result = dict(state.get("execution_result") or {})
    spec = state.get("experiment_spec") or {}
    scores = state.get("val_scores") or {}
    y_true, proba = scores.get("y_true") or [], scores.get("proba") or []

    before = dict(result.get("validation_metrics") or {})
    auc = before.get("roc_auc") or 0
    if auc < NO_SIGNAL_AUC:
        # The scores do not rank event days above clear days, so any "best" threshold would be
        # fitted to noise (in practice: flag every transaction). Keep the default.
        tuned = dict(before)
        change = f"kept 0.50 — val ROC-AUC {auc:.3f} is chance level, nothing to tune"
    else:
        tuned = scores_at(y_true, proba, tune_threshold(y_true, proba)["threshold"])
        change = (
            f"threshold 0.50 → {tuned['threshold']:.3f} · "
            f"f1 {before.get('f1') or 0:.3f} → {tuned['f1']:.3f}"
        )
    tuned["f1_at_default_threshold"] = before.get("f1")
    result["validation_metrics"] = tuned

    patch: dict = {
        "execution_result": result,
        "val_scores": {},  # no longer needed; keep the state small
        "steps": [
            {
                "node": "threshold_tuner",
                "note": change,
                "used_llm": False,
                "f1": tuned["f1"],
                "experiment_id": spec.get("experiment_id"),
                "validation_metrics": _flat(tuned),
                "output": {"before": before, "after": tuned},
            }
        ],
    }
    if tuned["f1"] > float(state.get("best_val_f1") if state.get("best_val_f1") is not None else -1):
        patch.update({"best_val_f1": tuned["f1"], "best_spec": spec, "best_threshold": tuned["threshold"]})
    return patch


def node_judge(state: SimpleLoopState) -> dict:
    """MODIFIED. LLM verdict on the tuned validation metrics, then code guardrails."""
    result = state.get("execution_result") or {}
    spec = state.get("experiment_spec") or {}
    plan = state.get("hypothesis") or {}
    history = list((state.get("memory") or {}).get("history") or [])
    target_score = float(state.get("target_score") or 0.8)
    cycle = int(state.get("cycle") or 0)
    model = _resolve_model(state)
    metrics = result.get("validation_metrics") or {}
    ok = result.get("status") == "succeeded"
    f1 = float(metrics.get("f1") or 0)
    auc = metrics.get("roc_auc")

    verdict = _heuristic_verdict(result, target_score)
    used_llm = False
    note = "heuristic"
    bench = state.get("benchmark") or {}
    skill = round(f1 - float(bench["f1"]), 4) if ok and bench.get("available") else None

    if llm_configured(model):
        system = (
            "You are the Judge only — not the Hypothesis Generator and not the ML Engineer. "
            "Decide from VALIDATION evidence only; never invent holdout scores. "
            "The positive class is rare, so accuracy is not informative — reason from F1, "
            "precision, recall and ROC-AUC (0.5 = coin flip). "
            "Return ONLY one JSON object with keys: "
            "action (accept|revise|reject), observation (1-2 sentences citing val F1 and ROC-AUC), "
            "next_action (short conceptual instruction for the hypothesis agent — no numbers). "
            "accept only if validation F1 meets or exceeds target_score; "
            "revise when under target but recoverable; reject for hopeless or broken runs."
        )
        user = (
            f"target_score={target_score}\n"
            f"hypothesis_plan={json.dumps(plan, default=str)}\n"
            f"spec={json.dumps(spec, default=str)}\n"
            f"validation_metrics={json.dumps(metrics, default=str)}\n"
            f"persistence_baseline_f1={bench.get('f1')} skill_over_persistence={skill}\n"
            f"status={result.get('status')} error={result.get('error')}\n"
            f"previous_trials={json.dumps([{k: h.get(k) for k in ('model_family', 'f1', 'roc_auc')} for h in history], default=str)}\n"
        )
        content, err = openrouter_chat(system, user, model=model, max_tokens=500, temperature=0.1)
        parsed = extract_json_object(content or "")
        if parsed and parsed.get("action") in {"accept", "revise", "reject"}:
            verdict = {
                "action": parsed["action"],
                "observation": parsed.get("observation") or verdict.get("observation"),
                "next_action": parsed.get("next_action") or verdict.get("next_action"),
            }
            used_llm = True
            note = f"llm verdict · {model}"
        else:
            note = f"{err or 'llm unparsable'}; heuristic"

    # --- Guardrails (code has the last word over the LLM) ---
    guardrail = None
    if verdict["action"] == "accept" and (not ok or f1 < target_score):
        verdict["action"] = "revise"
        guardrail = f"accept overruled: val F1 {f1:.3f} is below target {target_score:.2f}"
    if verdict["action"] == "reject" and ok and cycle < NO_SIGNAL_MIN_CYCLES:
        verdict["action"] = "revise"
        guardrail = f"reject overruled: only {cycle} trial(s) so far, try another model family first"
    aucs = [h.get("roc_auc") for h in history] + [auc if ok else None]
    known = [a for a in aucs if a is not None]
    if (
        verdict["action"] != "accept"
        and cycle >= NO_SIGNAL_MIN_CYCLES
        and len(known) >= NO_SIGNAL_MIN_CYCLES
        and max(known) < NO_SIGNAL_AUC
    ):
        verdict["action"] = "reject"
        guardrail = (
            f"stopping: {len(known)} trials, best validation ROC-AUC {max(known):.3f} — "
            "no model beats chance on the leak-free features, more tuning will not help"
        )
        verdict["next_action"] = "stop and report that the remaining features carry no signal"
    if guardrail:
        verdict["observation"] = f"{verdict.get('observation') or ''} [{guardrail}]".strip()
        note = f"{note} · guardrail"
    if skill is not None:
        verdict["skill_over_persistence"] = skill
        verdict["observation"] = (
            f"{verdict.get('observation') or ''} Skill over persistence (val F1 {bench['f1']:.3f}): {skill:+.3f}."
        ).strip()

    return {
        "verdict": verdict,
        "steps": [
            {
                "node": "judge",
                "note": note,
                "verdict": verdict,
                "guardrail": guardrail,
                "used_llm": used_llm,
                "model": model if used_llm else None,
            }
        ],
    }


def node_memory_writer(state: SimpleLoopState) -> dict:
    """MODIFIED. File this cycle's lesson AND build the next brief (context_builder removed)."""
    mem = dict(state.get("memory") or {"history": [], "findings": [], "lessons": []})
    spec = state.get("experiment_spec") or {}
    metrics = (state.get("execution_result") or {}).get("validation_metrics") or {}
    history = list(mem.get("history") or [])
    history.append(
        {
            "cycle": state.get("cycle"),
            "hypothesis": state.get("hypothesis"),
            "spec": spec,
            "model_family": spec.get("model_family"),
            "f1": metrics.get("f1"),
            "roc_auc": metrics.get("roc_auc"),
            "threshold": metrics.get("threshold"),
            "verdict": state.get("verdict"),
        }
    )
    lessons = [*(mem.get("lessons") or []), str((state.get("verdict") or {}).get("observation") or "")]
    mem.update({"history": history, "lessons": lessons})
    brief = {"history": history, "lessons": lessons[-3:]}
    return {
        "memory": mem,
        "brief": brief,
        "steps": [{"node": "memory_writer", "note": f"lesson {len(lessons)} filed · brief ready"}],
    }


def node_holdout(state: SimpleLoopState) -> dict:
    """MODIFIED. Score the best model once on the sealed holdout, with full metrics."""
    df = pd.read_csv(state["dataset_path"])
    target = state["target_column"]
    spec = dict(state.get("best_spec") or state.get("experiment_spec") or {})
    threshold = float(state.get("best_threshold") or 0.5)
    seed = int(state.get("seed") or 42)
    holdout_frac = float(state.get("holdout_frac") or 0.2)
    audit = state.get("audit") or {}
    if _validate_spec(spec):
        spec = _repair_heuristic(spec, "invalid best_spec")
        spec["drop_columns"] = list(audit.get("drop_columns") or [])

    metrics = fit_holdout(df, target, spec, threshold, seed, holdout_frac)
    metrics["model_family"] = spec.get("model_family")

    # Ablation: what the same model would score if the leaked columns were let back in.
    leaks = [c for c in (audit.get("leak_columns") or []) if c in df.columns]
    if leaks:
        leaky_spec = {**spec, "drop_columns": [c for c in spec.get("drop_columns") or [] if c not in leaks]}
        leaky = fit_holdout(df, target, leaky_spec, 0.5, seed, holdout_frac)
        metrics["leakage_ablation"] = {
            "columns_restored": leaks,
            "f1": leaky["f1"],
            "roc_auc": leaky["roc_auc"],
            "accuracy": leaky["accuracy"],
            "note": "reference only — these columns are derived from the label",
        }

    action = str((state.get("verdict") or {}).get("action") or "")
    stop_reason = {"accept": "target_met", "reject": "no_signal"}.get(action, "budget_exhausted")
    return {
        "holdout_metrics": metrics,
        "stop_reason": stop_reason,
        "steps": [
            {
                "node": "holdout_eval",
                "note": (
                    f"holdout f1={metrics['f1']:.3f} · auc={metrics.get('roc_auc') or 0:.3f} · "
                    f"acc={metrics['accuracy']:.3f}"
                ),
                "metrics": metrics,
                "output": metrics,
            }
        ],
    }


def route_after_judge(state: SimpleLoopState) -> str:
    """Router (plain code): revise loops while budget remains; anything else finalizes."""
    action = str((state.get("verdict") or {}).get("action") or "stop")
    if action == "revise" and int(state.get("cycle") or 0) < int(state.get("max_cycles") or 3):
        return "memory"
    return "holdout"


def _flat(metrics: dict) -> dict:
    """Numeric metrics only (the UI charts expect a flat name → number map)."""
    return {k: v for k, v in (metrics or {}).items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


# ---------------------------------------------------------------------------
# GRAPH WIRING
# ---------------------------------------------------------------------------
def build_graph():
    g = StateGraph(SimpleLoopState)

    g.add_node("data_scientist", node_data_scientist)
    g.add_node("leakage_auditor", node_leakage_auditor)
    g.add_node("baseline_benchmark", node_baseline_benchmark)
    g.add_node("hypothesis_generator", node_hypothesis)
    g.add_node("ml_engineer", node_ml_engineer)
    g.add_node("threshold_tuner", node_threshold_tuner)
    g.add_node("judge", node_judge)
    g.add_node("memory_writer", node_memory_writer)
    g.add_node("holdout_eval", node_holdout)

    g.add_edge(START, "data_scientist")
    g.add_edge("data_scientist", "leakage_auditor")
    g.add_edge("leakage_auditor", "baseline_benchmark")
    g.add_edge("baseline_benchmark", "hypothesis_generator")
    g.add_edge("hypothesis_generator", "ml_engineer")
    g.add_conditional_edges(
        "ml_engineer", route_after_training, {"tune": "threshold_tuner", "skip": "judge"}
    )
    g.add_edge("threshold_tuner", "judge")
    g.add_conditional_edges(
        "judge", route_after_judge, {"memory": "memory_writer", "holdout": "holdout_eval"}
    )
    g.add_edge("memory_writer", "hypothesis_generator")  # the loop
    g.add_edge("holdout_eval", END)
    return g.compile()


REQUIRED_NODES = frozenset(
    {
        "data_scientist",
        "leakage_auditor",
        "baseline_benchmark",
        "hypothesis_generator",
        "ml_engineer",
        "threshold_tuner",
        "judge",
        "memory_writer",
        "holdout_eval",
    }
)


def get_status() -> dict:
    """Nodes and edges read from the compiled LangGraph — the UI draws exactly this."""
    try:
        graph = build_graph().get_graph()
    except Exception as exc:  # noqa: BLE001
        return {"ready": False, "nodes": [], "edges": [], "missing_nodes": sorted(REQUIRED_NODES), "error": str(exc)}
    nodes = [n for n in graph.nodes if n not in {"__start__", "__end__"}]
    edges = [
        {"source": e.source, "target": e.target, "conditional": bool(e.conditional)} for e in graph.edges
    ]
    missing = sorted(REQUIRED_NODES - set(nodes))
    return {
        "ready": not missing,
        "nodes": nodes,
        "edges": edges,
        "missing_nodes": missing,
        "error": f"missing nodes: {', '.join(missing)}" if missing else None,
    }


# ---------------------------------------------------------------------------
# RUNNERS
# ---------------------------------------------------------------------------
def _initial_state(
    *,
    dataset_path: str,
    target_column: str,
    target_score: float,
    max_cycles: int,
    seed: int,
    holdout_frac: float,
    model: str | None = None,
) -> SimpleLoopState:
    return {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "target_score": target_score,
        "max_cycles": max_cycles,
        "cycle": 0,
        "seed": seed,
        "holdout_frac": holdout_frac,
        "best_val_f1": -1.0,
        "steps": [],
        "memory": {"history": [], "findings": [], "lessons": []},
        "brief": {},
        "model": (model or "").strip() or DEFAULT_MODEL,
    }


def _recursion_limit(max_cycles: int) -> int:
    return 10 + max_cycles * 6


def _finalize_payload(final: dict) -> dict:
    return {
        "dataset_path": final.get("dataset_path"),
        "target_column": final.get("target_column"),
        "cycles": final.get("cycle"),
        "best_val_f1": final.get("best_val_f1"),
        "best_threshold": final.get("best_threshold"),
        "best_spec": final.get("best_spec"),
        "holdout_metrics": final.get("holdout_metrics"),
        "audit": final.get("audit"),
        "benchmark": final.get("benchmark"),
        "verdict": final.get("verdict"),
        "stop_reason": final.get("stop_reason"),
        "profile": final.get("profile"),
        "memory": final.get("memory"),
        "model": final.get("model"),
        "steps": final.get("steps") or [],
    }


def run_simple_loop(
    *,
    dataset_path: str,
    target_column: str = "low_vis_next_day",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
) -> dict:
    """Run the whole graph to completion; return a summary dict."""
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    final = build_graph().invoke(initial, config={"recursion_limit": _recursion_limit(max_cycles)})
    return _finalize_payload(final)


def stream_simple_loop(
    *,
    dataset_path: str,
    target_column: str = "low_vis_next_day",
    target_score: float = 0.8,
    max_cycles: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    model: str | None = None,
):
    """Yield UI events as each node finishes (the gateway forwards them as SSE)."""
    app = build_graph()
    initial = _initial_state(
        dataset_path=dataset_path,
        target_column=target_column,
        target_score=target_score,
        max_cycles=max_cycles,
        seed=seed,
        holdout_frac=holdout_frac,
        model=model,
    )
    yield {
        "type": "run_started",
        "dataset_path": initial["dataset_path"],
        "target_column": target_column,
        "model": initial.get("model"),
    }

    merged: dict = dict(initial)
    last_node: str | None = None
    try:
        for update in app.stream(
            initial, stream_mode="updates", config={"recursion_limit": _recursion_limit(max_cycles)}
        ):
            if not isinstance(update, dict):
                continue
            for node_name, patch in update.items():
                # Routers are edges, not nodes, so announce their decision here.
                if last_node == "ml_engineer":
                    yield {
                        "type": "router",
                        "node": "train_router",
                        "status": "completed",
                        "action": str((merged.get("execution_result") or {}).get("status") or "failed"),
                        "route": "tune" if node_name == "threshold_tuner" else "skip",
                    }
                if last_node == "judge" and node_name in {"memory_writer", "holdout_eval"}:
                    yield {
                        "type": "router",
                        "node": "router",
                        "status": "completed",
                        "action": str((merged.get("verdict") or {}).get("action") or "stop"),
                        "route": "memory" if node_name == "memory_writer" else "holdout",
                    }

                yield {"type": "node_started", "node": node_name, "status": "running"}
                if isinstance(patch, dict):
                    steps = list(patch.get("steps") or [])
                    merged.update({k: v for k, v in patch.items() if k != "steps"})
                    if steps:
                        merged["steps"] = list(merged.get("steps") or []) + steps
                    yield {
                        "type": "node_completed",
                        "node": node_name,
                        "status": "completed",
                        "note": steps[-1].get("note") if steps else None,
                        "step": steps[-1] if steps else None,
                        "cycle": merged.get("cycle"),
                        "best_val_f1": merged.get("best_val_f1"),
                        "verdict": merged.get("verdict"),
                    }
                else:
                    yield {"type": "node_completed", "node": node_name, "status": "completed"}
                last_node = node_name
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}
        return

    yield {"type": "run_completed", "result": _finalize_payload(merged)}


if __name__ == "__main__":
    ds = REPO / "datasets" / "airflow" / "saudi_dust_daily_latest.csv"
    print(json.dumps(run_simple_loop(dataset_path=str(ds)), indent=2, default=str))
