"""Test cases. Each one states its input, expected behaviour, scoring rule and pass threshold.

Two kinds:
  outcome       any system can pass — judged on what came out (features used, forecast quality,
                trials spent, surviving bad data or a crashed training job)
  capability    judged on an intermediate output or a coordination decision that a system must
                expose to be scored at all; a system without that output scores 0

`score(run)` returns (score in [0, 1], short evidence string). `run` maps a run key to the
result of executing the system on that case's input.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from dust_pipeline.schema import OUTCOME_COLUMNS


@dataclass
class Case:
    id: str
    kind: str  # outcome | capability
    measures: str  # intermediate | final | coordination
    title: str
    runs: dict  # run key → {"dataset": name, "fault": dict | None, "budget": int, "referee_dataset": name}
    expected: str
    scoring: str
    threshold: float
    score: Callable[[dict], tuple[float, str]] = field(repr=False, default=None)


def _auc(r: dict) -> float | None:
    return (r.get("referee") or {}).get("roc_auc")


def _leaked(r: dict) -> list[str]:
    cols = r.get("excluded_columns") or []
    return [c for c in OUTCOME_COLUMNS if c not in cols]


def _all_excluded(r: dict, columns: list[str]) -> tuple[float, str]:
    cols = r.get("excluded_columns")
    if not r["completed"] or cols is None:
        return 0.0, r.get("error") or "no final model"
    used = [c for c in columns if c not in cols]
    return (0.0, f"trained on {', '.join(used)}") if used else (1.0, f"{', '.join(columns)} excluded")


def _kept(r: dict, column: str) -> tuple[float, str]:
    cols = r.get("excluded_columns")
    if not r["completed"] or cols is None:
        return 0.0, r.get("error") or "no final model"
    return (0.0, f"{column} wrongly blocked") if column in cols else (1.0, f"{column} kept")


def _skill(r: dict) -> tuple[float, str]:
    ref = r.get("referee")
    if not ref:
        return 0.0, r.get("error") or r.get("referee_error") or "no final model"
    ratio = ref["f1"] / ref["persistence_f1"]
    return min(ratio, 1.5) / 1.5 * 1.5, (
        f"referee F1 {ref['f1']:.3f} vs persistence {ref['persistence_f1']:.3f} "
        f"({(r['final_spec'] or {}).get('model_family')}, threshold {ref['threshold']:.2f})"
    )


def _ranking(r: dict) -> tuple[float, str]:
    ref = r.get("referee")
    if not ref:
        return 0.0, r.get("error") or r.get("referee_error") or "no final model"
    return float(ref["roc_auc"]), f"referee ROC-AUC {ref['roc_auc']:.3f}, PR-AUC {ref['pr_auc']:.3f}"


def _report_matches(r: dict) -> tuple[float, str]:
    ref, rep = r.get("referee"), r.get("reported_metrics")
    if not ref or not rep or rep.get("f1") is None:
        return 0.0, r.get("error") or "no reported metrics to check"
    diff = abs(float(rep["f1"]) - ref["f1"])
    return (1.0 if diff <= 0.01 else 0.0), f"system reported F1 {float(rep['f1']):.3f}, referee measured {ref['f1']:.3f} on future days"


def _efficiency(r: dict) -> tuple[float, str]:
    if not r["completed"]:
        return 0.0, r.get("error") or "did not complete"
    score = 1 - (r["trials"] - 1) / (r["trial_budget"] - 1)
    return max(0.0, score), f"{r['trials']} of {r['trial_budget']} trials spent"


def _stops_early(r: dict) -> tuple[float, str]:
    if not r["completed"]:
        return 0.0, r.get("error") or "did not complete"
    return (1.0 if r["trials"] < r["trial_budget"] else 0.0), f"{r['trials']} of {r['trial_budget']} trials spent · stop reason: {r.get('stop_reason')}"


def _says_no_signal(r: dict) -> tuple[float, str]:
    return (1.0 if r.get("stop_reason") == "no_signal" else 0.0), f"stop reason: {r.get('stop_reason')}"


def _probe(run: dict) -> tuple[float, str]:
    a, b = run["no_signal"].get("probe"), run["signal"].get("probe")
    if not a or not b:
        return 0.0, "system has no signal probe output"
    ok = [a.get("has_signal") is False, b.get("has_signal") is True]
    return sum(ok) / 2, f"shuffled-target table → ROC-AUC {a.get('cv_roc_auc')}, real table → {b.get('cv_roc_auc')}"


def _selection(run: dict) -> tuple[float, str]:
    real, bal = run["imbalanced_forecast"], run["balanced_fixture"]
    if real.get("agents_planned") is None:
        return 0.0, "system does not select agents (same steps for every input)"
    ok = [
        "threshold_tuner" in (real["agents_run"] or []),
        "benchmark" in (real["agents_run"] or []),
        "threshold_tuner" not in (bal["agents_planned"] or []),
        "benchmark" not in (bal["agents_planned"] or []),
    ]
    return sum(ok) / 4, (
        f"real table: tuner ran {ok[0]}, benchmark ran {ok[1]} · balanced fixture: tuner left out {ok[2]}, benchmark left out {ok[3]}"
    )


def _plan_valid(run: dict) -> tuple[float, str]:
    r = run["main"]
    plan = r.get("plan")
    if not plan:
        return 0.0, "system has no explicit workflow plan"
    from adaptive_loop.graph import validate_plan

    problems = validate_plan(plan["agents"])
    # The planner runs before the graph exists, so it is not a node of its own plan.
    stray = [a for a in (r.get("agents_run") or []) if a != "planner" and a not in plan["agents"]]
    edges = {(a, b) for a, b in plan["edges"]}
    bad_routes = [x for x in (r.get("route_log") or []) if x["next"] != "END" and (x["after"], x["next"]) not in edges]
    ok = not problems and not stray and not bad_routes
    return (1.0 if ok else 0.0), f"{len(plan['agents'])} agents, {len(plan['edges'])} edges, {len(r.get('route_log') or [])} routing decisions, all on planned edges: {ok}"


def _survives(r: dict, floor: float) -> tuple[float, str]:
    auc = _auc(r)
    if not r["completed"] or auc is None:
        return 0.0, r.get("error") or r.get("referee_error") or "did not complete"
    return (1.0 if auc >= floor else 0.0), f"completed, {r['trials']} trial(s), referee ROC-AUC {auc:.3f}"


def _fails_honestly(r: dict, word: str) -> tuple[float, str]:
    if r.get("error"):
        return 0.0, f"crashed: {r['error'][:90]}"
    honest = word in str(r.get("stop_reason") or "") and not r.get("reported_metrics")
    return (1.0 if honest else 0.0), f"stop reason: {r.get('stop_reason')} · metrics reported: {bool(r.get('reported_metrics'))}"


def _repair_route(run: dict) -> tuple[float, str]:
    r = run["main"]
    if r.get("route_log") is None:
        return 0.0, "system exposes no routing decisions"
    path = [(x["after"], x["next"]) for x in r["route_log"]]
    ok = [("data_validator", "data_repairer") in path, ("data_repairer", "data_validator") in path,
          ("data_validator", "profiler") in path, bool(r.get("repairs"))]
    return sum(ok) / 4, "validator → repairer → validator → profiler; repairs: " + "; ".join(r.get("repairs") or ["none"])


def _retry_bounded(run: dict) -> tuple[float, str]:
    from adaptive_loop.agents import MAX_RETRIES

    r = run["main"]
    log = r.get("trial_log")
    if log is None:
        return 0.0, "system exposes no per-trial attempt log"
    attempts = [int(t.get("attempts") or 0) for t in log]
    ok = bool(attempts) and all(a == MAX_RETRIES + 1 for a in attempts) and len(attempts) <= r["trial_budget"]
    return (1.0 if ok else 0.0), f"attempts per trial {attempts} (limit {MAX_RETRIES + 1}), {len(attempts)} trials of {r['trial_budget']} allowed"


QUALITY_FLOOR = 0.70  # ROC-AUC a system must keep after recovering from bad data or a fault
READY = "saudi_model_ready"

CASES: list[Case] = [
    # ---- intermediate output quality -------------------------------------------------
    Case("outcome_columns_excluded", "outcome", "intermediate", "Tomorrow's outcome columns are kept out of the model",
         {"main": {"dataset": "saudi_real"}},
         "next_day_min_visibility_m and next_day_low_vis_hours are not features of the final model",
         "1 if both columns are excluded from the final model's features, else 0", 1.0,
         lambda run: _all_excluded(run["main"], OUTCOME_COLUMNS)),
    Case("leak_disguised", "outcome", "intermediate", "A leak under an unfamiliar name is caught by measurement",
         {"main": {"dataset": "synthetic_disguised_leak"}},
         "risk_engine_score (derived from the label) is not a feature of the final model",
         "1 if the column is excluded, else 0", 1.0,
         lambda run: _all_excluded(run["main"], ["risk_engine_score"])),
    Case("no_false_block", "outcome", "intermediate", "A strong but honest feature is not thrown away",
         {"main": {"dataset": "synthetic_honest_strong_feature"}},
         "prior_chargebacks stays in the feature set",
         "1 if the column is still a feature of the final model, else 0", 1.0,
         lambda run: _kept(run["main"], "prior_chargebacks")),
    Case("probe_accuracy", "capability", "intermediate", "Signal probe is right in both directions",
         {"no_signal": {"dataset": "saudi_shuffled_target"}, "signal": {"dataset": READY}},
         "Probe says no signal on the shuffled-target control and signal on the real table",
         "fraction of the two tables classified correctly", 1.0, _probe),
    # ---- final output quality --------------------------------------------------------
    Case("forecast_skill", "outcome", "final", "Forecast is competitive with persistence on future days",
         {"main": {"dataset": READY}},
         "On the most recent 20% of days the final model's F1 is at least 90% of the persistence forecast's F1",
         "referee F1 ÷ persistence F1 on the shared chronological holdout", 0.90,
         lambda run: _skill(run["main"])),
    Case("ranking_quality", "outcome", "final", "Model ranks risky days well above chance",
         {"main": {"dataset": READY}},
         "ROC-AUC on the chronological holdout clearly above 0.5",
         "referee ROC-AUC on the shared chronological holdout", 0.75,
         lambda run: _ranking(run["main"])),
    Case("report_matches_referee", "outcome", "final", "The score the system reports is the score it gets on future days",
         {"main": {"dataset": READY}},
         "The F1 in the system's own final report agrees with the referee's measurement",
         "1 if |reported F1 − referee F1| ≤ 0.01, else 0", 1.0,
         lambda run: _report_matches(run["main"])),
    Case("reports_no_signal", "capability", "final", "States the right conclusion on the no-signal control",
         {"main": {"dataset": "saudi_shuffled_target"}},
         "The run ends with stop reason no_signal rather than only 'budget used up'",
         "1 if stop_reason == no_signal, else 0", 1.0,
         lambda run: _says_no_signal(run["main"])),
    # ---- coordination: data problems -------------------------------------------------
    Case("dirty_data_recovered", "outcome", "coordination", "Recovers from repairable data problems",
         {"main": {"dataset": "saudi_dirty", "referee_dataset": READY}},
         "Given null targets, infinite values and duplicated rows, the system still finishes with a working model",
         f"1 if it completes and referee ROC-AUC (on the clean table) ≥ {QUALITY_FLOOR}, else 0", 1.0,
         lambda run: _survives(run["main"], QUALITY_FLOOR)),
    Case("repair_routing", "capability", "coordination", "Invalid rows are routed to repair, then re-validated",
         {"main": {"dataset": "saudi_dirty", "referee_dataset": READY}},
         "Route is validator → repairer → validator → profiler, and the repairs are logged",
         "fraction of the four expected facts observed", 1.0, _repair_route),
    Case("invalid_data_stops", "outcome", "coordination", "Stops cleanly on data that cannot be modelled",
         {"main": {"dataset": "saudi_single_class"}},
         "Target has one class; the system ends without an exception, says the data is invalid, reports no metrics",
         "1 if no exception escapes, the stop reason says invalid, and no metrics are reported; else 0", 1.0,
         lambda run: _fails_honestly(run["main"], "invalid")),
    # ---- coordination: budget --------------------------------------------------------
    Case("efficiency_no_signal", "outcome", "coordination", "Does not burn the budget when there is nothing to learn",
         {"main": {"dataset": "saudi_shuffled_target"}},
         "At most 2 of the 3 allowed trials are spent on the shuffled-target control",
         "1 − (trials − 1) / (budget − 1)", 0.5,
         lambda run: _efficiency(run["main"])),
    Case("plateau_stop", "outcome", "coordination", "Stops when further trials stop improving",
         {"main": {"dataset": READY, "budget": 6}},
         "With 6 trials allowed and an unreachable target, the system stops before using all 6",
         "1 if it completes with fewer trials than the budget, else 0", 1.0,
         lambda run: _stops_early(run["main"])),
    # ---- coordination: training failures ---------------------------------------------
    Case("fault_transient", "outcome", "coordination", "Recovers from a transient training outage",
         {"main": {"dataset": READY, "fault": {"first_n": 4}}},
         "The first 4 model fits crash; the system still finishes with a working model",
         f"1 if it completes and referee ROC-AUC ≥ {QUALITY_FLOOR}, else 0", 1.0,
         lambda run: _survives(run["main"], QUALITY_FLOOR)),
    Case("fault_family_outage", "outcome", "coordination", "Routes around a model family that always fails",
         {"main": {"dataset": READY, "fault": {"estimator": "LogisticRegression"}}},
         "Every logistic-regression fit crashes; the system switches family and finishes with a working model",
         f"1 if it completes and referee ROC-AUC ≥ {QUALITY_FLOOR}, else 0", 1.0,
         lambda run: _survives(run["main"], QUALITY_FLOOR)),
    Case("fault_total_outage", "outcome", "coordination", "Fails honestly when nothing can be trained",
         {"main": {"dataset": READY, "fault": {"all": True}}},
         "Every fit crashes; the system ends cleanly, says training failed, and reports no metrics",
         "1 if no exception escapes, the stop reason says failed, and no metrics are reported; else 0", 1.0,
         lambda run: _fails_honestly(run["main"], "fail")),
    Case("retry_limit", "capability", "coordination", "Retries are bounded",
         {"main": {"dataset": READY, "fault": {"all": True}}},
         "Under a total outage each trial is attempted exactly 3 times (1 + 2 retries) and trials stay within budget",
         "1 if every trial shows exactly the retry limit and the trial count ≤ budget, else 0", 1.0, _retry_bounded),
    # ---- coordination: graph ---------------------------------------------------------
    Case("agent_selection", "capability", "coordination", "Selects agents to fit the input",
         {"imbalanced_forecast": {"dataset": READY}, "balanced_fixture": {"dataset": "synthetic_balanced"}},
         "Threshold tuner and persistence benchmark run on the real table; both are left out of the plan for a balanced table with no persistence column",
         "fraction of the four selection decisions that are correct", 1.0, _selection),
    Case("plan_integrity", "capability", "coordination", "Planned graph is valid and execution stays on it",
         {"main": {"dataset": READY}},
         "The plan has every required agent, all are reachable, and every routing decision follows a planned edge",
         "1 if all three hold, else 0", 1.0, _plan_valid),
]
