"""Planner → workflow graph → supervised execution.

1. The PLANNER agent looks at the task and writes a plan: which agents from the
   registry this dataset needs, plus the edges between them. The plan is data (JSON).
2. `compile_plan` turns that plan into a LangGraph StateGraph — so the graph that
   runs is the one the planner produced, not a hard-coded one.
3. The SUPERVISOR is the routing function behind every edge. After each agent it
   reads the intermediate results and picks the next agent: retry a failed fit,
   skip the tuner when there is nothing to tune, shrink the budget, stop early.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from langgraph.graph import END, START, StateGraph

REPO = Path(__file__).resolve().parent.parent
for _p in (REPO, Path("/app")):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from loop_runtime import DEFAULT_MODEL, extract_json_object, llm_configured, openrouter_chat  # noqa: E402
from adaptive_loop.agents import MAX_REPAIRS, MAX_RETRIES, NO_SIGNAL_AUC, REGISTRY, AdaptiveState  # noqa: E402
from dust_pipeline.schema import PERSISTENCE_COLUMN  # noqa: E402

IMBALANCE_RATE = 0.20  # minority share below this → plan a threshold tuner
MIN_ROWS_FOR_PROBE = 300

# Every transition the supervisor may take. A plan's edges are this list filtered
# to the agents the planner selected (with gaps bridged).
POSSIBLE_EDGES = [
    ("data_validator", "data_repairer"),  # repairable problems
    ("data_repairer", "data_validator"),  # re-validate after repair
    ("data_validator", "profiler"),
    ("data_validator", "reporter"),  # invalid data → stop cleanly
    ("profiler", "leakage_auditor"),
    ("leakage_auditor", "benchmark"),
    ("leakage_auditor", "signal_probe"),
    ("leakage_auditor", "strategist"),
    ("benchmark", "signal_probe"),
    ("benchmark", "strategist"),
    ("signal_probe", "strategist"),
    ("strategist", "trainer"),
    ("trainer", "trainer"),  # retry
    ("trainer", "threshold_tuner"),
    ("trainer", "critic"),
    ("threshold_tuner", "critic"),
    ("critic", "strategist"),  # revise loop
    ("critic", "evaluator"),
    ("critic", "reporter"),  # nothing trained successfully
    ("evaluator", "reporter"),
]


# ---------------------------------------------------------------------------
# PLANNER
# ---------------------------------------------------------------------------
def _edges_for(agents: list[str]) -> list[list[str]]:
    chosen = set(agents)
    return [[a, b] for a, b in POSSIBLE_EDGES if a in chosen and b in chosen]


def validate_plan(agents: list[str]) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems = [f"unknown agent: {a}" for a in agents if a not in REGISTRY]
    problems += [f"required agent missing: {n}" for n, (_f, _d, req) in REGISTRY.items() if req and n not in agents]
    if not problems:
        adj: dict[str, list[str]] = {}
        for a, b in _edges_for(agents):
            adj.setdefault(a, []).append(b)
        seen, stack = {"data_validator"}, ["data_validator"]
        while stack:
            for nxt in adj.get(stack.pop(), []):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        problems += [f"unreachable agent: {a}" for a in agents if a not in seen]
    return problems


def planner(state: AdaptiveState) -> dict:
    """Decide which agents this task needs and emit the workflow graph as data."""
    cfg = state.get("config") or {}
    try:
        df = pd.read_csv(state["dataset_path"])
        y = pd.to_numeric(df[state["target_column"]], errors="coerce").dropna()
        rate = float(y.mean()) if len(y) else 0.0
    except Exception:  # noqa: BLE001 — unreadable / target missing: plan the default graph, the validator will stop it
        df, rate = pd.DataFrame(), 0.5
    minority = min(rate, 1 - rate)
    facts = {"n_rows": int(len(df)), "n_columns": int(df.shape[1]), "minority_rate": round(minority, 4),
             "has_persistence_column": PERSISTENCE_COLUMN in df.columns}

    # Deterministic plan (also the fallback when the LLM plan is invalid).
    optional = {
        "signal_probe": len(df) >= MIN_ROWS_FOR_PROBE,
        "threshold_tuner": minority < IMBALANCE_RATE,
        "benchmark": PERSISTENCE_COLUMN in df.columns,
    }
    agents = [n for n, (_f, _d, req) in REGISTRY.items() if req or optional.get(n)]
    reasons = {
        "signal_probe": f"{'included' if optional['signal_probe'] else 'skipped'}: {len(df)} rows (probe needs ≥ {MIN_ROWS_FOR_PROBE})",
        "threshold_tuner": f"{'included' if optional['threshold_tuner'] else 'skipped'}: minority class is {minority:.1%} (imbalanced below {IMBALANCE_RATE:.0%})",
        "benchmark": f"{'included' if optional['benchmark'] else 'skipped'}: table {'has' if optional['benchmark'] else 'has no'} `{PERSISTENCE_COLUMN}` column for a persistence forecast",
    }
    source, repaired = "rules", None

    model = (cfg.get("model") or DEFAULT_MODEL).strip()
    if cfg.get("use_llm") and llm_configured(model):
        system = (
            "You are the Planner of a multi-agent ML system. Choose which agents this task needs. "
            'Return ONLY JSON: {"agents": ["..."], "reasons": {"agent": "why included or skipped"}}. '
            "Use only agent names from the registry. Required agents must all be included; "
            "optional agents only when the dataset facts justify them."
        )
        user = (
            f"dataset_facts={json.dumps(facts)}\n"
            f"registry={json.dumps({n: {'does': d, 'required': r} for n, (_f, d, r) in REGISTRY.items()})}\n"
        )
        content, _err = openrouter_chat(system, user, model=model, max_tokens=600, temperature=0.1, json_object=True)
        parsed = extract_json_object(content or "") or {}
        proposed = [a for a in parsed.get("agents") or [] if isinstance(a, str)]
        problems = validate_plan(proposed) if proposed else ["no plan returned"]
        if not problems:
            agents = [n for n in REGISTRY if n in proposed]  # canonical order
            reasons = {**reasons, **{k: str(v) for k, v in (parsed.get("reasons") or {}).items() if k in REGISTRY}}
            source = f"llm · {model}"
        else:
            repaired = f"LLM plan rejected ({'; '.join(problems[:3])}) → rule-based plan used"

    plan = {
        "agents": agents,
        "edges": _edges_for(agents),
        "entry": "data_validator",
        "facts": facts,
        "reasons": reasons,
        "source": source,
        "repaired": repaired,
    }
    note = f"{len(agents)} agents via {source}" + (f" · {repaired}" if repaired else "")
    return {"plan": plan, "steps": [{"agent": "planner", "note": note, "output": plan}]}


# ---------------------------------------------------------------------------
# SUPERVISOR — the routing function behind every edge
# ---------------------------------------------------------------------------
def supervise(state: AdaptiveState, after: str) -> tuple[str, str]:
    """Return (next agent, reason) given the state after `after` ran."""
    planned = set((state.get("plan") or {}).get("agents") or [])
    result = state.get("result") or {}
    metrics = result.get("validation_metrics") or {}

    if after == "data_validator":
        v = state.get("validation") or {}
        if v.get("status") == "invalid":
            return "reporter", "data is invalid → stop without modelling"
        if v.get("status") == "needs_repair":
            if int(state.get("repairs") or 0) < MAX_REPAIRS:
                return "data_repairer", "repairable problems → repair, then validate again"
            return "reporter", f"still not clean after {MAX_REPAIRS} repair pass(es) → stop"
        return "profiler", "data is valid"
    if after == "data_repairer":
        return "data_validator", "re-validate the repaired table"
    if after == "profiler":
        return "leakage_auditor", "always audit before modelling"
    if after in {"leakage_auditor", "benchmark"}:
        if after == "leakage_auditor" and "benchmark" in planned:
            return "benchmark", "persistence benchmark planned"
        if "signal_probe" in planned:
            return "signal_probe", "probe planned"
        return "strategist", "no probe in plan"
    if after == "signal_probe":
        return "strategist", f"budget set to {state.get('budget')} trial(s)"
    if after == "strategist":
        return "trainer", "train the proposed spec"
    if after == "trainer":
        if result.get("status") != "succeeded":
            if int(state.get("attempt") or 0) <= MAX_RETRIES:
                return "trainer", f"fit failed → retry {state.get('attempt')}/{MAX_RETRIES} with safe hyperparameters"
            return "critic", f"fit failed {state.get('attempt')} times → give up on this family"
        auc = metrics.get("roc_auc") or 0
        if "threshold_tuner" not in planned:
            return "critic", "no tuner in plan (classes are balanced)"
        if auc < NO_SIGNAL_AUC:
            return "critic", f"skip tuner: val ROC-AUC {auc:.3f} is chance level, nothing to tune"
        return "threshold_tuner", f"val ROC-AUC {auc:.3f} → tune the decision threshold"
    if after == "threshold_tuner":
        return "critic", "tuned"
    if after == "critic":
        if (state.get("validation") or {}).get("status") == "invalid":
            return "reporter", "invalid data"
        action = (state.get("verdict") or {}).get("action")
        if action == "revise":
            return "strategist", (state.get("verdict") or {}).get("reason", "revise")
        if not state.get("best"):
            return "reporter", "no successful model → skip holdout, report the failure"
        return "evaluator", f"{action}: {(state.get('verdict') or {}).get('reason')}"
    if after == "evaluator":
        return "reporter", "write the report"
    return END, "done"


# ---------------------------------------------------------------------------
# COMPILE the plan into a LangGraph
# ---------------------------------------------------------------------------
def _wrap(name: str):
    """Agent node + supervisor decision recorded in the same state update."""
    fn = REGISTRY[name][0]

    def node(state: AdaptiveState) -> dict:
        patch = fn(state)
        nxt, reason = supervise({**state, **{k: v for k, v in patch.items() if k not in {"steps", "route_log"}}}, name)
        patch["route_log"] = [{"after": name, "next": nxt if nxt != END else "END", "reason": reason}]
        return patch

    node.__name__ = name
    return node


def _route(state: AdaptiveState) -> str:
    return (state.get("route_log") or [{}])[-1].get("next", "END")


def compile_plan(plan: dict):
    """Build the runnable graph from the planner's JSON plan."""
    problems = validate_plan(plan["agents"])
    if problems:
        raise ValueError(f"invalid plan: {problems}")
    g = StateGraph(AdaptiveState)
    for name in plan["agents"]:
        g.add_node(name, _wrap(name))
    g.add_edge(START, plan["entry"])
    targets: dict[str, list[str]] = {}
    for a, b in plan["edges"]:
        targets.setdefault(a, []).append(b)
    for name in plan["agents"]:
        options = {t: t for t in targets.get(name, [])}
        if name == "reporter":
            options = {"END": END}
        g.add_conditional_edges(name, _route, options)
    return g.compile()


def plan_to_mermaid(plan: dict) -> str:
    """The planned workflow as a Mermaid flowchart (rendered by GitHub and the dashboard)."""
    labels = {("data_validator", "data_repairer"): "repairable", ("data_validator", "reporter"): "invalid data",
              ("data_repairer", "data_validator"): "re-validate", ("trainer", "trainer"): "retry", ("critic", "strategist"): "revise", ("critic", "reporter"): "all fits failed",
              ("trainer", "critic"): "failed / nothing to tune", ("trainer", "threshold_tuner"): "signal",
              ("critic", "evaluator"): "accept / stop / plateau"}
    lines = ["flowchart LR", "  START([START]) --> planner", f"  planner -- writes plan --> {plan['entry']}"]
    for a, b in plan["edges"]:
        label = labels.get((a, b))
        lines.append(f"  {a} -- {label} --> {b}" if label else f"  {a} --> {b}")
    lines.append("  reporter --> END([END])")
    optional = [n for n in plan["agents"] if not REGISTRY[n][2]]
    if optional:
        lines.append("  classDef optional stroke-dasharray: 4 3")
        lines.append(f"  class {','.join(optional)} optional")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------
def run_adaptive(
    *,
    dataset_path: str,
    target_column: str = "low_vis_next_day",
    target_score: float = 0.35,
    max_trials: int = 3,
    seed: int = 42,
    holdout_frac: float = 0.2,
    use_llm: bool = False,
    model: str | None = None,
) -> dict:
    """Plan, compile and execute. Never raises for a failed training job."""
    state: AdaptiveState = {
        "dataset_path": str(Path(dataset_path).resolve()),
        "target_column": target_column,
        "config": {
            "target_score": target_score,
            "max_trials": max_trials,
            "seed": seed,
            "holdout_frac": holdout_frac,
            "use_llm": use_llm,
            "model": (model or DEFAULT_MODEL),
        },
        "steps": [],
        "route_log": [],
        "trials": [],
        "failed_families": [],
    }
    planned = planner(state)  # the planner runs first, outside the graph it is about to create
    state = {**state, "plan": planned["plan"], "steps": planned["steps"]}
    app = compile_plan(state["plan"])
    limit = 20 + max_trials * (4 + MAX_RETRIES) * 2
    final = app.invoke(state, config={"recursion_limit": limit, "run_name": "adaptive_dust_forecast_workflow"})
    return {
        "plan": final["plan"],
        "mermaid": plan_to_mermaid(final["plan"]),
        "report": final.get("report"),
        "stop_reason": final.get("stop_reason"),
        "validation": final.get("validation"),
        "audit": final.get("audit"),
        "benchmark": final.get("benchmark"),
        "probe": final.get("probe"),
        "trials": final.get("trials"),
        "best": final.get("best"),
        "holdout": final.get("holdout"),
        "steps": final.get("steps"),
        "route_log": final.get("route_log"),
    }


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Adaptive multi-agent dust-visibility forecast workflow")
    p.add_argument("--dataset", default=str(REPO / "datasets" / "airflow" / "saudi_dust_daily_latest.csv"))
    p.add_argument("--target", default="low_vis_next_day")
    p.add_argument("--target-score", type=float, default=0.35)
    p.add_argument("--max-trials", type=int, default=3)
    p.add_argument("--llm", action="store_true", help="let the planner and strategist call the LLM")
    p.add_argument("--json", action="store_true", help="print the full result as JSON")
    a = p.parse_args()
    out = run_adaptive(
        dataset_path=a.dataset, target_column=a.target, target_score=a.target_score, max_trials=a.max_trials, use_llm=a.llm
    )
    if a.json:
        print(json.dumps(out, indent=2, default=str))
        return 0
    print("PLAN:", " → ".join(out["plan"]["agents"]), f"({out['plan']['source']})")
    for k, v in out["plan"]["reasons"].items():
        print(f"  {k}: {v}")
    print("\nTRACE:")
    for step, route in zip(out["steps"][1:], out["route_log"]):
        print(f"  {step['agent']:<16} {step['note']}\n  {'':<16} ↳ supervisor → {route['next']}: {route['reason']}")
    h = out.get("holdout") or {}
    print(f"\nSTOP: {out['stop_reason']} · trials {out['report']['trials']}/{a.max_trials} · retries {out['report']['retries']}")
    if h:
        print(f"HOLDOUT: f1={h['f1']:.3f} precision={h['precision']:.3f} recall={h['recall']:.3f} roc_auc={h['roc_auc']:.3f} accuracy={h['accuracy']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
