"""Run every case against every system; write results.json, results.md and dashboard.html.

    python -m evals.run_evals            # all systems
    python -m evals.run_evals adaptive   # one system
"""

from __future__ import annotations

import json
import sys
import tempfile
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from adaptive_loop.graph import run_adaptive  # noqa: E402
from evals.cases import CASES  # noqa: E402
from evals.dashboard import render  # noqa: E402
from evals.datasets import DESCRIPTIONS, SAUDI_REAL, materialize  # noqa: E402
from evals.systems import BUDGET, SYSTEM_LABELS, SYSTEMS, TARGET_SCORE, _no_llm, execute, inject  # noqa: E402

OUT = REPO / "evals" / "results"


def showcase(paths: dict) -> dict:
    """Four fully traced adaptive runs for the dashboard: normal, repair, retry, invalid data."""
    keys = ("plan", "mermaid", "steps", "route_log", "stop_reason", "holdout", "trials", "report")
    runs = {
        "saudi_real": lambda: run_adaptive(dataset_path=str(SAUDI_REAL), target_score=TARGET_SCORE, max_trials=6),
        "saudi_dirty": lambda: run_adaptive(dataset_path=paths["saudi_dirty"][0], target_score=TARGET_SCORE, max_trials=BUDGET),
        "invalid_data": lambda: run_adaptive(dataset_path=paths["saudi_single_class"][0], target_score=TARGET_SCORE, max_trials=BUDGET),
    }
    out = {}
    with _no_llm(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, fn in runs.items():
            result = fn()
            out[name] = {k: result[k] for k in keys}
        with inject({"first_n": 4}):
            r = run_adaptive(dataset_path=paths["saudi_model_ready"][0], target_score=TARGET_SCORE, max_trials=BUDGET)
        out["training_outage"] = {k: r[k] for k in keys}
    return out


def refresh_showcase() -> int:
    """Re-run only the traced showcase runs and re-render the dashboard from results.json."""
    payload = json.loads((OUT / "results.json").read_text())
    payload["showcase"] = showcase(materialize(Path(tempfile.mkdtemp(prefix="evals_"))))
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "dashboard.html").write_text(render(payload))
    return 0


def main(systems: list[str]) -> int:
    paths = materialize(Path(tempfile.mkdtemp(prefix="evals_")))
    cache: dict[tuple, dict] = {}

    def run(system: str, spec: dict) -> dict:
        budget = spec.get("budget", BUDGET)
        key = (system, spec["dataset"], json.dumps(spec.get("fault"), sort_keys=True), budget)
        if key not in cache:
            path, target = paths[spec["dataset"]]
            ref = paths[spec["referee_dataset"]][0] if spec.get("referee_dataset") else None
            cache[key] = execute(system, path, target, spec.get("fault"), budget, ref)
        return cache[key]

    rows = []
    for case in CASES:
        row = {
            "id": case.id, "kind": case.kind, "measures": case.measures, "title": case.title,
            "input": {k: {"dataset": v["dataset"], "dataset_description": DESCRIPTIONS[v["dataset"]], "fault": v.get("fault"),
                          "budget": v.get("budget", BUDGET)}
                      for k, v in case.runs.items()},
            "run_config": {"target_f1": TARGET_SCORE, "llm": "off (deterministic paths)"},
            "expected": case.expected, "scoring": case.scoring, "threshold": case.threshold, "results": {},
        }
        for system in systems:
            runs = {k: run(system, v) for k, v in case.runs.items()}
            score, evidence = case.score(runs)
            row["results"][system] = {"score": round(float(score), 3), "passed": bool(score >= case.threshold), "evidence": evidence}
            print(f"{case.id:<22} {system:<15} {'PASS' if score >= case.threshold else 'FAIL'}  {score:.3f}  {evidence}", flush=True)
        rows.append(row)

    def tally(system: str, kind: str | None = None) -> dict:
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        passed = sum(r["results"][system]["passed"] for r in sel)
        return {"passed": passed, "total": len(sel), "mean_score": round(sum(r["results"][system]["score"] for r in sel) / len(sel), 3)}

    def coordination(system: str) -> dict:
        clean = [r for (s, _d, fault, _b), r in cache.items() if s == system and fault == "null"]
        done = [r for r in clean if r["completed"]]
        faulted = [r for (s, _d, fault, _b), r in cache.items() if s == system and fault != "null"]
        mean = lambda xs: round(sum(xs) / len(xs), 2) if xs else None  # noqa: E731
        return {
            "runs_without_fault": len(clean),
            "mean_trials": mean([r["trials"] for r in done]),
            "mean_agent_steps": mean([r["agent_steps"] for r in done if r["agent_steps"] is not None]),
            "mean_seconds": mean([r["seconds"] for r in done]),
            "fault_runs_survived": f"{sum(1 for r in faulted if not r['error'])}/{len(faulted)}",
        }

    summary = {s: {"label": SYSTEM_LABELS[s], "all": tally(s), "outcome": tally(s, "outcome"),
                   "capability": tally(s, "capability"), "coordination": coordination(s)} for s in systems}

    # Predictive performance on the same clean table and holdout, separate from orchestration.
    predictive = {}
    for system in systems:
        r = cache.get((system, "saudi_model_ready", "null", BUDGET)) or run(system, {"dataset": "saudi_model_ready"})
        ref = r.get("referee") or {}
        predictive[system] = {
            "model_family": (r.get("final_spec") or {}).get("model_family"), "threshold": ref.get("threshold"),
            **{k: ref.get(k) for k in ("f1", "precision", "recall", "roc_auc", "pr_auc", "persistence_f1")},
            "trials": r.get("trials"), "seconds": r.get("seconds"), "error": r.get("error"),
        }

    payload = {"systems": systems, "summary": summary, "cases": rows, "predictive": predictive, "showcase": showcase(paths)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(payload, indent=2, default=str))
    (OUT / "dashboard.html").write_text(render(payload))

    head = "| Case | Measures | Threshold | " + " | ".join(SYSTEM_LABELS[s] for s in systems) + " |"
    lines = [head, "| --- | --- | ---: | " + " | ".join("---" for _ in systems) + " |"]
    for r in rows:
        cells = [f"{'✅' if r['results'][s]['passed'] else '❌'} {r['results'][s]['score']:.2f}" for s in systems]
        lines.append(f"| `{r['id']}` | {r['measures']} | {r['threshold']:.2f} | " + " | ".join(cells) + " |")
    lines.append("| **Passed** | | | " + " | ".join(f"**{summary[s]['all']['passed']}/{summary[s]['all']['total']}**" for s in systems) + " |")
    (OUT / "results.md").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(json.dumps({s: summary[s]["coordination"] for s in systems}, indent=1))
    return 0


if __name__ == "__main__":
    if "--showcase-only" in sys.argv:
        raise SystemExit(refresh_showcase())
    chosen = [a for a in sys.argv[1:] if a in SYSTEMS] or list(SYSTEMS)
    raise SystemExit(main(chosen))
