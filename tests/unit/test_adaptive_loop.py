"""Part 3: planner output, plan validation, supervisor routing, and recovery from faults."""

from __future__ import annotations

from pathlib import Path

import pytest

from adaptive_loop.graph import compile_plan, plan_to_mermaid, planner, run_adaptive, supervise, validate_plan
from evals.cases import CASES
from evals.datasets import learnable_balanced, learnable_imbalanced
from evals.systems import execute


@pytest.fixture(scope="module")
def tables(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("adaptive")
    out = {}
    for name, gen in (("imbalanced", learnable_imbalanced), ("balanced", learnable_balanced)):
        df = gen().head(1500)
        path = root / f"{name}.csv"
        df.to_csv(path, index=False)
        out[name] = str(path)
    return out


def _plan(path: str) -> dict:
    return planner({"dataset_path": path, "target_column": "label", "config": {}})["plan"]


def test_planner_selects_tuner_only_for_imbalanced_data(tables: dict[str, str]) -> None:
    assert "threshold_tuner" in _plan(tables["imbalanced"])["agents"]
    assert "threshold_tuner" not in _plan(tables["balanced"])["agents"]


def test_plan_validation_rejects_bad_plans(tables: dict[str, str]) -> None:
    good = _plan(tables["imbalanced"])["agents"]
    assert validate_plan(good) == []
    assert any("required agent missing: trainer" in p for p in validate_plan([a for a in good if a != "trainer"]))
    assert any("unknown agent" in p for p in validate_plan([*good, "magic_agent"]))
    with pytest.raises(ValueError):
        compile_plan({"agents": ["profiler"], "edges": [], "entry": "profiler"})


def test_compiled_graph_matches_the_plan(tables: dict[str, str]) -> None:
    plan = _plan(tables["balanced"])
    nodes = {n for n in compile_plan(plan).get_graph().nodes if not n.startswith("__")}
    assert nodes == set(plan["agents"])
    assert "threshold_tuner" not in plan_to_mermaid(plan)


def test_supervisor_retries_then_gives_up() -> None:
    plan = {"agents": ["trainer", "critic", "threshold_tuner"]}
    failed = {"plan": plan, "result": {"status": "failed"}}
    assert supervise({**failed, "attempt": 1}, "trainer")[0] == "trainer"
    assert supervise({**failed, "attempt": 3}, "trainer")[0] == "critic"
    ok = {"plan": plan, "result": {"status": "succeeded", "validation_metrics": {"roc_auc": 0.8}}}
    assert supervise(ok, "trainer")[0] == "threshold_tuner"
    chance = {"plan": plan, "result": {"status": "succeeded", "validation_metrics": {"roc_auc": 0.5}}}
    assert supervise(chance, "trainer")[0] == "critic"


def test_run_follows_planned_edges_and_reports(tables: dict[str, str]) -> None:
    out = run_adaptive(dataset_path=tables["imbalanced"], target_column="label", target_score=0.99, max_trials=2)
    edges = {(a, b) for a, b in out["plan"]["edges"]}
    assert all(r["next"] == "END" or (r["after"], r["next"]) in edges for r in out["route_log"])
    assert out["report"]["trials"] == 2 and out["holdout"]["roc_auc"] > 0.7


def test_total_outage_ends_cleanly(tables: dict[str, str]) -> None:
    result = execute("adaptive", tables["imbalanced"], "label", fault={"all": True})
    assert result["error"] is None
    assert result["stop_reason"] == "training_failed" and result["reported_metrics"] is None


def test_every_case_is_fully_specified() -> None:
    assert len({c.id for c in CASES}) == len(CASES)
    for c in CASES:
        assert c.runs and c.expected and c.scoring and 0 < c.threshold <= 1 and callable(c.score)


def test_validator_routes_dirty_data_through_repair(tables: dict[str, str], tmp_path: Path) -> None:
    import numpy as np
    import pandas as pd

    df = pd.read_csv(tables["imbalanced"]).astype({"label": "float"})
    df.loc[:9, "label"] = np.nan
    df.loc[20:24, "amount"] = np.inf
    dirty = tmp_path / "dirty.csv"
    pd.concat([df, df.head(30)]).to_csv(dirty, index=False)
    out = run_adaptive(dataset_path=str(dirty), target_column="label", target_score=0.99, max_trials=1)
    path = [(r["after"], r["next"]) for r in out["route_log"]]
    assert path[:3] == [("data_validator", "data_repairer"), ("data_repairer", "data_validator"), ("data_validator", "profiler")]
    assert len(out["report"]["repairs"]) == 3 and out["holdout"] is not None


def test_single_class_target_stops_before_modelling(tables: dict[str, str], tmp_path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(tables["imbalanced"])
    df["label"] = 0
    bad = tmp_path / "single.csv"
    df.to_csv(bad, index=False)
    out = run_adaptive(dataset_path=str(bad), target_column="label")
    assert out["stop_reason"] == "invalid_data" and out["trials"] == [] and out["holdout"] is None
    assert [r["next"] for r in out["route_log"]] == ["reporter", "END"]


def test_plateau_rule() -> None:
    from adaptive_loop.agents import _plateaued

    def t(family: str, f1: float) -> dict:
        return {"model_family": family, "status": "succeeded", "f1": f1}

    three = [t("logistic_regression", 0.20), t("random_forest", 0.25), t("hist_gradient_boosting", 0.22)]
    assert not _plateaued(three)  # every family once: not enough evidence yet
    assert not _plateaued([*three, t("random_forest", 0.30)])  # the refinement helped
    assert _plateaued([*three, t("random_forest", 0.255)])  # the refinement did not
