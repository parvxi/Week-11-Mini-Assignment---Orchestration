"""Part 2 agent workflow: graph shape, leakage audit, chronological split, and a full no-LLM run."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_loop.graph import REQUIRED_NODES, get_status, run_simple_loop
from student_loop.training import audit_leakage, scores_at, tune_threshold

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def events_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Small table with one real signal, one leaked column, an id and a date."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    rng = np.random.default_rng(0)
    n = 1500
    signal = rng.normal(size=n)
    y = (signal + rng.normal(scale=1.5, size=n) > 2.0).astype(int)  # useful, far from perfect
    df = pd.DataFrame(
        {
            "row_id": [f"T{i:05d}" for i in range(n)],
            "created": pd.date_range("2026-01-01", periods=n, freq="h").astype(str),
            "signal": signal,
            "noise": rng.normal(size=n),
            "channel": rng.choice(["web", "pos", "app"], size=n),
            "leaked_score": np.where(y == 1, rng.uniform(0.6, 1, n), rng.uniform(0, 0.4, n)),
            "label": y,
        }
    )
    path = tmp_path / "events.csv"
    df.to_csv(path, index=False)
    return path


def test_graph_has_the_modified_agents() -> None:
    status = get_status()
    assert status["ready"], status["error"]
    assert set(status["nodes"]) == set(REQUIRED_NODES)
    assert "context_builder" not in status["nodes"]
    edges = {(e["source"], e["target"]) for e in status["edges"]}
    assert ("data_scientist", "leakage_auditor") in edges
    assert ("ml_engineer", "threshold_tuner") in edges and ("ml_engineer", "judge") in edges
    assert ("memory_writer", "hypothesis_generator") in edges  # the loop


def test_audit_blocks_leak_id_and_date_but_keeps_real_signal(events_csv: Path) -> None:
    audit = audit_leakage(pd.read_csv(events_csv), "label", seed=42, holdout_frac=0.2)
    reasons = {f["column"]: f["reason"] for f in audit["findings"]}
    assert reasons == {"row_id": "identifier", "created": "raw_date", "leaked_score": "target_leak"}
    assert audit["leak_columns"] == ["leaked_score"]


def test_threshold_tuning_beats_default_on_rare_positives() -> None:
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=4000) < 0.05).astype(int)
    proba = np.clip(0.05 + 0.15 * y + rng.normal(scale=0.05, size=4000), 0, 1)
    tuned = scores_at(y, proba, tune_threshold(y, proba)["threshold"])
    assert tuned["f1"] > scores_at(y, proba, 0.5)["f1"]


def test_full_loop_without_llm_never_trains_on_the_leak(events_csv: Path) -> None:
    result = run_simple_loop(dataset_path=str(events_csv), target_column="label", target_score=0.99, max_cycles=2)
    assert result["cycles"] == 2  # target unreachable → revise once, then budget stop
    assert result["stop_reason"] == "budget_exhausted"
    for step in result["steps"]:
        if step["node"] == "ml_engineer":
            assert "leaked_score" in step["spec"]["drop_columns"]
    holdout = result["holdout_metrics"]
    assert 0.5 < holdout["roc_auc"] < 0.999  # real signal, but not the leaked answer
    assert holdout["leakage_ablation"]["roc_auc"] > 0.999
    assert set(holdout["confusion_matrix"]) == {"tn", "fp", "fn", "tp"}


def test_dag_is_scheduled_and_uses_task_groups() -> None:
    """Static check (no Airflow import needed on the host)."""
    source = (REPO / "airflow" / "dags" / "saudi_dust_visibility_pipeline.py").read_text()
    ast.parse(source)
    assert 'schedule="0 3 * * *"' in source and "catchup=False" in source
    for group in ("noaa_observations", "cams_dust", "load", "analysis"):
        assert f'@task_group(group_id="{group}")' in source
    assert "PostgresHook" in source


def test_time_column_gives_a_chronological_split() -> None:
    from student_loop.training import split_frames

    n = 400
    df = pd.DataFrame({"obs_date": pd.date_range("2024-01-01", periods=n).astype(str), "x": range(n), "label": [i % 7 == 0 for i in range(n)]})
    train, val, hold = split_frames(df.sample(frac=1, random_state=0), "label", seed=1, holdout_frac=0.2)
    assert train["obs_date"].max() < val["obs_date"].min() <= val["obs_date"].max() < hold["obs_date"].min()
    assert len(hold) == 80 and len(train) + len(val) + len(hold) == n


def test_declared_outcome_columns_are_always_blocked() -> None:
    rng = np.random.default_rng(3)
    n = 1200
    y = (rng.uniform(size=n) < 0.2).astype(int)
    df = pd.DataFrame({
        "x": rng.normal(size=n),
        # only weakly related to the label, so measurement alone would not flag it
        "next_day_min_visibility_m": 9000 - 500 * y + rng.normal(scale=3000, size=n),
        "label": y,
    })
    audit = audit_leakage(df, "label", seed=42, holdout_frac=0.2)
    assert {f["column"]: f["reason"] for f in audit["findings"]} == {"next_day_min_visibility_m": "declared_outcome"}
