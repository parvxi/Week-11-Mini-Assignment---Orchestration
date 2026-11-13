"""Training, threshold and leakage helpers for the modified agent workflow.

Everything here is deterministic code (no LLM). The agents in graph.py decide
*what* to try; these functions do the fitting and the measuring.

Split protocol (identical for every agent). Station-days are split by DATE:

    oldest ───────────────────────────────────────────────────────────── newest
    [ train: first 60% of days ][ validation: next 20% ][ holdout: last 20% ]
                                  ml_engineer / tuner / judge    opened once, by holdout_eval
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dust_pipeline.schema import OUTCOME_COLUMNS, PERSISTENCE_COLUMN, TIME_COLUMN
from loop_runtime import _build_model

# A single column that separates the classes this well on its own is not a
# feature, it is the answer key.
LEAK_AUC = 0.95
ID_UNIQUE_RATIO = 0.9
MAX_CATEGORIES = 50
MODEL_SEED = 42


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def split_frames(df: pd.DataFrame, target: str, seed: int, holdout_frac: float):
    """Return (train, validation, holdout) frames.

    Forecasting data (a TIME_COLUMN is present): split by date, oldest → newest, so the
    model is always evaluated on days that come after everything it was trained on.
    A random split would let it learn from tomorrow to predict yesterday.
    Otherwise: stratified random split (used by the synthetic evaluation tables).
    """
    if TIME_COLUMN in df.columns:
        dates = np.sort(pd.to_datetime(df[TIME_COLUMN]).unique())
        hold_start = dates[int(len(dates) * (1 - holdout_frac))]
        dev_dates = dates[dates < hold_start]
        val_start = dev_dates[int(len(dev_dates) * 0.75)]
        when = pd.to_datetime(df[TIME_COLUMN])
        return df[when < val_start], df[(when >= val_start) & (when < hold_start)], df[when >= hold_start]
    strat = df[target] if df[target].nunique() > 1 else None
    dev, hold = train_test_split(df, test_size=holdout_frac, random_state=seed, stratify=strat)
    strat_dev = dev[target] if dev[target].nunique() > 1 else None
    train, val = train_test_split(dev, test_size=0.25, random_state=seed, stratify=strat_dev)
    return train, val, hold


def split_description(df: pd.DataFrame, target: str, seed: int, holdout_frac: float) -> dict:
    """Human-readable summary of the three splits (row counts, date ranges, positive rates)."""
    out = {"strategy": "chronological" if TIME_COLUMN in df.columns else "stratified_random"}
    for name, part in zip(("train", "validation", "holdout"), split_frames(df, target, seed, holdout_frac)):
        info = {"rows": int(len(part)), "positive_rate": round(float(part[target].mean()), 4)}
        if TIME_COLUMN in part.columns and len(part):
            info["from"], info["to"] = str(part[TIME_COLUMN].min())[:10], str(part[TIME_COLUMN].max())[:10]
        out[name] = info
    return out


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------
def _looks_like_date(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty or not sample.str.contains(r"\d{4}-\d{2}-\d{2}").all():
        return False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(pd.to_datetime(sample, errors="coerce").notna().all())


def _single_feature_auc(x: pd.Series, y: pd.Series) -> float | None:
    """How well does this one column rank the positives, with no model at all?"""
    if y.nunique() != 2:
        return None
    if pd.api.types.is_numeric_dtype(x):
        score = pd.to_numeric(x, errors="coerce")
        if score.nunique(dropna=True) < 2:
            return None
        score = score.fillna(score.median())
    else:
        if x.nunique(dropna=True) > MAX_CATEGORIES:
            return None
        score = x.map(y.groupby(x).mean()).fillna(y.mean())  # target-rate encoding
    auc = float(roc_auc_score(y, score))
    return max(auc, 1 - auc)  # direction does not matter


def audit_leakage(df: pd.DataFrame, target: str, seed: int, holdout_frac: float) -> dict:
    """Measure every column instead of guessing from its name.

    Uses only the development rows, so the holdout stays sealed.
    Returns findings plus the list of columns the trainer must not use.
    """
    train, val, _hold = split_frames(df, target, seed, holdout_frac)
    dev = pd.concat([train, val])
    y = dev[target]
    findings: list[dict] = []
    aucs: dict[str, float] = {}

    for col in dev.columns:
        if col == target:
            continue
        x = dev[col]
        n_unique = int(x.nunique(dropna=True))
        unique_ratio = n_unique / max(len(x), 1)
        is_text = not pd.api.types.is_numeric_dtype(x)

        if col in OUTCOME_COLUMNS:
            # Pipeline contract: these hold tomorrow's raw outcome. Blocked whatever they measure.
            auc = _single_feature_auc(x, y)
            findings.append(
                {
                    "column": col,
                    "reason": "declared_outcome",
                    "evidence": "next-day outcome measurement (schema contract)"
                    + (f"; alone it gives ROC-AUC {auc:.3f}" if auc is not None else ""),
                }
            )
            continue
        if is_text and _looks_like_date(x):
            findings.append({"column": col, "reason": "raw_date", "evidence": "date string, not a model feature"})
            continue
        # Keys are identifiers whatever their dtype (station_id is a number, not a quantity).
        if col.lower().endswith("_id") or (is_text and unique_ratio >= ID_UNIQUE_RATIO):
            findings.append(
                {"column": col, "reason": "identifier", "evidence": f"{n_unique} distinct values in {len(x)} rows"}
            )
            continue
        if is_text and n_unique > MAX_CATEGORIES:
            findings.append(
                {"column": col, "reason": "high_cardinality", "evidence": f"{n_unique} categories"}
            )
            continue

        auc = _single_feature_auc(x, y)
        if auc is None:
            continue
        aucs[col] = round(auc, 4)
        if auc >= LEAK_AUC:
            findings.append(
                {
                    "column": col,
                    "reason": "target_leak",
                    "evidence": f"this column alone gives ROC-AUC {auc:.3f} on the development rows",
                }
            )

    legit = {c: a for c, a in aucs.items() if a < LEAK_AUC}
    strongest = sorted(legit.items(), key=lambda kv: -kv[1])[:5]
    return {
        "findings": findings,
        "drop_columns": [f["column"] for f in findings],
        "leak_columns": [f["column"] for f in findings if f["reason"] in {"target_leak", "declared_outcome"}],
        "strongest_legit_features": [{"column": c, "single_feature_auc": a} for c, a in strongest],
        "n_columns_checked": int(dev.shape[1] - 1),
    }


# ---------------------------------------------------------------------------
# Fit + score
# ---------------------------------------------------------------------------
def _pipeline(X: pd.DataFrame, spec: dict) -> Pipeline:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat,
            ),
        ]
    )
    family = str(spec.get("model_family") or "logistic_regression")
    model = _build_model(family, dict(spec.get("hyperparameters") or {}))
    if "random_state" in model.get_params():
        model.set_params(random_state=MODEL_SEED)  # same spec → same model, every run
    return Pipeline([("pre", pre), ("clf", model)])


def _features(df: pd.DataFrame, target: str, drops: list[str]) -> pd.DataFrame:
    return df.drop(columns=[target, *[c for c in drops if c in df.columns and c != target]], errors="ignore")


def scores_at(y_true, proba, threshold: float) -> dict:
    """Classification metrics for one decision threshold."""
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) == 2
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else None,
        "pr_auc": float(average_precision_score(y_true, proba)) if two_classes else None,
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def persistence_benchmark(df: pd.DataFrame, target: str, seed: int, holdout_frac: float) -> dict:
    """Score "tomorrow = today" on the validation rows: the floor any model has to beat."""
    _train, val, _hold = split_frames(df, target, seed, holdout_frac)
    if PERSISTENCE_COLUMN not in val.columns:
        return {"available": False, "reason": f"no {PERSISTENCE_COLUMN} column"}
    naive = val[PERSISTENCE_COLUMN].fillna(0).astype(int)
    y = val[target]
    return {
        "available": True,
        "rule": f"predict {target} = {PERSISTENCE_COLUMN}",
        "f1": float(f1_score(y, naive, zero_division=0)),
        "precision": float(precision_score(y, naive, zero_division=0)),
        "recall": float(recall_score(y, naive, zero_division=0)),
        "n_val": int(len(val)),
        "val_positive_rate": float(y.mean()),
    }


def fit_validate(df: pd.DataFrame, target: str, spec: dict, seed: int, holdout_frac: float) -> dict:
    """Train on the train rows, score the validation rows at the default 0.5 threshold."""
    train, val, hold = split_frames(df, target, seed, holdout_frac)
    drops = list(spec.get("drop_columns") or [])
    X_train, X_val = _features(train, target, drops), _features(val, target, drops)
    pipe = _pipeline(X_train, spec)
    t0 = time.time()
    pipe.fit(X_train, train[target])
    proba = pipe.predict_proba(X_val)[:, 1]
    duration = round(time.time() - t0, 3)
    return {
        "status": "succeeded",
        "experiment_id": spec.get("experiment_id"),
        "model_family": spec.get("model_family"),
        "hyperparameters": spec.get("hyperparameters") or {},
        "drop_columns": [c for c in drops if c in df.columns],
        "n_features": int(X_train.shape[1]),
        "validation_metrics": scores_at(val[target], proba, 0.5),
        "duration_seconds": duration,
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_holdout": int(len(hold)),
        # Kept in graph state so threshold_tuner does not have to refit.
        "val_scores": {"y_true": [int(v) for v in val[target]], "proba": [round(float(p), 6) for p in proba]},
    }


def tune_threshold(y_true, proba) -> dict:
    """Pick the decision threshold with the best validation F1.

    With ~3% positives, 0.5 is rarely the right cut-off. Candidates come from
    the precision-recall curve, i.e. every distinct score the model produced.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"threshold": 0.5, "reason": "only one class in validation"}
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    return {"threshold": float(thresholds[best]), "n_candidates": int(len(thresholds))}


def fit_holdout(
    df: pd.DataFrame, target: str, spec: dict, threshold: float, seed: int, holdout_frac: float
) -> dict:
    """Refit on train + validation, score the sealed holdout once."""
    train, val, hold = split_frames(df, target, seed, holdout_frac)
    dev = pd.concat([train, val])
    drops = list(spec.get("drop_columns") or [])
    X_dev, X_hold = _features(dev, target, drops), _features(hold, target, drops)
    pipe = _pipeline(X_dev, spec)
    pipe.fit(X_dev, dev[target])
    proba = pipe.predict_proba(X_hold)[:, 1]
    metrics = scores_at(hold[target], proba, threshold)
    positive_rate = float(hold[target].mean())
    if PERSISTENCE_COLUMN in hold.columns:
        # Simple predictive baseline: "tomorrow will be like today".
        naive = hold[PERSISTENCE_COLUMN].fillna(0).astype(int)
        metrics["persistence_baseline"] = {
            "f1": float(f1_score(hold[target], naive, zero_division=0)),
            "precision": float(precision_score(hold[target], naive, zero_division=0)),
            "recall": float(recall_score(hold[target], naive, zero_division=0)),
        }
    if TIME_COLUMN in hold.columns:
        metrics["holdout_period"] = f"{str(hold[TIME_COLUMN].min())[:10]} → {str(hold[TIME_COLUMN].max())[:10]}"
    metrics.update(
        {
            "n_holdout": int(len(hold)),
            "positive_rate": positive_rate,
            # Accuracy of "always predict the majority class", for context.
            "majority_class_accuracy": float(max(positive_rate, 1 - positive_rate)),
            "n_features": int(X_dev.shape[1]),
        }
    )
    return metrics
