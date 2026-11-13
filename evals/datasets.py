"""Inputs for the evaluation suite.

REAL — the station-day table the Airflow DAG publishes, plus variants derived from it:
  saudi_real            as published (still contains tomorrow's outcome columns)
  saudi_model_ready     outcome columns removed, so no system can leak — isolates predictive skill
  saudi_shuffled_target negative control: target permuted, so nothing is learnable
  saudi_dirty           duplicate rows, null targets and infinite values injected (repairable)
  saudi_single_class    target set to all zeros (cannot be modelled)

SYNTHETIC — small seeded tables used only as controlled fixtures where the truth must be
known in advance (which column is a leak, which strong column is honest, balanced classes).
They are test inputs, not Saudi statistics, and never feed the reported forecast results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dust_pipeline.schema import OUTCOME_COLUMNS, TARGET

REPO = Path(__file__).resolve().parent.parent
SAUDI_REAL = REPO / "datasets" / "airflow" / "saudi_dust_daily_latest.csv"


def saudi_variants(real: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(7)
    ready = real.drop(columns=OUTCOME_COLUMNS)

    shuffled = ready.copy()
    shuffled[TARGET] = rng.permutation(shuffled[TARGET].to_numpy())
    shuffled["low_vis_today"] = rng.permutation(shuffled["low_vis_today"].to_numpy())  # keep it unrelated too

    dirty = ready.copy().astype({TARGET: "float"})
    dirty.loc[dirty.sample(frac=0.01, random_state=1).index, TARGET] = np.nan
    dirty.loc[dirty.sample(n=40, random_state=2).index, "wind_max_ms"] = np.inf
    dirty = pd.concat([dirty, dirty.sample(frac=0.02, random_state=3)], ignore_index=True)

    single = ready.copy()
    single[TARGET] = 0
    return {"saudi_model_ready": ready, "saudi_shuffled_target": shuffled, "saudi_dirty": dirty, "saudi_single_class": single}


def _base(n: int, seed: int, intercept: float) -> pd.DataFrame:
    """Transactions with a real, moderate signal spread over several columns."""
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(mean=8, sigma=1, size=n)
    failed_logins = rng.poisson(1.2, size=n)
    distance_km = rng.exponential(60, size=n)
    night = rng.integers(0, 2, size=n)
    channel = rng.choice(["pos", "web", "app"], size=n, p=[0.5, 0.3, 0.2])
    account_age_days = rng.integers(1, 3000, size=n)
    logit = (
        intercept
        + 1.1 * (np.log(amount) - 8)
        + 0.9 * (failed_logins - 1.2)
        + 0.012 * (distance_km - 60)
        + 0.9 * night
        + 1.0 * (channel == "web")
        - 0.0006 * (account_age_days - 1500)
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "order_id": [f"ORD{i:06d}" for i in range(n)],
            "created_at": pd.date_range("2026-01-01", periods=n, freq="7min").astype(str),
            "amount": amount.round(2),
            "failed_logins": failed_logins,
            "distance_km": distance_km.round(1),
            "night": night,
            "channel": channel,
            "account_age_days": account_age_days,
            "noise_a": rng.normal(size=n).round(3),
            "noise_b": rng.normal(size=n).round(3),
            "label": y,
        }
    )


def learnable_imbalanced() -> pd.DataFrame:
    """~6% positives, real signal."""
    return _base(6000, seed=11, intercept=-4.6)


def learnable_balanced() -> pd.DataFrame:
    """~45% positives, real signal."""
    return _base(3000, seed=12, intercept=-0.9)


def disguised_leak() -> pd.DataFrame:
    """learnable_imbalanced plus a column derived from the label, under an innocent name."""
    df = learnable_imbalanced()
    rng = np.random.default_rng(13)
    y = df["label"].to_numpy()
    df.insert(5, "risk_engine_score", np.where(y == 1, rng.uniform(0.62, 1, len(df)), rng.uniform(0, 0.38, len(df))).round(3))
    return df


def legit_strong_feature() -> pd.DataFrame:
    """Balanced table with one honest but strong predictor (must NOT be blocked as a leak)."""
    df = learnable_balanced()
    rng = np.random.default_rng(14)
    df.insert(5, "prior_chargebacks", (1.6 * df["label"] + rng.normal(scale=1.0, size=len(df))).round(2))
    return df


GENERATORS = {
    "synthetic_balanced": learnable_balanced,
    "synthetic_disguised_leak": disguised_leak,
    "synthetic_honest_strong_feature": legit_strong_feature,
}

DESCRIPTIONS = {
    "saudi_real": "Airflow output as published: 29,486 station-days, 28 stations, 3.8% events, includes tomorrow's outcome columns",
    "saudi_model_ready": "saudi_real with the outcome columns removed (no leak possible)",
    "saudi_shuffled_target": "negative control: saudi_model_ready with the target permuted (no signal)",
    "saudi_dirty": "saudi_model_ready + 1% null targets, 40 infinite values, 2% duplicated rows",
    "saudi_single_class": "saudi_model_ready with the target set to all zeros",
    "synthetic_balanced": "seeded fixture, 3,000 rows, ~45% positives, real signal",
    "synthetic_disguised_leak": "seeded fixture, ~6% positives + risk_engine_score, a column derived from the label",
    "synthetic_honest_strong_feature": "seeded fixture, balanced + prior_chargebacks, a strong but honest predictor",
}


def materialize(out_dir: Path) -> dict[str, tuple[str, str]]:
    """Write every dataset; return name → (csv path, target column)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"saudi_real": (str(SAUDI_REAL), TARGET)}
    for name, df in saudi_variants(pd.read_csv(SAUDI_REAL)).items():
        path = out_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        paths[name] = (str(path), TARGET)
    for name, gen in GENERATORS.items():
        path = out_dir / f"{name}.csv"
        gen().to_csv(path, index=False)
        paths[name] = (str(path), "label")
    return paths
