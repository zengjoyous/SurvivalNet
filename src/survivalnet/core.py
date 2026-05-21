"""Core survival analysis utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

from .exceptions import DataValidationError


def _validate_survival_frame(data: pd.DataFrame, duration_col: str, event_col: str) -> None:
    missing = [col for col in (duration_col, event_col) if col not in data.columns]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")


def fit_km(data: pd.DataFrame, duration_col: str, event_col: str) -> KaplanMeierFitter:
    """Fit a Kaplan-Meier estimator."""
    _validate_survival_frame(data, duration_col, event_col)
    kmf = KaplanMeierFitter()
    kmf.fit(durations=data[duration_col], event_observed=data[event_col])
    return kmf


def run_logrank_test(
    data: pd.DataFrame,
    group_col: str,
    duration_col: str,
    event_col: str,
    group_a,
    group_b,
):
    """Run a log-rank test between two groups."""
    _validate_survival_frame(data, duration_col, event_col)
    if group_col not in data.columns:
        raise DataValidationError(f"Missing required column: {group_col}")

    group1 = data[data[group_col] == group_a]
    group2 = data[data[group_col] == group_b]
    if group1.empty or group2.empty:
        raise DataValidationError("Both groups must contain at least one sample.")

    return logrank_test(
        group1[duration_col],
        group2[duration_col],
        event_observed_A=group1[event_col],
        event_observed_B=group2[event_col],
    )


def logrank_p_value(
    data: pd.DataFrame,
    group_col: str,
    duration_col: str,
    event_col: str,
    group_a,
    group_b,
) -> float:
    """Return the p-value from a log-rank test."""
    return float(run_logrank_test(data, group_col, duration_col, event_col, group_a, group_b).p_value)


def c_index(event_col: pd.Series, duration_col: pd.Series, risk_scores: Iterable[float]) -> float:
    """Compute the concordance index."""
    return concordance_index(duration_col, -np.asarray(list(risk_scores)), event_col)


def split_risk_group(
    scores: pd.Series | np.ndarray,
    method: str = "median",
    cutoff: float | None = None,
) -> pd.Series:
    """Split scores into high/low risk groups."""
    scores = pd.Series(scores, name="risk_score")
    if method == "median":
        threshold = float(scores.median())
    elif method == "cutoff":
        if cutoff is None:
            raise ValueError("cutoff must be provided when method='cutoff'")
        threshold = float(cutoff)
    else:
        raise ValueError("method must be 'median' or 'cutoff'")

    return pd.Series(np.where(scores >= threshold, "high", "low"), index=scores.index, name="risk_group")


@dataclass
class KaplanMeierResult:
    fitter: KaplanMeierFitter
