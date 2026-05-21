"""Model wrappers for survival analysis."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from lifelines import CoxPHFitter

from .exceptions import DataValidationError, ModelNotFittedError


@dataclass
class CoxModel:
    """Thin wrapper around lifelines CoxPHFitter."""

    fitter: CoxPHFitter | None = None
    duration_col: str | None = None
    event_col: str | None = None

    def fit(self, data: pd.DataFrame, duration_col: str, event_col: str) -> "CoxModel":
        if duration_col not in data.columns or event_col not in data.columns:
            raise DataValidationError("Duration/event columns are missing.")
        self.fitter = CoxPHFitter()
        self.fitter.fit(data, duration_col=duration_col, event_col=event_col)
        self.duration_col = duration_col
        self.event_col = event_col
        return self

    @property
    def summary(self) -> pd.DataFrame:
        if self.fitter is None:
            raise ModelNotFittedError("CoxModel has not been fitted yet.")
        return self.fitter.summary

    @property
    def hazard_ratios(self) -> pd.DataFrame:
        """Return a compact HR and 95% CI summary."""
        if self.fitter is None:
            raise ModelNotFittedError("CoxModel has not been fitted yet.")

        summary = self.fitter.summary
        return summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].rename(
            columns={
                "exp(coef)": "HR",
                "exp(coef) lower 95%": "CI_lower",
                "exp(coef) upper 95%": "CI_upper",
                "p": "p_value",
            }
        )

    def predict_risk_score(self, data: pd.DataFrame) -> pd.Series:
        if self.fitter is None:
            raise ModelNotFittedError("CoxModel has not been fitted yet.")
        return self.fitter.predict_partial_hazard(data).rename("risk_score")

    def check_assumptions(self, data: pd.DataFrame) -> None:
        if self.fitter is None or self.duration_col is None or self.event_col is None:
            raise ModelNotFittedError("CoxModel has not been fitted yet.")
        self.fitter.check_assumptions(data, p_value_threshold=0.05, show_plots=False)
