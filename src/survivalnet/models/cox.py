"""Standard Cox proportional hazards model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from ..exceptions import (
    DataValidationError,
    ModelNotFittedError,
)

__all__ = ["CoxModel"]


@dataclass
class CoxModel:
    """
    Cox proportional hazards model.
    """

    penalizer: float = 0.0

    l1_ratio: float = 0.0

    fitter: CoxPHFitter | None = None

    duration_col: str | None = None

    event_col: str | None = None

    feature_cols: list[str] | None = None

    def _prepare_data(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> tuple[pd.DataFrame, list[str]]:

        if duration_col not in data.columns:
            raise DataValidationError(
                f"Missing duration column: {duration_col}"
            )

        if event_col not in data.columns:
            raise DataValidationError(
                f"Missing event column: {event_col}"
            )

        excluded_cols = {
            duration_col,
            event_col,
            "PATIENT_ID",
            "patient_id",
            "ID",
            "id",
        }

        feature_cols = []

        numeric_features = pd.DataFrame(index=data.index)

        for col in data.columns:

            if col in excluded_cols:
                continue

            values = pd.to_numeric(
                data[col],
                errors="coerce",
            )

            if values.notna().sum() == 0:
                continue

            if values.nunique(dropna=True) <= 1:
                continue

            numeric_features[col] = values

            feature_cols.append(col)

        if len(feature_cols) == 0:
            raise DataValidationError(
                "No usable numeric features found."
            )

        model_data = pd.concat(
            [
                data[[duration_col, event_col]],
                numeric_features,
            ],
            axis=1,
        )

        model_data = model_data.dropna()

        if len(model_data) == 0:
            raise DataValidationError(
                "No valid rows remain after cleaning."
            )

        return model_data, feature_cols

    def fit(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "CoxModel":

        model_data, feature_cols = self._prepare_data(
            data,
            duration_col,
            event_col,
        )

        self.feature_cols = feature_cols

        self.duration_col = duration_col

        self.event_col = event_col

        self.fitter = CoxPHFitter(
            penalizer=self.penalizer,
            l1_ratio=self.l1_ratio,
        )

        self.fitter.fit(
            model_data,
            duration_col=duration_col,
            event_col=event_col,
        )

        return self

    @property
    def summary(self) -> pd.DataFrame:

        if self.fitter is None:
            raise ModelNotFittedError(
                "CoxModel has not been fitted yet."
            )

        return self.fitter.summary

    @property
    def hazard_ratios(self) -> pd.DataFrame:

        if self.fitter is None:
            raise ModelNotFittedError(
                "CoxModel has not been fitted yet."
            )

        summary = self.fitter.summary

        return summary[
            [
                "exp(coef)",
                "exp(coef) lower 95%",
                "exp(coef) upper 95%",
                "p",
            ]
        ].rename(
            columns={
                "exp(coef)": "HR",
                "exp(coef) lower 95%": "CI_lower",
                "exp(coef) upper 95%": "CI_upper",
                "p": "p_value",
            }
        )

    def predict_risk_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        if self.fitter is None:
            raise ModelNotFittedError(
                "CoxModel has not been fitted yet."
            )

        if self.feature_cols is None:
            raise ModelNotFittedError(
                "Feature columns are unavailable."
            )

        X = data[self.feature_cols]

        return self.fitter.predict_partial_hazard(
            X
        ).rename(
            "risk_score"
        )

    def score(
        self,
        data: pd.DataFrame,
    ) -> float:

        if self.duration_col is None:
            raise ModelNotFittedError(
                "Model has not been fitted."
            )

        if self.event_col is None:
            raise ModelNotFittedError(
                "Model has not been fitted."
            )

        risk_scores = self.predict_risk_score(
            data
        )

        return float(
            concordance_index(
                data[self.duration_col],
                -risk_scores,
                data[self.event_col],
            )
        )

    def check_assumptions(
        self,
        data: pd.DataFrame,
    ) -> None:

        if self.fitter is None:
            raise ModelNotFittedError(
                "CoxModel has not been fitted yet."
            )

        self.fitter.check_assumptions(
            data,
            p_value_threshold=0.05,
            show_plots=False,
        )