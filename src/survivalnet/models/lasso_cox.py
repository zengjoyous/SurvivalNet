"""LASSO-regularized Cox proportional hazards model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from ..exceptions import (
    DataValidationError,
    ModelNotFittedError,
)


@dataclass
class LassoCoxModel:
    """
    LASSO-Cox proportional hazards model.

    Parameters
    ----------
    penalizer : float
        L1 penalty coefficient (lambda).
    cv : int
        Number of folds for cross-validation.
    random_state : int
        Random seed for reproducibility.
    """

    penalizer: float = 0.1

    cv: int = 5

    random_state: int = 42

    fitter: CoxPHFitter | None = None

    duration_col: str | None = None

    event_col: str | None = None

    feature_cols: list[str] | None = None

    def _prepare_model_data(
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

        feature_cols: list[str] = []

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

        model_data[duration_col] = pd.to_numeric(
            model_data[duration_col],
            errors="coerce",
        )

        model_data[event_col] = pd.to_numeric(
            model_data[event_col],
            errors="coerce",
        )

        model_data = model_data.dropna()

        if len(model_data) == 0:
            raise DataValidationError(
                "No usable rows remain after cleaning."
            )

        unique_events = (
            pd.Series(model_data[event_col])
            .dropna()
            .unique()
            .tolist()
        )

        if len(unique_events) < 2:
            raise DataValidationError(
                f"Event column '{event_col}' must contain at least two classes; "
                f"got {unique_events}."
            )

        return model_data, feature_cols

    @staticmethod
    def _make_folds(
        n_samples: int,
        cv: int,
        random_state: int = 42,
    ) -> list[tuple[np.ndarray, np.ndarray]]:

        if cv < 2:
            raise ValueError(
                "cv must be at least 2."
            )

        if n_samples < cv:
            raise ValueError(
                "cv cannot exceed sample size."
            )

        rng = np.random.default_rng(
            random_state
        )

        indices = np.arange(
            n_samples
        )

        rng.shuffle(
            indices
        )

        folds = np.array_split(
            indices,
            cv,
        )

        splits = []

        for i in range(cv):

            test_idx = folds[i]

            train_idx = np.concatenate(
                [
                    folds[j]
                    for j in range(cv)
                    if j != i
                ]
            )

            splits.append(
                (
                    train_idx,
                    test_idx,
                )
            )

        return splits

    def _fit_single_penalizer(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
        penalizer: float,
    ) -> CoxPHFitter:

        fitter = CoxPHFitter(
            penalizer=penalizer,
            l1_ratio=1.0,
        )

        fitter.fit(
            data,
            duration_col=duration_col,
            event_col=event_col,
        )

        return fitter

    def fit(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "LassoCoxModel":

        model_data, feature_cols = (
            self._prepare_model_data(
                data,
                duration_col,
                event_col,
            )
        )

        self.feature_cols = (
            feature_cols
        )

        self.fitter = (
            self._fit_single_penalizer(
                model_data,
                duration_col,
                event_col,
                self.penalizer,
            )
        )

        self.duration_col = (
            duration_col
        )

        self.event_col = (
            event_col
        )

        return self

    def fit_cv(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
        penalizers: list[float] | None = None,
    ) -> "LassoCoxModel":

        if penalizers is None:

            penalizers = [
                0.001,
                0.005,
                0.01,
                0.02,
                0.05,
                0.1,
            ]

        model_data, feature_cols = (
            self._prepare_model_data(
                data,
                duration_col,
                event_col,
            )
        )

        self.feature_cols = (
            feature_cols
        )

        splits = self._make_folds(
            len(model_data),
            cv=self.cv,
            random_state=self.random_state,
        )

        best_penalizer = None

        best_score = -np.inf

        cv_results = []

        for penalizer in penalizers:

            fold_scores = []

            failed_folds = 0

            for train_idx, test_idx in splits:

                train_df = model_data.iloc[
                    train_idx
                ]

                test_df = model_data.iloc[
                    test_idx
                ]

                try:

                    fitter = (
                        self._fit_single_penalizer(
                            train_df,
                            duration_col,
                            event_col,
                            penalizer,
                        )
                    )

                    risk_scores = (
                        fitter.predict_partial_hazard(
                            test_df
                        )
                    )

                    score = concordance_index(
                        test_df[duration_col],
                        -risk_scores.values,
                        test_df[event_col],
                    )

                    fold_scores.append(
                        float(score)
                    )

                except Exception:

                    failed_folds += 1

            mean_score = (
                float(np.mean(fold_scores))
                if fold_scores
                else float("-inf")
            )

            cv_results.append(
                {
                    "penalizer": penalizer,
                    "mean_c_index": mean_score,
                    "failed_folds": failed_folds,
                }
            )

            if mean_score > best_score:

                best_score = mean_score

                best_penalizer = penalizer

        if best_penalizer is None:

            raise DataValidationError(
                "Cross-validation failed."
            )

        self.penalizer = (
            best_penalizer
        )

        self.cv_results_ = (
            pd.DataFrame(cv_results)
            .sort_values(
                "mean_c_index",
                ascending=False,
            )
            .reset_index(
                drop=True
            )
        )

        self.fitter = (
            self._fit_single_penalizer(
                model_data,
                duration_col,
                event_col,
                self.penalizer,
            )
        )

        self.duration_col = (
            duration_col
        )

        self.event_col = (
            event_col
        )

        return self

    @property
    def summary(self) -> pd.DataFrame:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
            )

        return self.fitter.summary

    @property
    def hazard_ratios(self) -> pd.DataFrame:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
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

    @property
    def selected_features(
        self,
    ) -> list[str]:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
            )

        params = self.fitter.params_

        return params[
            params.abs() > 1e-4
        ].index.tolist()

    def predict_risk_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        if self.fitter is None:

            raise ModelNotFittedError(
                "LassoCoxModel has not been fitted yet."
            )

        if self.feature_cols is None:

            raise ModelNotFittedError(
                "Feature columns unavailable."
            )

        return (
            self.fitter
            .predict_partial_hazard(
                data[self.feature_cols]
            )
            .rename(
                "lasso_risk_score"
            )
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

        risk_scores = (
            self.predict_risk_score(
                data
            )
        )

        return float(
            concordance_index(
                data[self.duration_col],
                -risk_scores,
                data[self.event_col],
            )
        )