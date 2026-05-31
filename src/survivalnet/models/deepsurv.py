"""DeepSurv neural-network survival model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from pycox.models import CoxPH

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from lifelines.utils import concordance_index

from torchtuples import optim
import torchtuples as tt

from ..exceptions import (
    DataValidationError,
    ModelNotFittedError,
)


@dataclass
class DeepSurvModel:
    """
    DeepSurv survival model based on pycox.
    """

    hidden_dim1: int = 64
    hidden_dim2: int = 32

    dropout: float = 0.2

    learning_rate: float = 1e-3

    batch_size: int = 64

    epochs: int = 100

    validation_split: float = 0.2

    random_state: int = 42

    model: CoxPH | None = None

    scaler: StandardScaler | None = None

    duration_col: str | None = None

    event_col: str | None = None

    feature_cols: list[str] | None = None

    def _prepare_data(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ):

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

        return model_data, feature_cols

    def fit(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
    ) -> "DeepSurvModel":

        torch.manual_seed(self.random_state)

        np.random.seed(self.random_state)

        model_data, feature_cols = self._prepare_data(
            data,
            duration_col,
            event_col,
        )

        self.feature_cols = feature_cols

        self.duration_col = duration_col

        self.event_col = event_col

        X = model_data[feature_cols].values.astype("float32")

        y_time = model_data[duration_col].values

        y_event = model_data[event_col].values

        self.scaler = StandardScaler()

        X = self.scaler.fit_transform(X)

        (
            X_train,
            X_val,
            y_time_train,
            y_time_val,
            y_event_train,
            y_event_val,
        ) = train_test_split(
            X,
            y_time,
            y_event,
            test_size=self.validation_split,
            random_state=self.random_state,
        )

        num_features = X.shape[1]

        net = torch.nn.Sequential(

            torch.nn.Linear(
                num_features,
                self.hidden_dim1,
            ),

            torch.nn.ReLU(),

            torch.nn.BatchNorm1d(
                self.hidden_dim1
            ),

            torch.nn.Dropout(
                self.dropout
            ),

            torch.nn.Linear(
                self.hidden_dim1,
                self.hidden_dim2,
            ),

            torch.nn.ReLU(),

            torch.nn.BatchNorm1d(
                self.hidden_dim2
            ),

            torch.nn.Dropout(
                self.dropout
            ),

            torch.nn.Linear(
                self.hidden_dim2,
                1,
            ),
        )

        self.model = CoxPH(
            net,
            optim.Adam,
        )

        self.model.optimizer.set_lr(
            self.learning_rate
        )

        callbacks = [
            tt.callbacks.EarlyStopping()
        ]

        self.model.fit(
            X_train,
            (
                y_time_train,
                y_event_train,
            ),
            batch_size=self.batch_size,
            epochs=self.epochs,
            callbacks=callbacks,
            val_data=(
                X_val,
                (
                    y_time_val,
                    y_event_val,
                ),
            ),
            verbose=False,
        )

        return self

    def predict_risk_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        if self.model is None:
            raise ModelNotFittedError(
                "DeepSurvModel has not been fitted yet."
            )

        if self.scaler is None:
            raise ModelNotFittedError(
                "Scaler has not been fitted."
            )

        if self.feature_cols is None:
            raise ModelNotFittedError(
                "Feature columns are unavailable."
            )

        X = data[
            self.feature_cols
        ].values.astype("float32")

        X = self.scaler.transform(X)

        risk_scores = self.model.predict(X)

        return pd.Series(
            risk_scores.flatten(),
            index=data.index,
            name="deep_risk_score",
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