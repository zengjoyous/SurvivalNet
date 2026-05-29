"""DeepSurv neural-network survival model."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import torch
from pycox.models import CoxPH
from sklearn.preprocessing import StandardScaler
from torchtuples import optim

from ..exceptions import (
    DataValidationError,
    ModelNotFittedError,
)


@dataclass
class DeepSurvModel:

    model: CoxPH | None = None
    scaler: StandardScaler | None = None

    duration_col: str | None = None
    event_col: str | None = None

    def fit(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
        epochs: int = 100,
    ) -> "DeepSurvModel":

        if duration_col not in data.columns or event_col not in data.columns:
            raise DataValidationError("Duration/event columns are missing.")

        self.duration_col = duration_col
        self.event_col = event_col

        X = data.drop(columns=[duration_col, event_col]).values.astype("float32")

        y_time = data[duration_col].values
        y_event = data[event_col].values

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        num_features = X.shape[1]

        net = torch.nn.Sequential(
            torch.nn.Linear(num_features, 64),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(64),

            torch.nn.Dropout(0.2),

            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),

            torch.nn.Dropout(0.2),

            torch.nn.Linear(32, 1),
        )

        self.model = CoxPH(
            net,
            optim.Adam,
        )

        self.model.optimizer.set_lr(1e-3)

        self.model.fit(
            X,
            (y_time, y_event),
            batch_size=64,
            epochs=epochs,
            verbose=False,
        )

        return self

    def predict_risk_score(
        self,
        data: pd.DataFrame,
    ) -> pd.Series:

        if self.model is None or self.scaler is None:
            raise ModelNotFittedError("DeepSurvModel has not been fitted yet.")

        X = data.values.astype("float32")

        X = self.scaler.transform(X)

        risk_scores = self.model.predict(X)

        return pd.Series(
            risk_scores.flatten(),
            name="deep_risk_score",
        )