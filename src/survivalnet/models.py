"""Model wrappers for survival analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

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


@dataclass
class LassoCoxModel:
    """扩展模块：支持 L1 惩罚项的高维 LASSO-Cox 模型"""

    fitter: CoxPHFitter | None = None
    duration_col: str | None = None
    event_col: str | None = None
    penalizer: float = 0.1  # 对应惩罚项系数 lambda

    def _prepare_model_data(self, data: pd.DataFrame, duration_col: str, event_col: str) -> tuple[pd.DataFrame, list[str]]:
        """Select usable numeric features and build a clean modeling frame."""
        if duration_col not in data.columns or event_col not in data.columns:
            raise DataValidationError("Duration/event columns are missing.")

        excluded_cols = {duration_col, event_col, "PATIENT_ID", "patient_id", "ID", "id"}
        candidate_cols = [col for col in data.columns if col not in excluded_cols]
        feature_cols: list[str] = []
        numeric_features = pd.DataFrame(index=data.index)

        for col in candidate_cols:
            coerced = pd.to_numeric(data[col], errors="coerce")
            if coerced.notna().sum() == 0:
                continue
            if coerced.nunique(dropna=True) <= 1:
                continue
            feature_cols.append(col)
            numeric_features[col] = coerced

        if not feature_cols:
            raise DataValidationError("LassoCoxModel requires at least one numeric feature column.")

        model_data = pd.concat([data[[duration_col, event_col]].copy(), numeric_features], axis=1)
        model_data[duration_col] = pd.to_numeric(model_data[duration_col], errors="coerce")
        model_data[event_col] = pd.to_numeric(model_data[event_col], errors="coerce")
        model_data = model_data.dropna(axis=0, how="any")
        if len(model_data) == 0:
            raise DataValidationError("No usable rows remain after coercing model columns to numeric values.")

        unique_events = pd.Series(model_data[event_col]).dropna().unique().tolist()
        if len(unique_events) < 2:
            raise DataValidationError(
                f"Event column '{event_col}' must contain at least two classes for Cox fitting; got {unique_events}."
            )

        return model_data, feature_cols

    @staticmethod
    def _make_folds(n_samples: int, cv: int, random_state: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
        if cv < 2:
            raise ValueError("cv must be at least 2")
        if n_samples < cv:
            raise ValueError("cv cannot be greater than the number of samples")

        rng = np.random.default_rng(random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        folds = np.array_split(indices, cv)

        splits: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(cv):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(cv) if j != i])
            splits.append((train_idx, test_idx))
        return splits

    def _fit_single_penalizer(self, data: pd.DataFrame, duration_col: str, event_col: str, penalizer: float) -> CoxPHFitter:
        fitter = CoxPHFitter(penalizer=penalizer, l1_ratio=1.0)
        fitter.fit(data, duration_col=duration_col, event_col=event_col)
        return fitter

    def fit(self, data: pd.DataFrame, duration_col: str, event_col: str) -> "LassoCoxModel":
        numeric_model_data, _ = self._prepare_model_data(data, duration_col, event_col)
        print(f"开始进行 LASSO-Cox 回归，当前 penalizer (lambda) = {self.penalizer}")

        self.fitter = self._fit_single_penalizer(numeric_model_data, duration_col, event_col, self.penalizer)

        self.duration_col = duration_col
        self.event_col = event_col
        return self

    def fit_cv(
        self,
        data: pd.DataFrame,
        duration_col: str,
        event_col: str,
        penalizers: list[float] | tuple[float, ...] | None = None,
        cv: int = 5,
        random_state: int = 42,
    ) -> "LassoCoxModel":
        """Pick penalizer by cross-validation, then fit the final LASSO-Cox model."""
        if penalizers is None:
            penalizers = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]

        model_data, feature_cols = self._prepare_model_data(data, duration_col, event_col)
        splits = self._make_folds(len(model_data), cv=cv, random_state=random_state)

        best_penalizer = None
        best_score = -np.inf
        cv_results: list[dict[str, float]] = []

        for penalizer in penalizers:
            fold_scores: list[float] = []
            fold_failures = 0
            for train_idx, test_idx in splits:
                train_df = model_data.iloc[train_idx].copy()
                test_df = model_data.iloc[test_idx].copy()
                try:
                    fitter = self._fit_single_penalizer(train_df, duration_col, event_col, penalizer)
                    risk_scores = fitter.predict_partial_hazard(test_df).rename("risk_score")
                    score = concordance_index(test_df[duration_col], -risk_scores.values, test_df[event_col])
                    fold_scores.append(float(score))
                except Exception:
                    fold_failures += 1
            mean_score = float(np.mean(fold_scores)) if fold_scores else float("-inf")
            cv_results.append(
                {
                    "penalizer": float(penalizer),
                    "mean_c_index": mean_score,
                    "folds_failed": float(fold_failures),
                }
            )
            if mean_score > best_score:
                best_score = mean_score
                best_penalizer = float(penalizer)

        if best_penalizer is None:
            raise DataValidationError("Cross-validation failed for all penalizer candidates.")

        self.penalizer = best_penalizer
        self.cv_results_ = pd.DataFrame(cv_results).sort_values("mean_c_index", ascending=False).reset_index(drop=True)
        print(f"交叉验证选择的最优 penalizer (lambda) = {self.penalizer}")
        print(f"对应平均 C-index = {best_score:.4f}")
        self.fitter = self._fit_single_penalizer(model_data, duration_col, event_col, self.penalizer)
        self.duration_col = duration_col
        self.event_col = event_col
        return self

    @property
    def selected_features(self) -> list[str]:
        """关键任务：提取经 LASSO 筛选后，系数不为 0 的核心预后基因/特征"""
        if self.fitter is None:
            raise ModelNotFittedError("LassoCoxModel has not been fitted yet.")
        
        # 拿到所有特征的回归系数
        params = self.fitter.params_
        # 筛选出绝对值大于 1e-4（即没有被 LASSO 压缩到 0）的特征名
        selected = params[params.abs() > 1e-4].index.tolist()
        print(f"LASSO 筛选完成！从 {len(params)} 个原始特征中筛选出 {len(selected)} 个核心预后特征。")
        return selected

    def predict_risk_score(self, data: pd.DataFrame) -> pd.Series:
        """计算独立测试集或当前样本的风险评分 (Risk Score)"""
        if self.fitter is None:
            raise ModelNotFittedError("LassoCoxModel has not been fitted yet.")
        return self.fitter.predict_partial_hazard(data).rename("lasso_risk_score")
