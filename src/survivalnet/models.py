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


@dataclass
class LassoCoxModel:
    """扩展模块：支持 L1 惩罚项的高维 LASSO-Cox 模型"""

    fitter: CoxPHFitter | None = None
    duration_col: str | None = None
    event_col: str | None = None
    penalizer: float = 0.1  # 对应惩罚项系数 lambda

    def fit(self, data: pd.DataFrame, duration_col: str, event_col: str) -> "LassoCoxModel":
        if duration_col not in data.columns or event_col not in data.columns:
            raise DataValidationError("Duration/event columns are missing.")
            
        print(f"开始进行 LASSO-Cox 回归，当前 penalizer (lambda) = {self.penalizer}")
        
        # 核心变动：实例化时加上 penalizer 和 l1_ratio=1.0（1.0 代表纯 LASSO 惩罚）
        self.fitter = CoxPHFitter(penalizer=self.penalizer, l1_ratio=1.0)
        self.fitter.fit(data, duration_col=duration_col, event_col=event_col)
        
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