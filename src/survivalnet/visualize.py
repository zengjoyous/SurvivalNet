"""Plotting helpers for survival analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .core import fit_km, logrank_p_value


def plot_km_curve(km_fitter, label: str | None = None, ax=None):
    """Plot a Kaplan-Meier curve."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    km_fitter.plot_survival_function(ax=ax, label=label)
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    return ax


def plot_grouped_km(data: pd.DataFrame, duration_col: str, event_col: str, group_col: str, ax=None):
    """Plot Kaplan-Meier curves for each group."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for group_name, subset in data.groupby(group_col):
        kmf = fit_km(subset, duration_col=duration_col, event_col=event_col)
        kmf.plot_survival_function(ax=ax, label=str(group_name))
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, alpha=0.3)
    return ax


def plot_grouped_km_with_pvalue(
    data: pd.DataFrame,
    duration_col: str,
    event_col: str,
    group_col: str,
    group_a,
    group_b,
    ax=None,
):
    """Plot grouped KM curves and annotate the log-rank p-value."""
    ax = plot_grouped_km(data, duration_col=duration_col, event_col=event_col, group_col=group_col, ax=ax)
    p_value = logrank_p_value(data, group_col, duration_col, event_col, group_a, group_b)
    ax.text(0.98, 0.05, f"p = {p_value:.3g}", transform=ax.transAxes, ha="right", va="bottom")
    return ax
