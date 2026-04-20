"""
Kaplan-Meier survival curves and risk stratification visualization.

Key tools for evaluating survival models in clinical contexts:
- KM Estimator: Non-parametric step-function estimate of survival probability.
- Log-rank test: Non-parametric test for equality of survival distributions.
- Risk stratification: Divides patients by predicted risk score (high vs low).

The KM + log-rank workflow is the clinical standard for validating
prognostic signatures. A statistically significant separation (p < 0.05)
between risk groups demonstrates the model's clinical utility.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from typing import Optional, Tuple


def risk_stratify(
    risk_scores: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    cutoff: str = "median",
) -> Tuple[dict, dict]:
    """
    Stratify patients into high- and low-risk groups.

    Parameters
    ----------
    risk_scores : np.ndarray
        Predicted risk scores (higher = worse prognosis).
    durations : np.ndarray
        Observed survival times.
    events : np.ndarray
        Event indicators (1=event occurred, 0=censored).
    cutoff : str or float
        'median' to split at median risk score, or a float threshold.

    Returns
    -------
    high_risk : dict with keys 'durations', 'events'
    low_risk  : dict with keys 'durations', 'events'
    """
    threshold = np.median(risk_scores) if cutoff == "median" else float(cutoff)
    high_mask = risk_scores >= threshold
    low_mask = ~high_mask

    high_risk = {"durations": durations[high_mask], "events": events[high_mask]}
    low_risk = {"durations": durations[low_mask], "events": events[low_mask]}
    return high_risk, low_risk


def plot_kaplan_meier(
    high_risk: dict,
    low_risk: dict,
    title: str = "Kaplan-Meier Survival Curves",
    time_unit: str = "Months",
    figsize: Tuple[int, int] = (9, 6),
    ci_show: bool = True,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot Kaplan-Meier curves for high- and low-risk groups with log-rank test.

    The KM estimator:
        S(t) = Π_{t_i ≤ t} (1 - d_i / n_i)

    where d_i is the number of events and n_i is the risk set size at t_i.
    95% CI computed via Greenwood's formula.

    Parameters
    ----------
    high_risk : dict
        {'durations': array, 'events': array} for high-risk group.
    low_risk : dict
        {'durations': array, 'events': array} for low-risk group.
    title : str
        Plot title.
    time_unit : str
        X-axis label. Default 'Months'.
    ci_show : bool
        Whether to shade 95% confidence intervals. Default True.
    save_path : str, optional
        Save figure to this file path if provided.

    Returns
    -------
    matplotlib Figure object.
    """
    # Log-rank test for group difference
    result = logrank_test(
        high_risk["durations"],
        low_risk["durations"],
        event_observed_A=high_risk["events"],
        event_observed_B=low_risk["events"],
    )
    p_value = result.p_value

    fig, ax = plt.subplots(figsize=figsize)

    # High-risk group
    kmf_high = KaplanMeierFitter()
    kmf_high.fit(
        high_risk["durations"],
        event_observed=high_risk["events"],
        label=f"High Risk (n={len(high_risk['durations'])})",
    )
    kmf_high.plot_survival_function(
        ax=ax, ci_show=ci_show, color="#d62728", linewidth=2
    )

    # Low-risk group
    kmf_low = KaplanMeierFitter()
    kmf_low.fit(
        low_risk["durations"],
        event_observed=low_risk["events"],
        label=f"Low Risk (n={len(low_risk['durations'])})",
    )
    kmf_low.plot_survival_function(
        ax=ax, ci_show=ci_show, color="#1f77b4", linewidth=2
    )

    # Annotate with log-rank p-value
    p_text = f"p = {p_value:.4f}" if p_value >= 0.0001 else "p < 0.0001"
    ax.annotate(
        p_text,
        xy=(0.65, 0.85),
        xycoords="axes fraction",
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Time ({time_unit})", fontsize=12)
    ax.set_ylabel("Survival Probability S(t)", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cumulative_hazard(
    high_risk: dict,
    low_risk: dict,
    title: str = "Cumulative Hazard",
    time_unit: str = "Months",
    figsize: Tuple[int, int] = (9, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Nelson-Aalen cumulative hazard curves for both risk groups."""
    from lifelines import NelsonAalenFitter

    fig, ax = plt.subplots(figsize=figsize)

    for group_data, label, color in [
        (high_risk, "High Risk", "#d62728"),
        (low_risk, "Low Risk", "#1f77b4"),
    ]:
        naf = NelsonAalenFitter()
        n = len(group_data["durations"])
        naf.fit(
            group_data["durations"],
            event_observed=group_data["events"],
            label=f"{label} (n={n})",
        )
        naf.plot_cumulative_hazard(ax=ax, color=color, linewidth=2)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"Time ({time_unit})", fontsize=12)
    ax.set_ylabel("Cumulative Hazard H(t)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_c_index_comparison(
    model_names: list,
    c_indices: list,
    errors: Optional[list] = None,
    figsize: Tuple[int, int] = (8, 5),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Bar chart comparing C-index values across multiple survival models.

    Useful for showing DeepSurv vs Cox performance side-by-side.

    Parameters
    ----------
    model_names : list of str
    c_indices : list of float
    errors : list of float, optional
        Standard deviation for error bars (e.g., from cross-validation).
    """
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.Blues(np.linspace(0.4, 0.85, len(model_names)))
    bars = ax.bar(model_names, c_indices, color=colors, edgecolor="black", linewidth=0.8)

    if errors:
        ax.errorbar(
            range(len(model_names)), c_indices, yerr=errors,
            fmt="none", color="black", capsize=5, linewidth=1.5
        )

    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (C=0.5)")
    ax.set_ylabel("C-index (Concordance Index)", fontsize=12)
    ax.set_title("Survival Model Comparison — C-index", fontsize=13, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars, c_indices):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold",
        )

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
