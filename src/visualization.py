"""
visualization.py
================
Reusable plotting functions for football analytics. Built on top of
Matplotlib and Seaborn with a consistent visual style inspired by
modern football data journalism (The Athletic, StatsBomb).

All functions return (fig, ax) tuples so callers can further
customize or save the plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# Global style configuration
# ──────────────────────────────────────────────────────────────────────

PALETTE = {
    "primary": "#1a1a2e",
    "accent": "#e94560",
    "secondary": "#0f3460",
    "light": "#16213e",
    "bg": "#f5f5f5",
    "grid": "#dddddd",
}

POSITION_COLORS = {
    "Goalkeeper": "#FFC107",
    "Defender": "#2196F3",
    "Midfielder": "#4CAF50",
    "Forward": "#F44336",
}


def set_football_style():
    """Apply a clean, publication-ready matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "#ffffff",
        "axes.edgecolor": PALETTE["grid"],
        "axes.grid": True,
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
    })


# ──────────────────────────────────────────────────────────────────────
# Distribution plots
# ──────────────────────────────────────────────────────────────────────

def plot_market_value_distribution(df: pd.DataFrame, log_scale: bool = True):
    """
    Histogram of market values with optional log scale.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'market_value_eur'.
    log_scale : bool
        If True, plot on log10 scale (recommended due to skew).
    """
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    values = df["market_value_eur"] / 1e6  # Convert to millions
    if log_scale:
        values = np.log10(values.clip(lower=0.01))
        xlabel = "Market Value (log10 EUR millions)"
    else:
        xlabel = "Market Value (EUR millions)"

    ax.hist(values, bins=40, color=PALETTE["accent"], edgecolor="white",
            alpha=0.85, zorder=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Players")
    ax.set_title("LaLiga 2024-25: Market Value Distribution")

    plt.tight_layout()
    return fig, ax


def plot_age_distribution(df: pd.DataFrame):
    """KDE + histogram of player ages colored by position group."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    if "position_group" in df.columns:
        for group, color in POSITION_COLORS.items():
            subset = df[df["position_group"] == group]["age"]
            if len(subset) > 0:
                ax.hist(subset, bins=range(17, 40), alpha=0.4,
                        label=group, color=color, edgecolor="white")
    else:
        ax.hist(df["age"], bins=range(17, 40), color=PALETTE["secondary"],
                edgecolor="white", alpha=0.8)

    ax.set_xlabel("Age")
    ax.set_ylabel("Number of Players")
    ax.set_title("Age Distribution by Position Group")
    ax.legend(frameon=True, facecolor="white")
    plt.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Scatter plots
# ──────────────────────────────────────────────────────────────────────

def plot_goals_vs_market_value(df: pd.DataFrame, per90: bool = True):
    """
    Scatter plot of goal-scoring metric vs market value, colored by
    position group.

    Parameters
    ----------
    per90 : bool
        If True, use goals_per90 (requires feature engineering).
        If False, use raw goals.
    """
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    x_col = "goals_per90" if per90 else "goals"
    y_col = "market_value_eur"

    plot_df = df.dropna(subset=[x_col])

    if "position_group" in plot_df.columns:
        for group, color in POSITION_COLORS.items():
            subset = plot_df[plot_df["position_group"] == group]
            ax.scatter(subset[x_col], subset[y_col] / 1e6,
                       alpha=0.6, s=40, label=group, color=color,
                       edgecolors="white", linewidths=0.3, zorder=3)
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col] / 1e6,
                   alpha=0.6, s=40, color=PALETTE["accent"],
                   edgecolors="white", linewidths=0.3, zorder=3)

    xlabel = "Goals per 90 minutes" if per90 else "Total Goals"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Market Value (EUR millions)")
    ax.set_title(f"{xlabel} vs Market Value")
    ax.legend(frameon=True, facecolor="white")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    plt.tight_layout()
    return fig, ax


def plot_performance_vs_value(df: pd.DataFrame, x_col: str, y_col: str = "market_value_eur",
                               title: str = None, xlabel: str = None):
    """Generic scatter: any metric vs market value."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    plot_df = df.dropna(subset=[x_col])

    if "position_group" in plot_df.columns:
        for group, color in POSITION_COLORS.items():
            subset = plot_df[plot_df["position_group"] == group]
            ax.scatter(subset[x_col], subset[y_col] / 1e6,
                       alpha=0.6, s=40, label=group, color=color,
                       edgecolors="white", linewidths=0.3, zorder=3)
        ax.legend(frameon=True, facecolor="white")
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col] / 1e6,
                   alpha=0.6, s=40, color=PALETTE["accent"],
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel("Market Value (EUR millions)")
    ax.set_title(title or f"{x_col} vs Market Value")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    plt.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Correlation heatmap
# ──────────────────────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, columns: list = None,
                              figsize: tuple = (12, 10)):
    """
    Correlation heatmap for numeric features.

    Parameters
    ----------
    columns : list, optional
        Specific columns to include. If None, all numeric columns are used.
    """
    set_football_style()
    fig, ax = plt.subplots(figsize=figsize)

    if columns:
        corr = df[columns].corr()
    else:
        corr = df.select_dtypes(include=[np.number]).corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8, "label": "Pearson r"})

    ax.set_title("Feature Correlation Matrix", pad=20)
    plt.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Box plots
# ──────────────────────────────────────────────────────────────────────

def plot_value_by_position(df: pd.DataFrame):
    """Box plot of market value by position group."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    group_col = "position_group" if "position_group" in df.columns else "position"
    order = ["Goalkeeper", "Defender", "Midfielder", "Forward"] if group_col == "position_group" else None

    palette = POSITION_COLORS if group_col == "position_group" else "Set2"

    sns.boxplot(data=df, x=group_col, y=df["market_value_eur"] / 1e6,
                order=order, palette=palette, ax=ax,
                flierprops={"alpha": 0.4, "markersize": 3})

    ax.set_xlabel("Position Group")
    ax.set_ylabel("Market Value (EUR millions)")
    ax.set_title("Market Value Distribution by Position")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    plt.tight_layout()
    return fig, ax


def plot_value_by_age_bucket(df: pd.DataFrame):
    """Box plot of market value by age bucket."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    order = ["Young", "Rising", "Peak", "Experienced", "Twilight"]
    colors = ["#66BB6A", "#42A5F5", "#EF5350", "#FFA726", "#AB47BC"]

    sns.boxplot(data=df, x="age_bucket", y=df["market_value_eur"] / 1e6,
                order=order, palette=colors, ax=ax,
                flierprops={"alpha": 0.4, "markersize": 3})

    ax.set_xlabel("Career Stage")
    ax.set_ylabel("Market Value (EUR millions)")
    ax.set_title("Market Value by Career Stage")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    plt.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Model evaluation plots
# ──────────────────────────────────────────────────────────────────────

def plot_feature_importance(feature_names: list, importances: np.ndarray,
                            top_n: int = 15, title: str = "Feature Importance"):
    """Horizontal bar chart of feature importances."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 7))

    # Sort and take top N
    idx = np.argsort(importances)[-top_n:]
    ax.barh(range(len(idx)), importances[idx],
            color=PALETTE["accent"], edgecolor="white", zorder=3)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = "Predicted vs Actual Market Value"):
    """Scatter plot of predicted vs actual values with a 45-degree reference line."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(y_true / 1e6, y_pred / 1e6, alpha=0.5, s=30,
               color=PALETTE["secondary"], edgecolors="white",
               linewidths=0.3, zorder=3)

    # 45-degree line
    lims = [0, max(y_true.max(), y_pred.max()) / 1e6 * 1.05]
    ax.plot(lims, lims, "--", color=PALETTE["accent"], linewidth=2,
            alpha=0.8, label="Perfect prediction", zorder=4)

    ax.set_xlabel("Actual Market Value (EUR millions)")
    ax.set_ylabel("Predicted Market Value (EUR millions)")
    ax.set_title(title)
    ax.legend(frameon=True, facecolor="white")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    return fig, ax


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray):
    """Residual plot: predicted value on x-axis, residual on y-axis."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 5))

    residuals = (y_true - y_pred) / 1e6
    ax.scatter(y_pred / 1e6, residuals, alpha=0.4, s=25,
               color=PALETTE["secondary"], edgecolors="white",
               linewidths=0.3, zorder=3)
    ax.axhline(0, color=PALETTE["accent"], linewidth=2, linestyle="--", zorder=4)

    ax.set_xlabel("Predicted Market Value (EUR millions)")
    ax.set_ylabel("Residual (EUR millions)")
    ax.set_title("Residual Analysis")

    plt.tight_layout()
    return fig, ax


# ──────────────────────────────────────────────────────────────────────
# Team-level plots
# ──────────────────────────────────────────────────────────────────────

def plot_team_squad_value(df: pd.DataFrame, top_n: int = 20):
    """Horizontal bar chart of total squad market value by team."""
    set_football_style()
    fig, ax = plt.subplots(figsize=(10, 8))

    team_values = (
        df.groupby("team")["market_value_eur"]
        .sum()
        .sort_values(ascending=True)
        .tail(top_n)
        / 1e6
    )

    colors = [PALETTE["accent"] if v > team_values.median() else PALETTE["secondary"]
              for v in team_values.values]

    ax.barh(team_values.index, team_values.values, color=colors,
            edgecolor="white", zorder=3)
    ax.set_xlabel("Total Squad Value (EUR millions)")
    ax.set_title("LaLiga Squad Market Values")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}M"))

    plt.tight_layout()
    return fig, ax
