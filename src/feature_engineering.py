"""
feature_engineering.py
======================
Utility functions for transforming raw player statistics into
analytics-ready features. Designed for use in both notebooks and
production pipelines.

Features created:
    - Per-90-minute normalizations (goals_per90, assists_per90, etc.)
    - Composite indices (goal_contribution, defensive_index)
    - Age-based categorical buckets
    - Position group encoding
    - Log-transformed market value (for regression targets)
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# Per-90 normalizations
# ──────────────────────────────────────────────────────────────────────

def add_per90_metrics(df: pd.DataFrame, min_minutes: int = 450) -> pd.DataFrame:
    """
    Add per-90-minute columns for key counting stats.

    Players with fewer than `min_minutes` receive NaN to avoid
    misleading small-sample rates.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'minutes_played' and the counting stat columns.
    min_minutes : int
        Minimum minutes threshold (default 450 = ~5 full matches).

    Returns
    -------
    pd.DataFrame
        Copy of the input with new *_per90 columns appended.
    """
    df = df.copy()
    matches_90 = df["minutes_played"] / 90.0

    per90_cols = [
        "goals", "assists", "shots", "key_passes",
        "dribbles_completed", "tackles", "interceptions",
        "aerial_duels_won",
    ]

    for col in per90_cols:
        if col in df.columns:
            new_col = f"{col}_per90"
            df[new_col] = np.where(
                df["minutes_played"] >= min_minutes,
                (df[col] / matches_90).round(3),
                np.nan,
            )

    return df


# ──────────────────────────────────────────────────────────────────────
# Composite features
# ──────────────────────────────────────────────────────────────────────

def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create higher-level composite features from per-90 stats.

    - goal_contribution_per90: goals + assists per 90
    - defensive_index_per90:   tackles + interceptions per 90
    - creative_index_per90:    key_passes + assists per 90
    - involvement_score:       (goals + assists + key_passes) / 90-min blocks
    """
    df = df.copy()

    # Goal contribution
    if "goals_per90" in df.columns and "assists_per90" in df.columns:
        df["goal_contribution_per90"] = (
            df["goals_per90"] + df["assists_per90"]
        ).round(3)

    # Defensive index
    if "tackles_per90" in df.columns and "interceptions_per90" in df.columns:
        df["defensive_index_per90"] = (
            df["tackles_per90"] + df["interceptions_per90"]
        ).round(3)

    # Creative index
    if "key_passes_per90" in df.columns and "assists_per90" in df.columns:
        df["creative_index_per90"] = (
            df["key_passes_per90"] + df["assists_per90"]
        ).round(3)

    # Involvement score (raw count divided by 90-min blocks)
    matches_90 = df["minutes_played"] / 90.0
    if all(c in df.columns for c in ["goals", "assists", "key_passes"]):
        df["involvement_score"] = np.where(
            matches_90 > 0,
            ((df["goals"] + df["assists"] + df["key_passes"]) / matches_90).round(3),
            0.0,
        )

    return df


# ──────────────────────────────────────────────────────────────────────
# Categorical features
# ──────────────────────────────────────────────────────────────────────

def add_age_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin age into career-stage categories:
        - Young (17-21): Development phase
        - Rising (22-25): Establishing phase
        - Peak (26-29): Prime years
        - Experienced (30-33): Veteran phase
        - Twilight (34+): End of career
    """
    df = df.copy()
    bins = [16, 21, 25, 29, 33, 40]
    labels = ["Young", "Rising", "Peak", "Experienced", "Twilight"]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels, right=True)
    return df


def add_position_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map granular positions to broader groups:
        GK -> Goalkeeper
        CB, LB, RB -> Defender
        CDM, CM, CAM -> Midfielder
        LW, RW, ST -> Forward
    """
    df = df.copy()
    mapping = {
        "GK": "Goalkeeper",
        "CB": "Defender", "LB": "Defender", "RB": "Defender",
        "CDM": "Midfielder", "CM": "Midfielder", "CAM": "Midfielder",
        "LW": "Forward", "RW": "Forward", "ST": "Forward",
    }
    df["position_group"] = df["position"].map(mapping)
    return df


# ──────────────────────────────────────────────────────────────────────
# Target transformation
# ──────────────────────────────────────────────────────────────────────

def add_log_market_value(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log-transformed market value. Useful as a regression target
    because market value distributions are heavily right-skewed.
    """
    df = df.copy()
    df["log_market_value"] = np.log1p(df["market_value_eur"]).round(4)
    return df


# ──────────────────────────────────────────────────────────────────────
# Full pipeline
# ──────────────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, min_minutes: int = 450) -> pd.DataFrame:
    """
    Run the complete feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Raw player data from data_collection.
    min_minutes : int
        Minimum minutes for per-90 calculations.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with all engineered features.
    """
    df = add_per90_metrics(df, min_minutes=min_minutes)
    df = add_composite_features(df)
    df = add_age_bucket(df)
    df = add_position_group(df)
    df = add_log_market_value(df)
    return df


if __name__ == "__main__":
    # Quick demonstration
    sample = pd.DataFrame({
        "player_name": ["Demo Player"],
        "team": ["FC Barcelona"],
        "position": ["ST"],
        "age": [25],
        "minutes_played": [2700],
        "market_value_eur": [45_000_000],
        "goals": [18], "assists": [7], "shots": [72],
        "key_passes": [34], "dribbles_completed": [42],
        "tackles": [15], "interceptions": [8],
        "aerial_duels_won": [28], "yellow_cards": [4],
        "red_cards": [0], "clean_sheets": [0],
        "pass_accuracy": [81.3],
    })

    enriched = engineer_features(sample)
    print("Engineered features for demo player:")
    print(enriched.T.to_string())
