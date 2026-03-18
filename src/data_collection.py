"""
data_collection.py
==================
Loads and preprocesses the LaLiga 2024-25 player dataset sourced from
Transfermarkt via the transfermarkt-datasets project (davidcariboo/player-scores).

The dataset contains real player statistics, market valuations, and metadata
for all players who appeared in LaLiga during the 2024-25 season.

Data source: https://github.com/dcaribou/transfermarkt-datasets
License: CC BY-SA 4.0

Usage:
    from src.data_collection import load_dataset
    df = load_dataset()
"""

import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "laliga_players_2024_25.csv")

# Shortened team names for readability
# Includes variants with different whitespace from Transfermarkt source
TEAM_NAME_MAP = {
    "Real Madrid Club de Fútbol": "Real Madrid",
    "Futbol Club Barcelona": "FC Barcelona",
    "Club Atlético de Madrid S.A.D.": "Atletico Madrid",
    "Real Sociedad de Fútbol S.A.D.": "Real Sociedad",
    "Athletic Club Bilbao": "Athletic Bilbao",
    "Real Betis Balompié S.A.D.": "Real Betis",
    "Villarreal Club de Fútbol S.A.D.": "Villarreal",
    "Girona Fútbol Club S.A.D.": "Girona",
    "Girona Fútbol Club S. A. D.": "Girona",
    "Sevilla Fútbol Club S.A.D.": "Sevilla",
    "Valencia Club de Fútbol S.A.D.": "Valencia",
    "Valencia Club de Fútbol S. A. D.": "Valencia",
    "Club Atlético Osasuna": "Osasuna",
    "Real Club Celta de Vigo S.A.D.": "Celta Vigo",
    "Real Club Celta de Vigo S. A. D.": "Celta Vigo",
    "Real Club Deportivo Mallorca S.A.D.": "Mallorca",
    "Getafe Club de Fútbol S.A.D.": "Getafe",
    "Getafe Club de Fútbol S. A. D. Team Dubai": "Getafe",
    "Rayo Vallecano de Madrid S.A.D.": "Rayo Vallecano",
    "Rayo Vallecano de Madrid S. A. D.": "Rayo Vallecano",
    "Unión Deportiva Las Palmas S.A.D.": "Las Palmas",
    "UD Las Palmas": "Las Palmas",
    "Deportivo Alavés S.A.D.": "Alaves",
    "Deportivo Alavés S. A. D.": "Alaves",
    "Reial Club Deportiu Espanyol de Barcelona S.A.D.": "Espanyol",
    "Real Valladolid Club de Fútbol S.A.D.": "Real Valladolid",
    "Real Valladolid CF": "Real Valladolid",
    "CD Leganés": "Leganes",
    "Real Club Deportivo de La Coruña": "Deportivo",
    "Aberdeen Football Club": "Aberdeen (loan)",
}


# LaLiga 2024-25 final standings (computed from Transfermarkt match results)
# Used as a leakage-free team prestige proxy in the predictive model
LALIGA_2024_STANDINGS = {
    "FC Barcelona": 88, "Real Madrid": 84, "Atletico Madrid": 76,
    "Athletic Bilbao": 70, "Villarreal": 70, "Real Betis": 60,
    "Celta Vigo": 55, "Osasuna": 52, "Rayo Vallecano": 52,
    "Mallorca": 48, "Valencia": 46, "Real Sociedad": 46,
    "Alaves": 42, "Espanyol": 42, "Sevilla": 41,
    "Girona": 41, "Leganes": 40, "Getafe": 39,
    "Las Palmas": 32, "Real Valladolid": 16,
}


def get_team_points() -> dict:
    """Return team → points mapping for LaLiga 2024-25."""
    return LALIGA_2024_STANDINGS.copy()


def load_dataset(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load and clean the LaLiga 2024-25 dataset.

    Returns a DataFrame with standardized team names and clean column types.
    """
    df = pd.read_csv(path)

    # Standardize team names
    df["team"] = df["team"].map(TEAM_NAME_MAP).fillna(df["team"])

    # Ensure numeric types
    numeric_cols = ["age", "squad_selections", "games_played", "minutes_played", "goals", "assists",
                    "yellow_cards", "red_cards", "market_value_eur", "height_cm",
                    "highest_market_value_in_eur"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows without market value
    df = df.dropna(subset=["market_value_eur"]).copy()
    df["market_value_eur"] = df["market_value_eur"].astype(int)

    # Sort by market value
    df = df.sort_values("market_value_eur", ascending=False).reset_index(drop=True)

    return df


def main():
    """Entry point: load and display dataset summary."""
    print("Loading LaLiga 2024-25 dataset (Transfermarkt)...")
    df = load_dataset()

    print(f"\nDataset: {len(df)} players, {df['team'].nunique()} teams")
    print(f"Columns: {list(df.columns)}")
    print(f"\nTop 10 players by market value:")
    print(df[["player_name", "team", "position", "age", "goals",
              "assists", "market_value_eur"]].head(10).to_string(index=False))

    print(f"\nPosition distribution:")
    print(df["position"].value_counts().to_string())


if __name__ == "__main__":
    main()
