"""
data_collection.py
==================
Generates a synthetic but realistic dataset of LaLiga player statistics
for the 2024-25 season. The data mirrors the structure and distributions
found in real football analytics databases (e.g., FBRef, Transfermarkt).

Usage:
    python src/data_collection.py

Output:
    data/laliga_players_2024_25.csv
"""

import os
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────
SEED = 42
NUM_PLAYERS = 460  # ~23 players x 20 teams
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "laliga_players_2024_25.csv")

TEAMS = [
    "Real Madrid", "FC Barcelona", "Atletico Madrid", "Real Sociedad",
    "Athletic Bilbao", "Real Betis", "Villarreal", "Girona",
    "Sevilla", "Valencia", "Osasuna", "Celta Vigo",
    "Mallorca", "Getafe", "Rayo Vallecano", "Las Palmas",
    "Alaves", "Espanyol", "Real Valladolid", "Leganes",
]

# Top-tier team indices (0-7) get a quality boost
TOP_TIER_INDICES = set(range(8))

POSITIONS = ["GK", "CB", "LB", "RB", "CDM", "CM", "CAM", "LW", "RW", "ST"]
POSITION_WEIGHTS = [0.08, 0.14, 0.07, 0.07, 0.08, 0.14, 0.08, 0.10, 0.10, 0.14]

# Real LaLiga player names (2024-25 season) to ensure realistic output.
# We include enough names to cover all generated players; extras are filled
# with realistic fictional names that don't collide with real-player combos.
REAL_PLAYER_NAMES = [
    # Real Madrid
    "Thibaut Courtois", "Andriy Lunin", "Dani Carvajal", "Eder Militao",
    "Antonio Rudiger", "David Alaba", "Ferland Mendy", "Fran Garcia",
    "Aurelien Tchouameni", "Eduardo Camavinga", "Fede Valverde", "Jude Bellingham",
    "Luka Modric", "Arda Guler", "Brahim Diaz", "Rodrygo Goes",
    "Vinicius Junior", "Kylian Mbappe", "Endrick Felipe",
    # FC Barcelona
    "Marc-Andre ter Stegen", "Inaki Pena", "Jules Kounde", "Pau Cubarsi",
    "Ronald Araujo", "Alejandro Balde", "Andreas Christensen", "Marc Casado",
    "Pedri Gonzalez", "Gavi Lopez", "Frenkie de Jong", "Dani Olmo",
    "Fermin Lopez", "Pablo Torre", "Raphinha Dias", "Lamine Yamal",
    "Robert Lewandowski", "Ansu Fati",
    # Atletico Madrid
    "Jan Oblak", "Jose Gimenez", "Robin Le Normand", "Reinildo Mandava",
    "Nahuel Molina", "Marcos Llorente", "Rodrigo De Paul", "Koke Resurreccion",
    "Pablo Barrios", "Conor Gallagher", "Antoine Griezmann", "Angel Correa",
    "Alexander Sorloth", "Julian Alvarez", "Samuel Lino",
    # Real Sociedad
    "Alex Remiro", "Igor Zubeldia", "Robin Le Normand", "Aihen Munoz",
    "Aritz Elustondo", "Martin Zubimendi", "Mikel Merino", "Takefusa Kubo",
    "Mikel Oyarzabal", "Brais Mendez", "Ander Barrenetxea", "Orri Oskarsson",
    # Athletic Bilbao
    "Unai Simon", "Aitor Paredes", "Yeray Alvarez", "Yuri Berchiche",
    "Oscar de Marcos", "Oihan Sancet", "Nico Williams", "Inaki Williams",
    "Gorka Guruzeta", "Dani Vivian", "Mikel Vesga", "Benat Prados",
    # Real Betis
    "Rui Silva", "Marc Bartra", "Hector Bellerin", "Aitor Ruibal",
    "Guido Rodriguez", "Johnny Cardoso", "Isco Alarcon", "Nabil Fekir",
    "Ayoze Perez", "Giovani Lo Celso", "Chimy Avila", "Ez Abde",
    # Villarreal
    "Filip Jorgensen", "Raul Albiol", "Logan Costa", "Alfonso Pedraza",
    "Kiko Femenia", "Dani Parejo", "Santi Comesana", "Alex Baena",
    "Yeremi Pino", "Nicolas Jackson", "Gerard Moreno", "Ayoze Perez",
    # Girona
    "Paulo Gazzaniga", "Daley Blind", "David Lopez", "Miguel Gutierrez",
    "Arnau Martinez", "Ivan Martin", "Yangel Herrera", "Bryan Gil",
    "Viktor Tsygankov", "Abel Ruiz", "Cristhian Stuani", "Bojan Miovski",
    # Sevilla
    "Orjan Nyland", "Loic Bade", "Tanguy Nianzou", "Marcos Acuna",
    "Gonzalo Montiel", "Nemanja Gudelj", "Lucien Agoume", "Suso Fernandez",
    "Dodi Lukebakio", "Youssef En-Nesyri", "Isaac Romero", "Jesus Navas",
    # Valencia
    "Giorgi Mamardashvili", "Mouctar Diakhaby", "Cesar Tarrega", "Jose Gaya",
    "Thierry Correia", "Hugo Guillamón", "Javi Guerra", "Pepelu Sanchez",
    "Diego Lopez", "Hugo Duro", "Andre Almeida", "Fran Perez",
    # Osasuna
    "Sergio Herrera", "David Garcia", "Unai Garcia", "Juan Cruz",
    "Nacho Vidal", "Lucas Torro", "Jon Moncayola", "Aimar Oroz",
    "Bryan Zaragoza", "Ante Budimir", "Chimy Avila", "Ruben Garcia",
    # Celta Vigo
    "Ivan Villar", "Carl Starfelt", "Joseph Aidoo", "Javi Galan",
    "Oscar Mingueza", "Fran Beltran", "Hugo Alvarez", "Iago Aspas",
    "Borja Iglesias", "Anastasios Douvikas", "Williot Swedberg", "Franco Cervi",
    # Mallorca
    "Predrag Rajkovic", "Martin Valjent", "Antonio Raillo", "Johan Mojica",
    "Pablo Maffeo", "Manu Morlanes", "Sergi Darder", "Antonio Sanchez",
    "Dani Rodriguez", "Vedat Muriqi", "Cyle Larin", "Samu Costa",
    # Getafe
    "David Soria", "Djene Dakonam", "Juan Iglesias", "Juan Berrocal",
    "Omar Alderete", "Luis Milla", "Mauro Arambarri", "Carles Alena",
    "Borja Mayoral", "Alvaro Rodriguez", "Mason Greenwood", "Christantus Uche",
    # Rayo Vallecano
    "Stole Dimitrievski", "Alejandro Catena", "Abdul Mumin", "Alfonso Espino",
    "Ivan Balliu", "Oscar Trejo", "Unai Lopez", "Isi Palazon",
    "Alvaro Garcia", "Raul de Tomas", "Jorge de Frutos", "Sergio Camello",
    # Las Palmas
    "Alvaro Valles", "Alex Suarez", "Mika Marmol", "Marvin Park",
    "Viti Rozada", "Javi Munoz", "Alberto Moleiro", "Kirian Rodriguez",
    "Sandro Ramirez", "Marc Cardona", "Oli McBurnie", "Fabio Silva",
    # Alaves
    "Antonio Sivera", "Abdelkabir Abqar", "Moussa Diarra", "Nahuel Tenaglia",
    "Santiago Mourino", "Antonio Blanco", "Joan Jordan", "Carlos Vicente",
    "Luis Rioja", "Toni Martinez", "Kike Garcia", "Carlos Martin",
    # Espanyol
    "Joan Garcia", "Leandro Cabrera", "Fernando Calero", "Brian Olivan",
    "Omar El Hilali", "Edu Exposito", "Alex Kral", "Javi Puado",
    "Irvin Cardona", "Martin Braithwaite", "Alejo Veliz", "Pere Milla",
    # Real Valladolid
    "Karl Hein", "Javi Sanchez", "Luis Perez", "Lucas Rosa",
    "Ivan Fresneda", "Kike Perez", "Juanmi Latasa", "Selim Amallah",
    "Raul Moro", "Marcos Andre", "Darwin Machis", "Victor Meseguer",
    # Leganes
    "Marko Dmitrovic", "Jorge Saenz", "Sergio Gonzalez", "Valentin Rosier",
    "Enric Franquesa", "Dario Poveda", "Oscar Rodriguez", "Sebastian Cristoforo",
    "Diego Garcia", "Miguel De La Fuente", "Juan Cruz Armada", "Munir El Haddadi",
]


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

_FALLBACK_FIRST = [
    "Sergio", "Pablo", "Marco", "Adrian", "Victor", "Raul", "Manuel", "Jorge",
    "Miguel", "Carlos", "Alejandro", "Fernando", "Gonzalo", "Hector", "Ivan",
    "Jaime", "Kevin", "Lorenzo", "Nicolas", "Oscar", "Ruben", "Tomas", "Unai",
    "Xavier", "Yago", "Aitor", "Borja", "Cesar", "Emilio", "Fabian",
]
_FALLBACK_LAST = [
    "Ruiz", "Navarro", "Reyes", "Ortega", "Delgado", "Blanco", "Medina",
    "Vargas", "Pena", "Campos", "Soto", "Carrasco", "Vera", "Rios", "Fuentes",
    "Caballero", "Serrano", "Pascual", "Ibarra", "Salazar", "Ochoa", "Cano",
    "Aguilar", "Marin", "Cortes", "Prieto", "Lara", "Duran", "Nieto", "Rubio",
]

def _generate_player_name(rng: np.random.Generator, used_names: set,
                          name_index: list) -> str:
    """Pick the next real player name, or generate a unique fictional one if exhausted."""
    while name_index:
        name = name_index.pop()
        if name not in used_names:
            used_names.add(name)
            return name
    # Fallback: generate fictional name (30x30 = 900 combos, more than enough)
    while True:
        name = f"{rng.choice(_FALLBACK_FIRST)} {rng.choice(_FALLBACK_LAST)}"
        if name not in used_names:
            used_names.add(name)
            return name


def _base_market_value(age: int, position: str, is_top_tier: bool, rng: np.random.Generator) -> float:
    """
    Estimate a base market value (EUR) using realistic heuristics:
    - Peak age range (23-28) commands highest values
    - Attackers generally valued higher than defenders
    - Top-tier clubs inflate value by ~40-80%
    """
    # Age curve: peaks around 25-26
    if age <= 20:
        age_factor = 0.5 + (age - 17) * 0.12
    elif age <= 24:
        age_factor = 0.85 + (age - 20) * 0.04
    elif age <= 28:
        age_factor = 1.0
    elif age <= 32:
        age_factor = 1.0 - (age - 28) * 0.12
    else:
        age_factor = 0.45 - (age - 32) * 0.08

    age_factor = max(age_factor, 0.08)

    # Position multiplier
    pos_multipliers = {
        "GK": 0.55, "CB": 0.75, "LB": 0.70, "RB": 0.70,
        "CDM": 0.80, "CM": 0.90, "CAM": 1.05,
        "LW": 1.10, "RW": 1.10, "ST": 1.15,
    }
    pos_factor = pos_multipliers.get(position, 0.80)

    # Club tier boost
    tier_factor = rng.uniform(1.4, 1.8) if is_top_tier else 1.0

    # Base value in millions, with noise
    base = rng.lognormal(mean=2.0, sigma=0.8)  # median ~7.4M
    value = base * age_factor * pos_factor * tier_factor * 1_000_000

    # Clamp to realistic range
    return np.clip(value, 200_000, 180_000_000)


def _generate_stats(position: str, minutes: int, age: int,
                    market_value: float, is_top_tier: bool,
                    rng: np.random.Generator) -> dict:
    """
    Generate performance statistics conditioned on position, minutes,
    and a quality signal derived from market value.
    """
    # Quality signal: higher-value players tend to produce better stats
    quality = np.log10(max(market_value, 1)) / 8.0  # 0.6 - 1.0 roughly
    quality = np.clip(quality + rng.normal(0, 0.05), 0.3, 1.2)

    matches_90 = minutes / 90.0

    # ── Goals ────────────────────────────────────────────────────────
    if position == "GK":
        goals_per90 = rng.exponential(0.002)
    elif position in ("CB", "LB", "RB"):
        goals_per90 = rng.exponential(0.03) * quality
    elif position in ("CDM", "CM"):
        goals_per90 = rng.exponential(0.06) * quality
    elif position == "CAM":
        goals_per90 = rng.exponential(0.12) * quality
    else:  # LW, RW, ST
        goals_per90 = rng.exponential(0.22) * quality
    goals = int(round(goals_per90 * matches_90))

    # ── Assists ──────────────────────────────────────────────────────
    if position == "GK":
        assists_per90 = rng.exponential(0.005)
    elif position in ("CB", "CDM"):
        assists_per90 = rng.exponential(0.03) * quality
    elif position in ("LB", "RB"):
        assists_per90 = rng.exponential(0.06) * quality
    elif position in ("CM", "CAM"):
        assists_per90 = rng.exponential(0.10) * quality
    else:
        assists_per90 = rng.exponential(0.12) * quality
    assists = int(round(assists_per90 * matches_90))

    # ── Pass accuracy ────────────────────────────────────────────────
    base_pass = {"GK": 72, "CB": 82, "LB": 78, "RB": 78, "CDM": 84,
                 "CM": 85, "CAM": 82, "LW": 78, "RW": 78, "ST": 75}
    pass_accuracy = np.clip(
        base_pass[position] + quality * 8 + rng.normal(0, 3), 55, 96
    )

    # ── Tackles per 90 ───────────────────────────────────────────────
    base_tackles = {"GK": 0.1, "CB": 2.8, "LB": 2.5, "RB": 2.5, "CDM": 3.0,
                    "CM": 2.0, "CAM": 1.2, "LW": 1.0, "RW": 1.0, "ST": 0.6}
    tackles_per90 = max(0, base_tackles[position] + rng.normal(0, 0.6))
    tackles = int(round(tackles_per90 * matches_90))

    # ── Interceptions per 90 ─────────────────────────────────────────
    base_interceptions = {"GK": 0.05, "CB": 1.8, "LB": 1.3, "RB": 1.3,
                          "CDM": 2.0, "CM": 1.2, "CAM": 0.7, "LW": 0.5,
                          "RW": 0.5, "ST": 0.3}
    interceptions_per90 = max(0, base_interceptions[position] + rng.normal(0, 0.4))
    interceptions = int(round(interceptions_per90 * matches_90))

    # ── Shots per 90 ─────────────────────────────────────────────────
    base_shots = {"GK": 0.0, "CB": 0.4, "LB": 0.3, "RB": 0.3, "CDM": 0.6,
                  "CM": 1.0, "CAM": 1.8, "LW": 2.0, "RW": 2.0, "ST": 2.8}
    shots_per90 = max(0, base_shots[position] * quality + rng.normal(0, 0.3))
    shots = int(round(shots_per90 * matches_90))

    # ── Key passes per 90 ────────────────────────────────────────────
    base_kp = {"GK": 0.1, "CB": 0.3, "LB": 0.7, "RB": 0.7, "CDM": 0.8,
               "CM": 1.2, "CAM": 1.8, "LW": 1.4, "RW": 1.4, "ST": 0.9}
    key_passes_per90 = max(0, base_kp[position] * quality + rng.normal(0, 0.3))
    key_passes = int(round(key_passes_per90 * matches_90))

    # ── Dribbles completed per 90 ────────────────────────────────────
    base_dribbles = {"GK": 0.0, "CB": 0.2, "LB": 0.8, "RB": 0.8, "CDM": 0.5,
                     "CM": 0.9, "CAM": 1.5, "LW": 2.2, "RW": 2.2, "ST": 1.3}
    dribbles_per90 = max(0, base_dribbles[position] * quality + rng.normal(0, 0.4))
    dribbles_completed = int(round(dribbles_per90 * matches_90))

    # ── Aerial duels won per 90 ──────────────────────────────────────
    base_aerial = {"GK": 0.3, "CB": 2.5, "LB": 0.8, "RB": 0.8, "CDM": 1.2,
                   "CM": 0.8, "CAM": 0.5, "LW": 0.3, "RW": 0.3, "ST": 1.8}
    aerial_per90 = max(0, base_aerial[position] + rng.normal(0, 0.5))
    aerial_duels_won = int(round(aerial_per90 * matches_90))

    # ── Yellow / Red cards ───────────────────────────────────────────
    base_yc = {"GK": 0.04, "CB": 0.18, "LB": 0.14, "RB": 0.14, "CDM": 0.20,
               "CM": 0.12, "CAM": 0.08, "LW": 0.06, "RW": 0.06, "ST": 0.08}
    yellow_cards = int(round(max(0, base_yc[position] * matches_90 + rng.normal(0, 0.8))))
    red_cards = 1 if rng.random() < 0.04 else 0

    # ── Clean sheets (GK only, 0 for others) ────────────────────────
    if position == "GK":
        cs_rate = 0.25 + quality * 0.15 + rng.normal(0, 0.05)
        clean_sheets = int(round(max(0, cs_rate * (minutes / 90))))
    else:
        clean_sheets = 0

    return {
        "goals": max(0, goals),
        "assists": max(0, assists),
        "shots": max(0, shots),
        "key_passes": max(0, key_passes),
        "pass_accuracy": round(pass_accuracy, 1),
        "dribbles_completed": max(0, dribbles_completed),
        "tackles": max(0, tackles),
        "interceptions": max(0, interceptions),
        "aerial_duels_won": max(0, aerial_duels_won),
        "yellow_cards": max(0, yellow_cards),
        "red_cards": red_cards,
        "clean_sheets": clean_sheets,
    }


# ──────────────────────────────────────────────────────────────────────
# Main generation pipeline
# ──────────────────────────────────────────────────────────────────────

def generate_dataset(seed: int = SEED) -> pd.DataFrame:
    """Generate the full synthetic LaLiga dataset."""
    rng = np.random.default_rng(seed)
    used_names: set = set()
    records = []

    # Deduplicate and shuffle real player names so they distribute across teams
    name_index = list(dict.fromkeys(REAL_PLAYER_NAMES))  # preserve order, remove dupes
    rng.shuffle(name_index)

    for team_idx, team in enumerate(TEAMS):
        is_top = team_idx in TOP_TIER_INDICES
        n_players = rng.integers(22, 26)

        for _ in range(n_players):
            name = _generate_player_name(rng, used_names, name_index)

            # Position
            position = rng.choice(POSITIONS, p=POSITION_WEIGHTS)

            # Age: realistic distribution (18-37, peak around 26)
            age = int(np.clip(rng.normal(26.5, 3.5), 17, 38))

            # Minutes played (0-3420 max for 38 matches)
            if rng.random() < 0.12:
                # Fringe / injured players
                minutes_played = int(rng.integers(0, 600))
            else:
                minutes_played = int(np.clip(rng.normal(1800, 700), 200, 3420))

            # Market value
            market_value = _base_market_value(age, position, is_top, rng)
            market_value = round(market_value, -4)  # round to nearest 10k

            # Performance stats
            stats = _generate_stats(position, minutes_played, age,
                                    market_value, is_top, rng)

            records.append({
                "player_name": name,
                "team": team,
                "position": position,
                "age": age,
                "minutes_played": minutes_played,
                "market_value_eur": int(market_value),
                **stats,
            })

    df = pd.DataFrame(records)

    # Sort by market value descending for readability
    df = df.sort_values("market_value_eur", ascending=False).reset_index(drop=True)

    return df


def main():
    """Entry point: generate data and save to CSV."""
    print("Generating LaLiga 2024-25 synthetic dataset...")
    df = generate_dataset()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Dataset saved to {OUTPUT_PATH}")
    print(f"  Players: {len(df)}")
    print(f"  Teams:   {df['team'].nunique()}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\nSample (top 5 by market value):")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
