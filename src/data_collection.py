"""
data_collection.py
==================
Generates a synthetic but realistic dataset of LaLiga player statistics
for the 2024-25 season. Players are mapped to their real teams and positions,
with performance stats generated using realistic distributions conditioned on
position, age, and club quality.

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
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "laliga_players_2024_25.csv")

TOP_TIER_TEAMS = {
    "Real Madrid", "FC Barcelona", "Atletico Madrid", "Real Sociedad",
    "Athletic Bilbao", "Real Betis", "Villarreal", "Girona",
}

# ──────────────────────────────────────────────────────────────────────
# Full LaLiga 2024-25 rosters: (name, team, position, age)
# Positions: GK, CB, LB, RB, CDM, CM, CAM, LW, RW, ST
# ──────────────────────────────────────────────────────────────────────
ROSTER = [
    # ── Real Madrid ─────────────────────────────────────────────────
    ("Thibaut Courtois", "Real Madrid", "GK", 32),
    ("Andriy Lunin", "Real Madrid", "GK", 25),
    ("Dani Carvajal", "Real Madrid", "RB", 32),
    ("Eder Militao", "Real Madrid", "CB", 26),
    ("Antonio Rudiger", "Real Madrid", "CB", 31),
    ("David Alaba", "Real Madrid", "CB", 32),
    ("Ferland Mendy", "Real Madrid", "LB", 29),
    ("Fran Garcia", "Real Madrid", "LB", 25),
    ("Aurelien Tchouameni", "Real Madrid", "CDM", 24),
    ("Eduardo Camavinga", "Real Madrid", "CM", 21),
    ("Fede Valverde", "Real Madrid", "CM", 26),
    ("Jude Bellingham", "Real Madrid", "CAM", 21),
    ("Luka Modric", "Real Madrid", "CM", 39),
    ("Arda Guler", "Real Madrid", "CAM", 19),
    ("Brahim Diaz", "Real Madrid", "RW", 25),
    ("Rodrygo Goes", "Real Madrid", "RW", 23),
    ("Vinicius Junior", "Real Madrid", "LW", 24),
    ("Kylian Mbappe", "Real Madrid", "ST", 25),
    ("Endrick Felipe", "Real Madrid", "ST", 18),
    # ── FC Barcelona ────────────────────────────────────────────────
    ("Marc-Andre ter Stegen", "FC Barcelona", "GK", 32),
    ("Inaki Pena", "FC Barcelona", "GK", 25),
    ("Jules Kounde", "FC Barcelona", "RB", 26),
    ("Pau Cubarsi", "FC Barcelona", "CB", 17),
    ("Ronald Araujo", "FC Barcelona", "CB", 25),
    ("Alejandro Balde", "FC Barcelona", "LB", 21),
    ("Andreas Christensen", "FC Barcelona", "CB", 28),
    ("Marc Casado", "FC Barcelona", "CDM", 21),
    ("Pedri Gonzalez", "FC Barcelona", "CM", 22),
    ("Gavi Lopez", "FC Barcelona", "CM", 20),
    ("Frenkie de Jong", "FC Barcelona", "CM", 27),
    ("Dani Olmo", "FC Barcelona", "CAM", 26),
    ("Fermin Lopez", "FC Barcelona", "CAM", 21),
    ("Pablo Torre", "FC Barcelona", "CM", 21),
    ("Raphinha Dias", "FC Barcelona", "LW", 27),
    ("Lamine Yamal", "FC Barcelona", "RW", 17),
    ("Robert Lewandowski", "FC Barcelona", "ST", 36),
    ("Ansu Fati", "FC Barcelona", "LW", 22),
    # ── Atletico Madrid ─────────────────────────────────────────────
    ("Jan Oblak", "Atletico Madrid", "GK", 31),
    ("Jose Gimenez", "Atletico Madrid", "CB", 29),
    ("Robin Le Normand", "Atletico Madrid", "CB", 27),
    ("Reinildo Mandava", "Atletico Madrid", "LB", 30),
    ("Nahuel Molina", "Atletico Madrid", "RB", 26),
    ("Marcos Llorente", "Atletico Madrid", "CM", 29),
    ("Rodrigo De Paul", "Atletico Madrid", "CM", 30),
    ("Koke Resurreccion", "Atletico Madrid", "CM", 32),
    ("Pablo Barrios", "Atletico Madrid", "CDM", 21),
    ("Conor Gallagher", "Atletico Madrid", "CM", 24),
    ("Antoine Griezmann", "Atletico Madrid", "ST", 33),
    ("Angel Correa", "Atletico Madrid", "RW", 29),
    ("Alexander Sorloth", "Atletico Madrid", "ST", 28),
    ("Julian Alvarez", "Atletico Madrid", "ST", 24),
    ("Samuel Lino", "Atletico Madrid", "LW", 24),
    # ── Real Sociedad ───────────────────────────────────────────────
    ("Alex Remiro", "Real Sociedad", "GK", 29),
    ("Igor Zubeldia", "Real Sociedad", "CB", 27),
    ("Jon Aramburu", "Real Sociedad", "RB", 22),
    ("Aihen Munoz", "Real Sociedad", "LB", 27),
    ("Aritz Elustondo", "Real Sociedad", "CB", 28),
    ("Martin Zubimendi", "Real Sociedad", "CDM", 25),
    ("Mikel Merino", "Real Sociedad", "CM", 28),
    ("Takefusa Kubo", "Real Sociedad", "RW", 23),
    ("Mikel Oyarzabal", "Real Sociedad", "LW", 27),
    ("Brais Mendez", "Real Sociedad", "CAM", 27),
    ("Ander Barrenetxea", "Real Sociedad", "LW", 22),
    ("Orri Oskarsson", "Real Sociedad", "ST", 19),
    # ── Athletic Bilbao ─────────────────────────────────────────────
    ("Unai Simon", "Athletic Bilbao", "GK", 27),
    ("Aitor Paredes", "Athletic Bilbao", "CB", 24),
    ("Dani Vivian", "Athletic Bilbao", "CB", 25),
    ("Yeray Alvarez", "Athletic Bilbao", "CB", 29),
    ("Yuri Berchiche", "Athletic Bilbao", "LB", 34),
    ("Oscar de Marcos", "Athletic Bilbao", "RB", 35),
    ("Oihan Sancet", "Athletic Bilbao", "CAM", 24),
    ("Mikel Vesga", "Athletic Bilbao", "CDM", 31),
    ("Benat Prados", "Athletic Bilbao", "CM", 20),
    ("Nico Williams", "Athletic Bilbao", "LW", 22),
    ("Inaki Williams", "Athletic Bilbao", "RW", 30),
    ("Gorka Guruzeta", "Athletic Bilbao", "ST", 28),
    # ── Real Betis ──────────────────────────────────────────────────
    ("Rui Silva", "Real Betis", "GK", 30),
    ("Marc Bartra", "Real Betis", "CB", 33),
    ("Hector Bellerin", "Real Betis", "RB", 29),
    ("Aitor Ruibal", "Real Betis", "RB", 29),
    ("Guido Rodriguez", "Real Betis", "CDM", 30),
    ("Johnny Cardoso", "Real Betis", "CM", 23),
    ("Isco Alarcon", "Real Betis", "CAM", 32),
    ("Nabil Fekir", "Real Betis", "CAM", 31),
    ("Giovani Lo Celso", "Real Betis", "CM", 28),
    ("Ayoze Perez", "Real Betis", "ST", 31),
    ("Chimy Avila", "Real Betis", "ST", 30),
    ("Ez Abde", "Real Betis", "LW", 22),
    # ── Villarreal ──────────────────────────────────────────────────
    ("Filip Jorgensen", "Villarreal", "GK", 22),
    ("Raul Albiol", "Villarreal", "CB", 38),
    ("Logan Costa", "Villarreal", "CB", 23),
    ("Alfonso Pedraza", "Villarreal", "LB", 28),
    ("Kiko Femenia", "Villarreal", "RB", 33),
    ("Dani Parejo", "Villarreal", "CM", 35),
    ("Santi Comesana", "Villarreal", "CM", 27),
    ("Alex Baena", "Villarreal", "CAM", 23),
    ("Yeremi Pino", "Villarreal", "RW", 22),
    ("Nicolas Jackson", "Villarreal", "ST", 23),
    ("Gerard Moreno", "Villarreal", "ST", 32),
    ("Ilias Akhomach", "Villarreal", "LW", 20),
    # ── Girona ──────────────────────────────────────────────────────
    ("Paulo Gazzaniga", "Girona", "GK", 32),
    ("Daley Blind", "Girona", "CB", 34),
    ("David Lopez", "Girona", "CB", 33),
    ("Miguel Gutierrez", "Girona", "LB", 23),
    ("Arnau Martinez", "Girona", "RB", 21),
    ("Ivan Martin", "Girona", "CM", 25),
    ("Yangel Herrera", "Girona", "CM", 26),
    ("Bryan Gil", "Girona", "LW", 23),
    ("Viktor Tsygankov", "Girona", "RW", 26),
    ("Abel Ruiz", "Girona", "ST", 24),
    ("Cristhian Stuani", "Girona", "ST", 37),
    ("Bojan Miovski", "Girona", "ST", 25),
    # ── Sevilla ─────────────────────────────────────────────────────
    ("Orjan Nyland", "Sevilla", "GK", 33),
    ("Loic Bade", "Sevilla", "CB", 24),
    ("Tanguy Nianzou", "Sevilla", "CB", 22),
    ("Marcos Acuna", "Sevilla", "LB", 32),
    ("Gonzalo Montiel", "Sevilla", "RB", 27),
    ("Nemanja Gudelj", "Sevilla", "CDM", 32),
    ("Lucien Agoume", "Sevilla", "CM", 22),
    ("Suso Fernandez", "Sevilla", "RW", 30),
    ("Dodi Lukebakio", "Sevilla", "LW", 27),
    ("Isaac Romero", "Sevilla", "ST", 24),
    ("Jesus Navas", "Sevilla", "RB", 38),
    ("Saul Niguez", "Sevilla", "CM", 29),
    # ── Valencia ────────────────────────────────────────────────────
    ("Giorgi Mamardashvili", "Valencia", "GK", 23),
    ("Mouctar Diakhaby", "Valencia", "CB", 27),
    ("Cesar Tarrega", "Valencia", "CB", 22),
    ("Jose Gaya", "Valencia", "LB", 29),
    ("Thierry Correia", "Valencia", "RB", 25),
    ("Hugo Guillamon", "Valencia", "CDM", 24),
    ("Javi Guerra", "Valencia", "CM", 21),
    ("Pepelu Sanchez", "Valencia", "CM", 26),
    ("Diego Lopez", "Valencia", "RW", 24),
    ("Hugo Duro", "Valencia", "ST", 25),
    ("Andre Almeida", "Valencia", "LW", 24),
    ("Fran Perez", "Valencia", "CAM", 22),
    # ── Osasuna ─────────────────────────────────────────────────────
    ("Sergio Herrera", "Osasuna", "GK", 31),
    ("David Garcia", "Osasuna", "CB", 29),
    ("Unai Garcia", "Osasuna", "CB", 32),
    ("Juan Cruz", "Osasuna", "LB", 25),
    ("Nacho Vidal", "Osasuna", "RB", 29),
    ("Lucas Torro", "Osasuna", "CDM", 29),
    ("Jon Moncayola", "Osasuna", "CM", 26),
    ("Aimar Oroz", "Osasuna", "CAM", 22),
    ("Bryan Zaragoza", "Osasuna", "LW", 23),
    ("Ante Budimir", "Osasuna", "ST", 33),
    ("Ruben Garcia", "Osasuna", "RW", 31),
    ("Iker Munoz", "Osasuna", "CM", 19),
    # ── Celta Vigo ──────────────────────────────────────────────────
    ("Ivan Villar", "Celta Vigo", "GK", 27),
    ("Carl Starfelt", "Celta Vigo", "CB", 29),
    ("Joseph Aidoo", "Celta Vigo", "CB", 28),
    ("Javi Galan", "Celta Vigo", "LB", 29),
    ("Oscar Mingueza", "Celta Vigo", "RB", 25),
    ("Fran Beltran", "Celta Vigo", "CM", 25),
    ("Hugo Alvarez", "Celta Vigo", "CM", 20),
    ("Iago Aspas", "Celta Vigo", "ST", 37),
    ("Borja Iglesias", "Celta Vigo", "ST", 31),
    ("Anastasios Douvikas", "Celta Vigo", "ST", 24),
    ("Williot Swedberg", "Celta Vigo", "LW", 20),
    ("Franco Cervi", "Celta Vigo", "RW", 30),
    # ── Mallorca ────────────────────────────────────────────────────
    ("Predrag Rajkovic", "Mallorca", "GK", 28),
    ("Martin Valjent", "Mallorca", "CB", 29),
    ("Antonio Raillo", "Mallorca", "CB", 33),
    ("Johan Mojica", "Mallorca", "LB", 32),
    ("Pablo Maffeo", "Mallorca", "RB", 27),
    ("Manu Morlanes", "Mallorca", "CM", 25),
    ("Sergi Darder", "Mallorca", "CM", 30),
    ("Antonio Sanchez", "Mallorca", "CM", 26),
    ("Dani Rodriguez", "Mallorca", "CAM", 30),
    ("Vedat Muriqi", "Mallorca", "ST", 30),
    ("Cyle Larin", "Mallorca", "ST", 29),
    ("Samu Costa", "Mallorca", "CDM", 24),
    # ── Getafe ──────────────────────────────────────────────────────
    ("David Soria", "Getafe", "GK", 31),
    ("Djene Dakonam", "Getafe", "CB", 32),
    ("Omar Alderete", "Getafe", "CB", 27),
    ("Juan Berrocal", "Getafe", "LB", 22),
    ("Juan Iglesias", "Getafe", "RB", 23),
    ("Luis Milla", "Getafe", "CM", 28),
    ("Mauro Arambarri", "Getafe", "CDM", 28),
    ("Carles Alena", "Getafe", "CM", 26),
    ("Borja Mayoral", "Getafe", "ST", 27),
    ("Mason Greenwood", "Getafe", "RW", 22),
    ("Christantus Uche", "Getafe", "LW", 22),
    ("Alvaro Rodriguez", "Getafe", "ST", 22),
    # ── Rayo Vallecano ──────────────────────────────────────────────
    ("Stole Dimitrievski", "Rayo Vallecano", "GK", 30),
    ("Alejandro Catena", "Rayo Vallecano", "CB", 29),
    ("Abdul Mumin", "Rayo Vallecano", "CB", 25),
    ("Alfonso Espino", "Rayo Vallecano", "LB", 32),
    ("Ivan Balliu", "Rayo Vallecano", "RB", 32),
    ("Oscar Trejo", "Rayo Vallecano", "CAM", 36),
    ("Unai Lopez", "Rayo Vallecano", "CM", 28),
    ("Isi Palazon", "Rayo Vallecano", "RW", 29),
    ("Alvaro Garcia", "Rayo Vallecano", "LW", 27),
    ("Raul de Tomas", "Rayo Vallecano", "ST", 29),
    ("Jorge de Frutos", "Rayo Vallecano", "RW", 27),
    ("Sergio Camello", "Rayo Vallecano", "ST", 23),
    # ── Las Palmas ──────────────────────────────────────────────────
    ("Alvaro Valles", "Las Palmas", "GK", 27),
    ("Alex Suarez", "Las Palmas", "CB", 24),
    ("Mika Marmol", "Las Palmas", "CB", 23),
    ("Marvin Park", "Las Palmas", "RB", 24),
    ("Viti Rozada", "Las Palmas", "LB", 21),
    ("Javi Munoz", "Las Palmas", "CM", 27),
    ("Alberto Moleiro", "Las Palmas", "CAM", 20),
    ("Kirian Rodriguez", "Las Palmas", "CDM", 28),
    ("Sandro Ramirez", "Las Palmas", "ST", 28),
    ("Marc Cardona", "Las Palmas", "ST", 29),
    ("Oli McBurnie", "Las Palmas", "ST", 28),
    ("Fabio Silva", "Las Palmas", "LW", 22),
    # ── Alaves ──────────────────────────────────────────────────────
    ("Antonio Sivera", "Alaves", "GK", 28),
    ("Abdelkabir Abqar", "Alaves", "CB", 24),
    ("Moussa Diarra", "Alaves", "CB", 23),
    ("Nahuel Tenaglia", "Alaves", "RB", 25),
    ("Santiago Mourino", "Alaves", "CB", 24),
    ("Antonio Blanco", "Alaves", "CDM", 23),
    ("Joan Jordan", "Alaves", "CM", 30),
    ("Carlos Vicente", "Alaves", "RW", 22),
    ("Luis Rioja", "Alaves", "LW", 30),
    ("Toni Martinez", "Alaves", "ST", 27),
    ("Kike Garcia", "Alaves", "ST", 35),
    ("Carlos Martin", "Alaves", "CAM", 22),
    # ── Espanyol ────────────────────────────────────────────────────
    ("Joan Garcia", "Espanyol", "GK", 23),
    ("Leandro Cabrera", "Espanyol", "CB", 33),
    ("Fernando Calero", "Espanyol", "CB", 29),
    ("Brian Olivan", "Espanyol", "LB", 29),
    ("Omar El Hilali", "Espanyol", "RB", 22),
    ("Edu Exposito", "Espanyol", "CM", 28),
    ("Alex Kral", "Espanyol", "CDM", 26),
    ("Javi Puado", "Espanyol", "RW", 26),
    ("Irvin Cardona", "Espanyol", "ST", 27),
    ("Martin Braithwaite", "Espanyol", "ST", 33),
    ("Alejo Veliz", "Espanyol", "ST", 21),
    ("Pere Milla", "Espanyol", "LW", 32),
    # ── Real Valladolid ─────────────────────────────────────────────
    ("Karl Hein", "Real Valladolid", "GK", 22),
    ("Javi Sanchez", "Real Valladolid", "CB", 27),
    ("Luis Perez", "Real Valladolid", "CB", 25),
    ("Lucas Rosa", "Real Valladolid", "LB", 22),
    ("Ivan Fresneda", "Real Valladolid", "RB", 20),
    ("Kike Perez", "Real Valladolid", "CM", 32),
    ("Selim Amallah", "Real Valladolid", "CM", 27),
    ("Juanmi Latasa", "Real Valladolid", "ST", 24),
    ("Raul Moro", "Real Valladolid", "LW", 22),
    ("Marcos Andre", "Real Valladolid", "ST", 28),
    ("Darwin Machis", "Real Valladolid", "RW", 31),
    ("Victor Meseguer", "Real Valladolid", "CM", 23),
    # ── Leganes ─────────────────────────────────────────────────────
    ("Marko Dmitrovic", "Leganes", "GK", 32),
    ("Jorge Saenz", "Leganes", "CB", 28),
    ("Sergio Gonzalez", "Leganes", "CB", 26),
    ("Valentin Rosier", "Leganes", "RB", 27),
    ("Enric Franquesa", "Leganes", "LB", 25),
    ("Sebastian Cristoforo", "Leganes", "CDM", 31),
    ("Oscar Rodriguez", "Leganes", "CAM", 27),
    ("Dario Poveda", "Leganes", "ST", 25),
    ("Diego Garcia", "Leganes", "CM", 22),
    ("Miguel De La Fuente", "Leganes", "ST", 25),
    ("Juan Cruz Armada", "Leganes", "LW", 23),
    ("Munir El Haddadi", "Leganes", "RW", 28),
]


# ──────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────

def _base_market_value(age: int, position: str, is_top_tier: bool, rng: np.random.Generator) -> float:
    """
    Estimate a base market value (EUR) using realistic heuristics.
    """
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

    pos_multipliers = {
        "GK": 0.55, "CB": 0.75, "LB": 0.70, "RB": 0.70,
        "CDM": 0.80, "CM": 0.90, "CAM": 1.05,
        "LW": 1.10, "RW": 1.10, "ST": 1.15,
    }
    pos_factor = pos_multipliers.get(position, 0.80)

    tier_factor = rng.uniform(1.4, 1.8) if is_top_tier else 1.0

    base = rng.lognormal(mean=2.0, sigma=0.8)
    value = base * age_factor * pos_factor * tier_factor * 1_000_000

    return np.clip(value, 200_000, 180_000_000)


def _generate_stats(position: str, minutes: int, age: int,
                    market_value: float, is_top_tier: bool,
                    rng: np.random.Generator) -> dict:
    """
    Generate performance statistics conditioned on position, minutes,
    and a quality signal derived from market value.
    """
    quality = np.log10(max(market_value, 1)) / 8.0
    quality = np.clip(quality + rng.normal(0, 0.05), 0.3, 1.2)

    matches_90 = minutes / 90.0

    # Goals
    if position == "GK":
        goals_per90 = rng.exponential(0.002)
    elif position in ("CB", "LB", "RB"):
        goals_per90 = rng.exponential(0.03) * quality
    elif position in ("CDM", "CM"):
        goals_per90 = rng.exponential(0.06) * quality
    elif position == "CAM":
        goals_per90 = rng.exponential(0.12) * quality
    else:
        goals_per90 = rng.exponential(0.22) * quality
    goals = int(round(goals_per90 * matches_90))

    # Assists
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

    # Pass accuracy
    base_pass = {"GK": 72, "CB": 82, "LB": 78, "RB": 78, "CDM": 84,
                 "CM": 85, "CAM": 82, "LW": 78, "RW": 78, "ST": 75}
    pass_accuracy = np.clip(
        base_pass[position] + quality * 8 + rng.normal(0, 3), 55, 96
    )

    # Tackles per 90
    base_tackles = {"GK": 0.1, "CB": 2.8, "LB": 2.5, "RB": 2.5, "CDM": 3.0,
                    "CM": 2.0, "CAM": 1.2, "LW": 1.0, "RW": 1.0, "ST": 0.6}
    tackles_per90 = max(0, base_tackles[position] + rng.normal(0, 0.6))
    tackles = int(round(tackles_per90 * matches_90))

    # Interceptions per 90
    base_interceptions = {"GK": 0.05, "CB": 1.8, "LB": 1.3, "RB": 1.3,
                          "CDM": 2.0, "CM": 1.2, "CAM": 0.7, "LW": 0.5,
                          "RW": 0.5, "ST": 0.3}
    interceptions_per90 = max(0, base_interceptions[position] + rng.normal(0, 0.4))
    interceptions = int(round(interceptions_per90 * matches_90))

    # Shots per 90
    base_shots = {"GK": 0.0, "CB": 0.4, "LB": 0.3, "RB": 0.3, "CDM": 0.6,
                  "CM": 1.0, "CAM": 1.8, "LW": 2.0, "RW": 2.0, "ST": 2.8}
    shots_per90 = max(0, base_shots[position] * quality + rng.normal(0, 0.3))
    shots = int(round(shots_per90 * matches_90))

    # Key passes per 90
    base_kp = {"GK": 0.1, "CB": 0.3, "LB": 0.7, "RB": 0.7, "CDM": 0.8,
               "CM": 1.2, "CAM": 1.8, "LW": 1.4, "RW": 1.4, "ST": 0.9}
    key_passes_per90 = max(0, base_kp[position] * quality + rng.normal(0, 0.3))
    key_passes = int(round(key_passes_per90 * matches_90))

    # Dribbles completed per 90
    base_dribbles = {"GK": 0.0, "CB": 0.2, "LB": 0.8, "RB": 0.8, "CDM": 0.5,
                     "CM": 0.9, "CAM": 1.5, "LW": 2.2, "RW": 2.2, "ST": 1.3}
    dribbles_per90 = max(0, base_dribbles[position] * quality + rng.normal(0, 0.4))
    dribbles_completed = int(round(dribbles_per90 * matches_90))

    # Aerial duels won per 90
    base_aerial = {"GK": 0.3, "CB": 2.5, "LB": 0.8, "RB": 0.8, "CDM": 1.2,
                   "CM": 0.8, "CAM": 0.5, "LW": 0.3, "RW": 0.3, "ST": 1.8}
    aerial_per90 = max(0, base_aerial[position] + rng.normal(0, 0.5))
    aerial_duels_won = int(round(aerial_per90 * matches_90))

    # Yellow / Red cards
    base_yc = {"GK": 0.04, "CB": 0.18, "LB": 0.14, "RB": 0.14, "CDM": 0.20,
               "CM": 0.12, "CAM": 0.08, "LW": 0.06, "RW": 0.06, "ST": 0.08}
    yellow_cards = int(round(max(0, base_yc[position] * matches_90 + rng.normal(0, 0.8))))
    red_cards = 1 if rng.random() < 0.04 else 0

    # Clean sheets (GK only)
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
    records = []

    for name, team, position, age in ROSTER:
        is_top = team in TOP_TIER_TEAMS

        # Minutes played (0-3420 max for 38 matches)
        if rng.random() < 0.12:
            minutes_played = int(rng.integers(0, 600))
        else:
            minutes_played = int(np.clip(rng.normal(1800, 700), 200, 3420))

        # Market value
        market_value = _base_market_value(age, position, is_top, rng)
        market_value = round(market_value, -4)

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
    df = df.sort_values("market_value_eur", ascending=False).reset_index(drop=True)

    return df


def main():
    """Entry point: generate data and save to CSV."""
    print("Generating LaLiga 2024-25 synthetic dataset...")
    df = generate_dataset()

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
