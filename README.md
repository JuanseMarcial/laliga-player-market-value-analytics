# LaLiga Player Performance & Market Value Analytics

A data analytics project exploring the relationship between player performance metrics and market valuations across LaLiga (2024-25 season) using **real Transfermarkt data**. Built from the complete [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) — **622 players across all 20 teams** with full matchday squad rosters, market valuations, and appearance statistics. This project applies exploratory data analysis, feature engineering, and machine learning to uncover what truly drives a footballer's market value in Spain's top division.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Key Findings

1. **Minutes played is the strongest predictor of market value** (31% feature importance). Consistent starters command far higher valuations — regular playing time signals trust from the manager and reliability.

2. **Team prestige is the second-most important driver** (20%), captured via league standings points (not squad market values, which would cause target leakage). The *shirt a player wears* matters almost as much as what they do on the pitch.

3. **Tuned Gradient Boosting achieves CV R² = 0.65** (test R² = 0.84). Hyperparameters selected via `RandomizedSearchCV` (50 iterations). We report cross-validation as the primary metric — a single test split can be misleadingly optimistic.

4. **Demographics alone explain ~26% of variance** (baseline model on all 622 players using only age, position, team, height). Match statistics add substantial predictive power, improving CV R² from 0.26 to 0.65.

---

## Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Source** | [Transfermarkt](https://www.transfermarkt.com/) via [transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets) |
| **Data Manipulation** | pandas, NumPy |
| **Machine Learning** | scikit-learn (Random Forest, Gradient Boosting, Ridge Regression) |
| **Visualization** | Matplotlib, Seaborn |
| **Environment** | Jupyter Notebook |

---

## Project Structure

```
football-analytics-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── laliga_players_2024_25.csv     # Real Transfermarkt data (622 players, 20 teams)
├── images/                             # Auto-generated plot PNGs
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA: distributions, correlations, visual analysis
│   └── 02_predictive_model.ipynb       # ML: feature engineering, model training, evaluation
└── src/
    ├── data_collection.py              # Data loading & cleaning (Transfermarkt source)
    ├── feature_engineering.py          # Per-90 metrics, age buckets, composite indices
    └── visualization.py               # Reusable plotting functions for football analytics
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/JuanseMarcial/laliga-player-market-value-analytics.git
cd laliga-player-market-value-analytics
```

### 2. Create a virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the notebooks
The dataset (`data/laliga_players_2024_25.csv`) is included in the repo — real Transfermarkt data for the 2024-25 season.
```bash
jupyter notebook notebooks/
```
- Start with `01_data_exploration.ipynb` for EDA
- Then run `02_predictive_model.ipynb` for the predictive modeling pipeline

---

## Sample Visualizations

### Market Value Distribution
![Market Value Distribution](images/market_value_distribution.png)

### Goals vs Market Value
![Goals vs Market Value](images/goals_vs_market_value.png)

### Market Value by Position
![Value by Position](images/value_by_position.png)

### Value by Age Bucket
![Value by Age Bucket](images/value_by_age_bucket.png)

### Squad Values by Team
![Squad Values](images/squad_values.png)

### Correlation Heatmap
![Correlation Heatmap](images/correlation_heatmap.png)

### Feature Importance (Best Model)
![Feature Importance](images/feature_importance.png)

### Predicted vs Actual Market Value
![Predicted vs Actual](images/predicted_vs_actual.png)

---

## Future Work

- Incorporate event-level data (xG, xA, progressive carries) for deeper performance modeling
- Add temporal analysis: how do market values evolve across transfer windows?
- Build an interactive Streamlit dashboard for club-level exploration
- Extend to multi-league comparison (LaLiga vs Premier League vs Serie A)

---

## Author

**Juan Sebastian Marcial**
Data & Business Analytics — IE University
[GitHub](https://github.com/JuanseMarcial) | [LinkedIn](https://www.linkedin.com/in/juan-sebastian-marcial-piedra)

---

## License

This project is licensed under the MIT License. See `LICENSE` for details.
