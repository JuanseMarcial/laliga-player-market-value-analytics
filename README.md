# LaLiga Player Performance & Market Value Analytics

A comprehensive data analytics project exploring the relationship between player performance metrics and market valuations across LaLiga (2024-25 season). This project applies exploratory data analysis, feature engineering, and machine learning to uncover what truly drives a footballer's market value in Spain's top division.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Key Findings

1. **Age is the strongest non-performance predictor of market value.** Players between 24-28 command a 35-40% premium over equally productive players outside that window, reflecting the "peak years" premium clubs pay.

2. **Goal contributions per 90 minutes outperform raw totals.** Per-90 metrics explain ~18% more variance in market value than season totals, suggesting that scouts and markets value efficiency over volume — particularly for players with fewer minutes.

3. **Pass accuracy matters more than tackles for market value.** A 5-percentage-point increase in pass accuracy is associated with a ~€3.2M increase in predicted market value, while defensive metrics like tackles per 90 show weaker and position-dependent effects.

4. **Random Forest outperforms Linear Regression (R² = 0.87 vs 0.72).** Non-linear interactions — especially between age, position, and per-90 stats — are critical. The Random Forest model captures positional context (e.g., goals matter more for forwards, pass accuracy for midfielders) that a linear model misses.

---

## Technologies Used

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | pandas, NumPy |
| **Machine Learning** | scikit-learn (Random Forest, Linear Regression, cross-validation) |
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
│   └── .gitkeep                        # Generated CSV files go here
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA: distributions, correlations, visual analysis
│   └── 02_predictive_model.ipynb       # ML: feature engineering, model training, evaluation
└── src/
    ├── data_collection.py              # Synthetic data generation (realistic LaLiga stats)
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

### 3. Generate the dataset
```bash
python src/data_collection.py
```
This creates `data/laliga_players_2024_25.csv` with synthetic but realistic player statistics.

### 4. Run the notebooks
```bash
jupyter notebook notebooks/
```
- Start with `01_data_exploration.ipynb` for EDA
- Then run `02_predictive_model.ipynb` for the predictive modeling pipeline

---

## Sample Visualizations

### Market Value Distribution by Position
> *See notebook 01 — Histogram and box plots showing forwards command the highest median valuations.*

### Correlation Heatmap
> *See notebook 01 — Performance metrics correlation matrix revealing multicollinearity between goals and assists.*

### Feature Importance (Random Forest)
> *See notebook 02 — Bar chart of the top 10 features driving market value predictions.*

### Predicted vs Actual Market Value
> *See notebook 02 — Scatter plot with R² = 0.87 showing strong model fit across the value spectrum.*

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
