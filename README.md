# E-Commerce Inventory & Dynamic Pricing Optimization

An end-to-end data science and BI pipeline built on a 50,000-SKU women's apparel catalog. The project moves beyond descriptive EDA into engineered business metrics, a dynamic pricing engine, an automated restocking system, and two machine learning models — all exported into a single Tableau-ready dataset.

## 📁 Repository Contents

| File | Description |
|---|---|
| `eda_fixed.ipynb` | Full analysis notebook — data audit, feature engineering, pricing/restocking logic, ML models, and export pipeline. |
| `women_clothing_50k.csv` | Raw source dataset (50,000 SKUs, 15 attributes: price, stock, sales, rating, category, etc.). |
| `clothing_analytics_tableau_ready.csv` | Final enriched dataset (30 columns) with all engineered metrics and model predictions, ready for BI import. |

## 🎯 Objective

Analyze seasonal demand shifts, engineer dynamic margin-expansion formulas based on return risk, build an automated inventory restock queue, and train predictive ML models for demand forecasting and return-risk classification.

## 🔍 Analysis Walkthrough

1. **Data Ingestion & Quality Audit** — Verified 0 missing values, 0 duplicates, and no invalid domain values (negative prices/stock, out-of-range ratings) across all 50,000 records.
2. **Seasonal Demand Analysis** — Found a structural seasonal pivot: T-Shirts dominate Summer (~503K units, ~0 in Winter) while Sweaters and Coats drive Winter demand; Jeans sell consistently year-round (~178K units each season).
3. **Bayesian Average Rating (BAR)** — Applied Bayesian shrinkage to smooth product ratings, pulling low-review-count items toward the global mean (~4.17) instead of letting sparse reviews distort rankings.
4. **Margin Expansion Engine** — For the 202 SKUs with return rates ≤ 5%, computed dynamic anchor pricing that allows a displayed 20% discount while increasing net captured revenue per unit (e.g., +$11.53/unit on a sample Coats SKU).
5. **Warehouse Restocking & Dead Stock Audit** — Flagged 147 SKUs as urgent restock priorities (urgency score > 1000) and identified 688 dead-stock SKUs tying up ~$3.81M in capital.
6. **ML Preprocessing** — Built a `ColumnTransformer` + `OneHotEncoder` pipeline over categorical and numerical features to prevent data leakage.
7. **Model 1 — Demand Forecasting** — Random Forest Regressor predicting `units_sold` (R² ≈ 0.80, MAE ≈ 76 units). Noted `review_count` likely introduces leakage/reverse causation as the top feature.
8. **Model 2 — Return Risk Classifier** — Random Forest Classifier predicting high-return-risk SKUs (ROC-AUC ≈ 0.77, precision ≈ 0.81, recall ≈ 0.54).
9. **Feature Importance Analysis** — Identified `review_count`, `BAR`, `price_usd`, `stock_quantity`, and `discount_percent` as the leading demand drivers, with a caveat on the leakage risk from `review_count`.
10. **Export for Tableau** — Combined raw attributes + 15 engineered features (BAR, urgency score, dead-stock flag, target markup, predicted demand, return risk probability, etc.) into `clothing_analytics_tableau_ready.csv` to power an Inventory Command Center dashboard.

## 🛠️ Tech Stack

- **Python** (Jupyter Notebook)
- **pandas**, **numpy** — data manipulation
- **matplotlib**, **seaborn** — visualization
- **scikit-learn** — preprocessing pipelines, Random Forest models, evaluation metrics
- **Tableau** — downstream dashboarding

## 🚀 Getting Started

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### Run the notebook
```bash
git clone https://github.com/srikar-up/eda-project.git
cd eda-project
jupyter notebook eda_fixed.ipynb
```

### Use the Tableau-ready data
Open `clothing_analytics_tableau_ready.csv` directly in Tableau Desktop/Public to build the Inventory Command Center dashboard.

## ⚠️ Known Caveats

- The demand forecasting model's top feature, `review_count`, is likely a leakage artifact (review counts accumulate *because* a product already sold well), so its importance reflects reverse causation rather than a true actionable driver.
- The return-risk classifier has decent precision (~0.81) but weaker recall (~0.54) — it misses close to half of true high-return items, so it's better used to *prioritize* review rather than as a standalone gatekeeper.

## 📄 License

This project is licensed under the [MIT License](LICENSE).
