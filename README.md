# International Logistics & Supply Chain Risk Prediction 🚛 Predict Lead Times & Disruption Risks

An end-to-end Machine Learning pipeline designed to evaluate supply chain risks, classify disruption probabilities, and forecast delivery lead times using operational, geographical, and weather metrics.

---

## 📌 Features & Key Methodologies

- **Data Cleaning & Pipeline Integration:** Handled missing values, date-time transformations, categorical encodings, and outlier detection using the Interquartile Range (IQR).
- **Feature Engineering:** Developed custom metrics including `dangerous_percentage` (combining weather and geopolitical risk factors) and transport speed ratios.
- **Data Leakage Prevention:** Applied `RobustScaler` strictly after train-test splitting to ensure unbiased evaluation.
- **Disruption Risk Classification:**
  - Models: Logistic Regression, Random Forest, XGBoost
  - Evaluation: Confusion Matrices, Classification Reports, and ROC-AUC Scores.
- **Lead Time Regression:**
  - Models: Linear Regression, Random Forest Regressor, XGBoost, LightGBM
  - Metrics: $R^2$ Score, MAE, and RMSE (evaluated in actual days).

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, LightGBM, Seaborn, Matplotlib

---

## 🚀 How to Run
```bash
pip install pandas numpy scikit-learn xgboost lightgbm seaborn matplotlib
python supply_chain_ml_pipeline.py
