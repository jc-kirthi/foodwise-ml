# 🍽️ FoodWise — ML-Based Food Demand & Waste Intelligence Platform

FoodWise is an ML-powered decision-support platform that helps restaurants, cafeterias, hostels, and institutional kitchens make better food-preparation decisions.

Instead of relying only on historical averages or manual intuition, FoodWise predicts expected food demand and converts the prediction into actionable preparation, surplus-risk, and shortage-risk insights.

> **Important:** The current dataset does not contain direct food-waste measurements. Therefore, FoodWise estimates **potential over-preparation/surplus risk** rather than directly predicting actual food waste.

---

## 🚀 Live Demo

**Live Application:**  
https://foodwise-ml.streamlit.app

**GitHub Repository:**  
https://github.com/jc-kirthi/foodwise-ml

---

## 🎯 Problem Statement

Restaurants and institutional kitchens must decide how much food to prepare before actual demand is known.

Over-preparation can lead to:

- Food waste
- Higher ingredient costs
- Resource wastage

Under-preparation can lead to:

- Stock-outs
- Unsatisfied customers
- Lost revenue

FoodWise addresses this problem by forecasting demand at the meal and fulfillment-center level and converting the forecast into preparation recommendations and risk indicators.

---

## 💡 Solution

FoodWise follows this pipeline:

User Inputs  
↓  
Historical Feature Engineering  
↓  
XGBoost Demand Forecast  
↓  
Predicted Demand  
↓  
Decision Engine  
↓  
Recommended Preparation  
↓  
Surplus / Shortage Analysis  
↓  
Waste & Shortage Risk  
↓  
Explainable AI using SHAP

---

## 🧠 Machine Learning Approach

### Target

The target variable is:

`num_orders`

representing the number of orders for a meal at a fulfillment center during a particular week.

### Prediction Unit

Each prediction corresponds to:

`meal_id + center_id + week`

### Models Evaluated

FoodWise evaluated multiple approaches:

- Mean-based baseline
- Random Forest Regressor
- XGBoost Regressor

XGBoost was selected as the final forecasting model based on held-out test performance.

---

## 📊 Model Performance

### XGBoost — Held-Out Test Set

| Metric | Result |
|---|---:|
| MAE | 103.70 |
| RMSE | 158.78 |
| MAPE | 87.17% |
| R² | 0.72 |

The model was evaluated using a chronological train/validation/test split to preserve the temporal nature of the problem.

---

## ⏳ Temporal Validation

The dataset contains 145 weeks.

The project uses a chronological split:

- **Training:** Weeks 1–100
- **Validation:** Weeks 101–125
- **Testing:** Weeks 126–145

This prevents future observations from influencing earlier predictions.

---

## 🛠️ Feature Engineering

FoodWise creates price and historical demand features.

### Price Features

- `discount_amount`
- `discount_percentage`

### Meal-Level Historical Features

- `meal_lag_1`
- `meal_lag_2`
- `meal_lag_4`
- `meal_rolling_mean_4`
- `meal_rolling_mean_8`

### Center-Level Historical Features

- `center_lag_1`
- `center_lag_2`
- `center_lag_4`
- `center_rolling_mean_4`
- `center_rolling_mean_8`

Historical features are constructed using only information available before the prediction week.

---

## 🔐 Leakage Prevention

Temporal leakage was treated as a major design concern.

For a prediction at week `N`, historical demand features only use:

`week < N`

The prediction week's `num_orders` and all future observations are excluded from historical feature calculations.

Rolling features use shifted historical values so that the current prediction week is never included.

---

## 🔍 Explainable AI

FoodWise uses **SHAP (SHapley Additive exPlanations)** to explain model predictions.

Instead of treating the XGBoost model as a black box, SHAP helps identify which features contributed most strongly to an individual prediction.

The system provides:

- Global feature importance
- Individual prediction explanations
- Positive and negative feature contributions

SHAP explanations describe model behavior and should not be interpreted as causal relationships.

---

## 🍛 Food Preparation Decision Engine

FoodWise goes beyond demand prediction.

Given:

- Predicted demand
- Planned preparation

the system calculates:

### Recommended Preparation

A configurable safety buffer is applied to predicted demand.

### Potential Surplus

If planned preparation exceeds predicted demand:

`Potential Surplus = Planned Preparation − Predicted Demand`

### Potential Shortage

If planned preparation is below predicted demand:

`Potential Shortage = Predicted Demand − Planned Preparation`

---

## ⚠️ Risk Engine

FoodWise uses transparent business rules to classify:

### Surplus / Over-Preparation Risk

- LOW
- MEDIUM
- HIGH

### Shortage Risk

- LOW
- MEDIUM
- HIGH

The risk thresholds are prototype business rules rather than ML predictions.

---

## 🖥️ Application

The final application is built using **Streamlit**.

Users provide:

- Week
- Meal
- Fulfillment center
- Checkout price
- Base price
- Email promotion status
- Homepage feature status
- Planned preparation quantity

The application returns:

- Predicted demand
- Recommended preparation
- Potential surplus
- Surplus percentage
- Waste/over-preparation risk
- Potential shortage
- Shortage percentage
- Shortage risk
- Model explanation

---

## 🏗️ Project Architecture

```text
                         FOODWISE
                            │
                            ▼
                    Streamlit Interface
                            │
                            ▼
                       User Inputs
                            │
                            ▼
                  Feature Engineering
                            │
                            ▼
                      XGBoost Model
                            │
                            ▼
                    Predicted Demand
                            │
                            ▼
                    Decision Engine
                       /         \
                      /           \
                     ▼             ▼
             Surplus Risk     Shortage Risk
                     \             /
                      \           /
                       ▼         ▼
                    Preparation
                    Recommendation
                            │
                            ▼
                     SHAP Explanation

```

## 📁 Project Structure
foodwise-ml/
│
├── app.py
├── foodwise_core_functions.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Food demand.csv
│
└── models/
    ├── xgboost_model.joblib
    └── imputation_values.joblib
