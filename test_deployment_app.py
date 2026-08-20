
import sys
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# --- Configuration & Paths for Deployment Test ---
# The deployment path is the current directory for this test script
deployment_root = Path('.')
MODELS_PATH = deployment_root / "models"
DATA_RAW_PATH = deployment_root / 'data' / 'Food demand.csv'

# Add the deployment root to sys.path so we can import foodwise_core_functions
sys.path.insert(0, str(deployment_root))

# Import the core functions from our module within the deployment folder
# This assumes foodwise_core_functions.py is copied to the deployment root
from foodwise_core_functions import predict_demand, analyze_preparation

print("--- Starting Deployment Test --- ")

# --- Load Resources and Verify --- 

# 1. Model loads.
try:
    xgb_model_test = joblib.load(MODELS_PATH / "xgboost_model.joblib")
    print("✅ 1. Model loads: xgboost_model.joblib loaded successfully.")
except Exception as e:
    print(f"❌ 1. Model failed to load: {e}")
    sys.exit(1)

# 2. Imputation values load.
try:
    imputation_values_test = joblib.load(MODELS_PATH / "imputation_values.joblib")
    print("✅ 2. Imputation values load: imputation_values.joblib loaded successfully.")
except Exception as e:
    print(f"❌ 2. Imputation values failed to load: {e}")
    sys.exit(1)

# 3. Dataset loads.
try:
    raw_df_test = pd.read_csv(DATA_RAW_PATH)
    print("✅ 3. Dataset loads: Food demand.csv loaded successfully.")
except Exception as e:
    print(f"❌ 3. Dataset failed to load: {e}")
    sys.exit(1)

# Get model features from the loaded model
model_features_test = xgb_model_test.get_booster().feature_names

# Create a proxy for df_train_for_dtypes as in app.py
df_train_for_dtypes_test = raw_df_test.copy()

print(f"✅ Model has {len(model_features_test)} expected features.")

# --- Simulate one valid prediction flow --- 
print("\n--- Simulating one valid prediction --- ")

# Select a valid row for testing from the loaded raw_df_test
sample_row = raw_df_test[raw_df_test['week'] == 100].iloc[0]

prediction_week = sample_row['week'] + 1 # Predict for the next week
meal_id_input = sample_row['meal_id']
center_id_input = sample_row['center_id']
checkout_price_input = sample_row['checkout_price']
base_price_input = sample_row['base_price']
emr_for_promo_input = sample_row['emailer_for_promotion']
hpg_featured_input = sample_row['homepage_featured']
planned_preparation_input = 130 # Arbitrary planned preparation for analysis

# 4. predict_demand() works.
try:
    predicted_demand_raw, feature_df_sent_to_model = predict_demand(
        week=prediction_week,
        meal_id=meal_id_input,
        center_id=center_id_input,
        checkout_price=checkout_price_input,
        base_price=base_price_input,
        emailer_for_promotion=emr_for_promo_input,
        homepage_featured=hpg_featured_input,
        raw_df_historical=raw_df_test,
        imputation_values_dict=imputation_values_test,
        model_features_order=model_features_test,
        model=xgb_model_test,
        df_train_for_dtypes=df_train_for_dtypes_test
    )
    predicted_demand = round(predicted_demand_raw)
    print("✅ 4. predict_demand() works: Prediction generated successfully.")
except Exception as e:
    print(f"❌ 4. predict_demand() failed: {e}")
    sys.exit(1)

# 5. analyze_preparation() works.
try:
    analysis_results = analyze_preparation(predicted_demand, planned_preparation_input)
    print("✅ 5. analyze_preparation() works: Analysis performed successfully.")
except Exception as e:
    print(f"❌ 5. analyze_preparation() failed: {e}")
    sys.exit(1)

# 6. XGBoost receives exactly the 18 expected features.
if feature_df_sent_to_model.columns.tolist() == model_features_test:
    print("✅ 6. XGBoost receives exactly the 18 expected features: Feature columns and order match.")
else:
    print(f"❌ 6. XGBoost feature mismatch. Expected: {model_features_test}, Got: {feature_df_sent_to_model.columns.tolist()}")
    sys.exit(1)

# 7. One valid prediction succeeds (implied by successful predict_demand and analysis).
print(f"✅ 7. One valid prediction succeeds. Predicted Demand: {predicted_demand:.0f} units. Waste Risk: {analysis_results['waste_risk']}")

print("\n--- Deployment Test Complete and Successful! --- ")

# Clean up sys.path
sys.path.pop(0)
