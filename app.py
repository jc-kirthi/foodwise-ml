
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Import the core functions from our module
from foodwise_core_functions import predict_demand, analyze_preparation

# --- Configuration & Paths (UPDATED FOR DEPLOYMENT) ---
# Relative paths for deployment
MODELS_PATH = Path("models")
DATA_RAW_PATH = Path('data') / 'Food demand.csv'
DATA_PROCESSED_PATH = Path('data') / 'foodwise_features.csv' # This file is not copied for deployment based on instructions, but it's referenced in the original app.py

# --- Load Cached Resources ---
@st.cache_resource
def load_model_and_data():
    xgb_model = joblib.load(MODELS_PATH / "xgboost_model.joblib")
    
    # The deployment instructions did not include `foodwise_features.csv` in the `data/` folder.
    # For the purpose of deployment without this file, we will load `raw_df` and
    # extract unique values for `center_id` and `meal_id` directly from it, 
    # and also use `raw_df` as a proxy for `df_train_for_dtypes` if `foodwise_features.csv` is not present.
    # If `foodwise_features.csv` *was* intended to be deployed, this logic would need adjustment.

    raw_df = pd.read_csv(DATA_RAW_PATH)
    imputation_values = joblib.load(MODELS_PATH / "imputation_values.joblib")
    model_features = xgb_model.get_booster().feature_names
    
    # Using raw_df for unique values as df_train is not deployed
    # Note: If the model's categorical features were trained on specific categories from df_train that are NOT just unique values from raw_df,
    # this approach might introduce issues. However, based on previous steps, df_train's categorical columns were just int64, 
    # and then converted to categorical with `unique()` values at prediction time. So raw_df unique values should be sufficient.
    df_train_for_dtypes = raw_df.copy() # Using raw_df as a proxy for df_train_for_dtypes for categorical handling.
    
    return xgb_model, raw_df, imputation_values, model_features, df_train_for_dtypes

xgb_model, raw_df, imputation_values, model_features, df_train_for_dtypes = load_model_and_data()

# Get unique center_ids and meal_ids for input selection
unique_center_ids = sorted(raw_df['center_id'].unique().tolist())
unique_meal_ids = sorted(raw_df['meal_id'].unique().tolist())

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("FOODWISE")
st.subheader("ML-Based Food Demand & Waste Intelligence Platform")

st.write("---")

# Input Section
st.header("1. Input Parameters for Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    week = st.number_input("Week (e.g., 101 for next week's prediction)", min_value=1, value=101)
    # Pre-fill with a common one if it exists in unique IDs
    meal_id_default_idx = unique_meal_ids.index(2956) if 2956 in unique_meal_ids else 0
    meal_id = st.selectbox("Meal ID", options=unique_meal_ids, index=meal_id_default_idx)

    center_id_default_idx = unique_center_ids.index(104) if 104 in unique_center_ids else 0
    center_id = st.selectbox("Center ID", options=unique_center_ids, index=center_id_default_idx)

with col2:
    checkout_price = st.number_input("Checkout Price", min_value=0.0, value=486.03, format="%.2f")
    base_price = st.number_input("Base Price", min_value=0.0, value=583.03, format="%.2f")
    planned_preparation = st.number_input("Planned Preparation (Units)", min_value=0, value=130)

with col3:
    emailer_for_promotion = st.checkbox("Emailer for Promotion", value=False)
    homepage_featured = st.checkbox("Homepage Featured", value=False)

# Convert checkboxes to 0/1
emr_for_promo_val = 1 if emailer_for_promotion else 0
hpg_featured_val = 1 if homepage_featured else 0

st.write("---")

# Predict Button
if st.button("Predict Demand and Analyze Preparation", type="primary"):
    st.subheader("2. Prediction & Decision Analysis Results")

    # Call predict_demand
    predicted_demand_raw, _ = predict_demand(
        week=week,
        meal_id=meal_id,
        center_id=center_id,
        checkout_price=checkout_price,
        base_price=base_price,
        emailer_for_promotion=emr_for_promo_val,
        homepage_featured=hpg_featured_val,
        raw_df_historical=raw_df,
        imputation_values_dict=imputation_values,
        model_features_order=model_features,
        model=xgb_model,
        df_train_for_dtypes=df_train_for_dtypes
    )

    predicted_demand = round(predicted_demand_raw) # Round to nearest integer for display

    # Call analyze_preparation
    analysis_results = analyze_preparation(predicted_demand, planned_preparation)

    # Display Results
    st.markdown("### Demand Prediction:")
    st.metric("Predicted Demand", f"{predicted_demand:.0f} units")

    st.markdown("### Preparation Analysis:")
    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Recommended Preparation", f"{analysis_results['recommended_preparation']:.0f} units")
        st.metric("Planned Preparation", f"{analysis_results['planned_preparation']:.0f} units")
    with col_res2:
        st.metric("Potential Surplus", f"{analysis_results['potential_surplus']:.0f} units")
        st.metric("Surplus Percentage", f"{analysis_results['surplus_percentage']:.2f}%")
        risk_status = st.success if analysis_results['waste_risk'] == "LOW" else (st.warning if analysis_results['waste_risk'] == "MEDIUM" else st.error)
        risk_status(f"Waste Risk: {analysis_results['waste_risk']}")
    with col_res3:
        st.metric("Potential Shortage", f"{analysis_results['potential_shortage']:.0f} units")
        st.metric("Shortage Percentage", f"{analysis_results['shortage_percentage']:.2f}%")
        risk_status = st.success if analysis_results['shortage_risk'] == "LOW" else (st.warning if analysis_results['shortage_risk'] == "MEDIUM" else st.error)
        risk_status(f"Shortage Risk: {analysis_results['shortage_risk']}")

st.write("---")

st.header("3. Model Performance")
st.markdown("**Held-Out Test Performance:**")
st.write(f"MAE: 103.70")
st.write(f"RMSE: 158.78")
st.write(f"MAPE: 87.17%")
st.write(f"R²: 0.72")

st.write("---")

st.info("""
    **Disclaimer:** FoodWise currently estimates demand and potential over-preparation/shortage risk.
    The current dataset does not contain direct food-waste measurements, so waste risk
    is a decision-support estimate rather than a direct prediction of actual food waste.
""")
