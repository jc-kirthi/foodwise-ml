
import pandas as pd
import numpy as np
import math

def predict_demand(
    week,
    meal_id,
    center_id,
    checkout_price,
    base_price,
    emailer_for_promotion,
    homepage_featured,
    raw_df_historical,
    imputation_values_dict,
    model_features_order,
    model,
    df_train_for_dtypes
):
    discount_amount = base_price - checkout_price
    discount_percentage = 0.0
    if base_price != 0:
        discount_percentage = ((base_price - checkout_price) / base_price) * 100

    current_prediction_features = {
        'center_id': center_id,
        'meal_id': meal_id,
        'checkout_price': checkout_price,
        'base_price': base_price,
        'emailer_for_promotion': emailer_for_promotion,
        'homepage_featured': homepage_featured,
        'discount_amount': discount_amount,
        'discount_percentage': discount_percentage
    }

    historical_df = raw_df_historical[raw_df_historical['week'] < week].copy()

    meal_weekly_orders = historical_df.groupby(['week', 'meal_id'])['num_orders'].sum().reset_index()
    center_weekly_orders = historical_df.groupby(['week', 'center_id'])['num_orders'].sum().reset_index()

    specific_meal_orders = meal_weekly_orders[meal_weekly_orders['meal_id'] == meal_id].set_index('week')['num_orders']
    current_prediction_features['meal_lag_1'] = specific_meal_orders.get(int(week) - 1, np.nan)
    current_prediction_features['meal_lag_2'] = specific_meal_orders.get(int(week) - 2, np.nan)
    current_prediction_features['meal_lag_4'] = specific_meal_orders.get(int(week) - 4, np.nan)

    meal_rolling_data_weeks = range(int(max(1, week - 8)), int(week))
    meal_rolling_data = specific_meal_orders.reindex(meal_rolling_data_weeks)
    current_prediction_features['meal_rolling_mean_4'] = meal_rolling_data.loc[meal_rolling_data.index.intersection(range(int(max(1, week - 4)), int(week)))].mean()
    current_prediction_features['meal_rolling_mean_8'] = meal_rolling_data.loc[meal_rolling_data.index.intersection(range(int(max(1, week - 8)), int(week)))].mean()

    specific_center_orders = center_weekly_orders[center_weekly_orders['center_id'] == center_id].set_index('week')['num_orders']
    current_prediction_features['center_lag_1'] = specific_center_orders.get(int(week) - 1, np.nan)
    current_prediction_features['center_lag_2'] = specific_center_orders.get(int(week) - 2, np.nan)
    current_prediction_features['center_lag_4'] = specific_center_orders.get(int(week) - 4, np.nan)

    center_rolling_data_weeks = range(int(max(1, week - 8)), int(week))
    center_rolling_data = specific_center_orders.reindex(center_rolling_data_weeks)
    current_prediction_features['center_rolling_mean_4'] = center_rolling_data.loc[center_rolling_data.index.intersection(range(int(max(1, week - 4)), int(week)))].mean()
    current_prediction_features['center_rolling_mean_8'] = center_rolling_data.loc[center_rolling_data.index.intersection(range(int(max(1, week - 8)), int(week)))].mean()

    feature_series = pd.Series(current_prediction_features)

    for feature_name, imputation_value in imputation_values_dict.items():
        if pd.isna(feature_series.get(feature_name)):
            feature_series[feature_name] = imputation_value

    feature_df = pd.DataFrame([feature_series[model_features_order].values], columns=model_features_order)

    feature_df['center_id'] = pd.Categorical(feature_df['center_id'].astype(int), categories=df_train_for_dtypes['center_id'].unique().astype(int))
    feature_df['meal_id'] = pd.Categorical(feature_df['meal_id'].astype(int), categories=df_train_for_dtypes['meal_id'].unique().astype(int))

    for col in ['checkout_price', 'base_price', 'discount_amount', 'discount_percentage']:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype(float)
    for col in ['emailer_for_promotion', 'homepage_featured']:
        if col in feature_df.columns:
            feature_df[col] = feature_df[col].astype(int)

    predicted_demand = model.predict(feature_df)[0]

    return predicted_demand, feature_df

def analyze_preparation(
    predicted_demand,
    planned_preparation,
    buffer_percentage=5
):
    recommended_preparation = math.ceil(predicted_demand * (1 + buffer_percentage / 100))

    potential_surplus = max(0, planned_preparation - predicted_demand)
    surplus_percentage = 0.0
    if planned_preparation > 0:
        surplus_percentage = (potential_surplus / planned_preparation) * 100

    potential_shortage = max(0, predicted_demand - planned_preparation)
    shortage_percentage = 0.0
    if predicted_demand > 0:
        shortage_percentage = (potential_shortage / predicted_demand) * 100

    waste_risk = "LOW"
    if surplus_percentage > 20:
        waste_risk = "HIGH"
    elif surplus_percentage > 5:
        waste_risk = "MEDIUM"

    shortage_risk = "LOW"
    if shortage_percentage > 20:
        shortage_risk = "HIGH"
    elif shortage_percentage > 5:
        shortage_risk = "MEDIUM"

    return {
        "predicted_demand": predicted_demand,
        "planned_preparation": planned_preparation,
        "recommended_preparation": recommended_preparation,
        "potential_surplus": potential_surplus,
        "surplus_percentage": surplus_percentage,
        "waste_risk": waste_risk,
        "potential_shortage": potential_shortage,
        "shortage_percentage": shortage_percentage,
        "shortage_risk": shortage_risk
    }
