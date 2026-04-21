import joblib


dynamic_pricing_model = joblib.load("ai_training/dynamic_pricing_model.pkl")
model_columns = joblib.load("ai_training/model_columns.pkl")
forecast_model = joblib.load("ai_training/demand_forecast_model.pkl")
commodity_cols = joblib.load("ai_training/forecast_features.pkl")
