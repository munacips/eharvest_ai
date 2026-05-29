import joblib


dynamic_pricing_model = joblib.load("ai_training/dynamic_pricing_model.pkl")
model_columns = joblib.load("ai_training/model_columns.pkl")

# Price forecasting models (one model per commodity)
price_forecast_models = joblib.load("new_ai_training/forecast_models.pkl")
price_forecast_features = joblib.load("new_ai_training/forecast_features.pkl")
price_forecast_commodities = joblib.load(
    "new_ai_training/forecast_commodity_list.pkl")

# Quantity forecasting models (demand and supply)
demand_models = joblib.load("new_ai_training/demand_models.pkl")
demand_features = joblib.load("new_ai_training/demand_features.pkl")
demand_commodity_list = joblib.load(
    "new_ai_training/demand_commodity_list.pkl")

supply_models = joblib.load("new_ai_training/supply_models.pkl")
supply_features = joblib.load("new_ai_training/supply_features.pkl")
supply_commodity_list = joblib.load(
    "new_ai_training/supply_commodity_list.pkl")
