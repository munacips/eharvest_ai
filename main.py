from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from typing import List

app = FastAPI(title="eHarvest AI API", description="API for eHarvest AI services", version="1.0.0")

dynamic_pricing_model = joblib.load('ai_training/dynamic_pricing_model.pkl')
model_columns = joblib.load('ai_training/model_columns.pkl')
forecast_model = joblib.load('ai_training/demand_forecast_model.pkl')
commodity_cols = joblib.load('ai_training/forecast_features.pkl')

#['commodity', 'market', 'category', 'unit', 'month', 'latitude', 'longitude', 'currency', 'priceflag']
class PricePredictionRequest(BaseModel):
    commodity: str
    market: str
    category: str
    unit: str
    month: int
    latitude: float
    longitude: float
    currency: str
    priceflag: int


@app.post("/predict-price")
async def predict_price(request: PricePredictionRequest):
    #Convert request to a dict and then to a dataframe
    input_data = pd.DataFrame([request.dict()])

    #One-Hot Encode the categorical features
    input_encoded = pd.get_dummies(input_data)

    #Reindex the columns to match the model's expected input
    #this adds any missing columns with 0 values and ensures the order is correct
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

    #Make prediction using the model
    prediction = dynamic_pricing_model.predict(final_input)

    return {
        "suggested_price": round(float(prediction[0]), 2),
        "currency": request.currency,
        "status" : "success"
    }


@app.get("/forecast/{commodity}")
async def get_forecast(commodity: str, periods: int = 30):
    # 1. Create future dates
    future = forecast_model.make_future_dataframe(periods=periods)
    
    # 2. Set the "Switch" for the requested commodity
    target_col = f"commodity_{commodity}"
    
    for col in commodity_cols:
        future[col] = 1 if col == target_col else 0
            
    # 3. Predict
    forecast = forecast_model.predict(future)
    
    # Return last 'n' days
    result = forecast[['ds', 'yhat']].tail(periods).to_dict(orient="records")
    return {"commodity": commodity, "forecast": result}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "API is running. Use /predict-price for price predictions and /forecast/commodity for demand forecasts."}