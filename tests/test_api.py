# How to run: pytest -q

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class DummyPricingModel:
    def predict(self, X):
        return np.array([1.25] * len(X))


class DummyForecastModel:
    def make_future_dataframe(self, periods=30):
        start = pd.Timestamp("2024-01-01")
        return pd.DataFrame({"ds": pd.date_range(start, periods=periods + 1, freq="D")})

    def predict(self, future):
        return pd.DataFrame({"ds": future["ds"], "yhat": np.linspace(10, 20, len(future))})


def _install_fake_joblib():
    def fake_load(path):
        path_str = str(path)
        if "dynamic_pricing_model.pkl" in path_str:
            return DummyPricingModel()
        if "model_columns.pkl" in path_str:
            return [
                "month",
                "year",
                "latitude",
                "longitude",
                "market_id",
                "commodity_id",
                "commodity_maize",
                "market_harare",
                "category_cereals",
                "unit_KG",
                "currency_USD",
                "priceflag_actual",
                "pricetype_Retail",
                "admin1_Manicaland",
                "admin2_Mutare",
            ]
        if "demand_forecast_model.pkl" in path_str:
            return DummyForecastModel()
        if "forecast_features.pkl" in path_str:
            return ["commodity_maize", "admin1_Manicaland"]
        raise FileNotFoundError(path)

    fake_joblib = types.SimpleNamespace(load=fake_load)
    sys.modules["joblib"] = fake_joblib


_install_fake_joblib()
import main  # noqa: E402


client = TestClient(main.app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_pricing_schema():
    resp = client.get("/pricing/schema")
    assert resp.status_code == 200
    body = resp.json()
    assert "model_columns" in body
    assert "required_fields" in body


def test_predict_price():
    payload = {
        "commodity": "maize",
        "market": "harare",
        "category": "cereals",
        "unit": "KG",
        "month": 11,
        "latitude": -17.8,
        "longitude": 31.0,
        "currency": "USD",
        "priceflag": "actual",
        "admin1": "Manicaland",
        "admin2": "Mutare",
        "pricetype": "Retail",
        "market_id": 1,
        "commodity_id": 51,
        "year": 2024,
    }
    resp = client.post("/predict-price", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["currency"] == "USD"


def test_predict_price_batch():
    payload = {
        "items": [
            {
                "commodity": "maize",
                "market": "harare",
                "category": "cereals",
                "unit": "KG",
                "month": 11,
                "latitude": -17.8,
                "longitude": 31.0,
                "currency": "USD",
                "priceflag": "actual",
                "admin1": "Manicaland",
                "admin2": "Mutare",
                "pricetype": "Retail",
                "market_id": 1,
                "commodity_id": 51,
                "year": 2024,
            },
            {
                "commodity": "beans",
                "market": "mutare",
                "category": "pulses",
                "unit": "KG",
                "month": 12,
                "latitude": -18.9,
                "longitude": 32.6,
                "currency": "USD",
                "priceflag": "actual",
                "admin1": "Manicaland",
                "admin2": "Mutare",
                "pricetype": "Retail",
                "market_id": 2,
                "commodity_id": 52,
                "year": 2024,
            },
        ]
    }
    resp = client.post("/pricing/batch", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 2


def test_forecast_endpoint():
    resp = client.get(
        "/forecast/maize?periods=5&region=Manicaland&visual=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["commodity"] == "maize"
    assert body["region"] == "Manicaland"
    assert len(body["forecast"]) == 5
    assert "visual" in body


def test_demand_supply_forecast():
    payload = {
        "region": "Manicaland",
        "season": "rainy",
        "periods": 3,
        "historical_sales": [
            {"date": "2024-01-01", "commodity": "maize",
                "quantity": 120, "region": "Manicaland"},
            {"date": "2024-02-01", "commodity": "maize",
                "quantity": 135, "region": "Manicaland"},
            {"date": "2024-01-01", "commodity": "beans",
                "quantity": 80, "region": "Manicaland"},
        ],
        "weather": [
            {"date": "2024-01-01", "rainfall_mm": 85,
                "temperature_c": 24, "region": "Manicaland"},
            {"date": "2024-02-01", "rainfall_mm": 90,
                "temperature_c": 25, "region": "Manicaland"},
        ],
        "market_data": [
            {"date": "2024-01-01", "commodity": "maize", "price": 0.35,
                "market": "Harare", "region": "Manicaland"},
            {"date": "2024-02-01", "commodity": "maize", "price": 0.38,
                "market": "Harare", "region": "Manicaland"},
        ],
    }
    resp = client.post("/forecast/demand-supply", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["region"] == "Manicaland"
    assert len(body["forecasts"]) >= 1
    assert len(body["forecasts"][0]["demand"]) == 3


def test_prescriptive_recommendations():
    payload = {
        "region": "Manicaland",
        "month": 11,
        "budget_usd": 600,
        "climate": {"rainfall_mm": 700, "temperature_c": 25},
        "demand_forecast": [{"commodity": "maize", "expected_demand": 140}],
        "top_n": 2,
    }
    resp = client.post("/recommendations/prescriptive", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["region"] == "Manicaland"
    assert len(body["recommendations"]) == 2
