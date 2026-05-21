"""API tests for the eharvest application."""

# How to run: pytest -q

import json
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
from app.services import integrations as integrations_service  # noqa: E402
from app.services import pricing as pricing_service  # noqa: E402
from app.services import trust as trust_service  # noqa: E402
from app.routers import forecasting as forecasting_router  # noqa: E402
from app.routers import pricing as pricing_router  # noqa: E402
from app.routers import system as system_router  # noqa: E402


client = TestClient(main.app)


def _load_real_forecast_assets():
    fake_joblib = sys.modules.get("joblib")
    sys.modules.pop("joblib", None)
    try:
        import joblib as real_joblib

        forecast_model = real_joblib.load(
            ROOT / "ai_training" / "demand_forecast_model.pkl")
        commodity_cols = real_joblib.load(
            ROOT / "ai_training" / "forecast_features.pkl")
    finally:
        if fake_joblib is not None:
            sys.modules["joblib"] = fake_joblib

    return forecast_model, commodity_cols


def test_root():
    resp = client.get("/")
    assert resp.status_code == 200
    body = resp.json()
    assert "message" in body
    assert "/predict-price" in body["message"]


def test_health(monkeypatch):
    monkeypatch.setattr(system_router, "_models_health_check",
                        lambda: {"status": "ok"})

    async def _ok_platform():
        return {"status": "ok", "url": "http://localhost:8080/api/v1/produce", "status_code": 200}

    async def _ok_reviews():
        return {"status": "ok", "url": "http://localhost:8080", "status_code": 200}

    monkeypatch.setattr(
        system_router, "_platform_api_health_check", _ok_platform)
    monkeypatch.setattr(
        system_router, "_review_service_health_check", _ok_reviews)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["models"]["status"] == "ok"
    assert body["checks"]["platform_api"]["status"] == "ok"
    assert body["checks"]["review_service"]["status"] == "ok"


def test_health_returns_503_when_dependency_fails(monkeypatch):
    monkeypatch.setattr(system_router, "_models_health_check",
                        lambda: {"status": "ok"})

    async def _platform_error():
        return {"status": "error", "detail": "timeout"}

    async def _ok_reviews():
        return {"status": "ok", "url": "http://localhost:8080", "status_code": 200}

    monkeypatch.setattr(
        system_router, "_platform_api_health_check", _platform_error)
    monkeypatch.setattr(
        system_router, "_review_service_health_check", _ok_reviews)

    resp = client.get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "error"


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


def test_predict_price_normalizes_categorical_inputs(monkeypatch):
    captured = {}

    class CapturePricingModel:
        def predict(self, X):
            captured["frame"] = X.copy()
            return np.array([1.25] * len(X))

    monkeypatch.setattr(
        pricing_router, "dynamic_pricing_model", CapturePricingModel())
    monkeypatch.setattr(
        pricing_router,
        "model_columns",
        [
            "month",
            "year",
            "latitude",
            "longitude",
            "market_id",
            "commodity_id",
            "commodity_Maize",
            "market_Harare",
            "category_Cereals",
            "unit_KG",
            "currency_USD",
            "priceflag_actual",
            "pricetype_Retail",
            "admin1_Manicaland",
            "admin2_Mutare",
        ],
    )

    payload = {
        "commodity": "maize",
        "market": "harare",
        "category": "cereals",
        "unit": "kg",
        "month": 11,
        "latitude": -17.8,
        "longitude": 31.0,
        "currency": "usd",
        "priceflag": "actual",
        "admin1": "manicaland",
        "admin2": "mutare",
        "pricetype": "retail",
        "market_id": 1,
        "commodity_id": 51,
        "year": 2024,
    }

    resp = client.post("/predict-price", json=payload)
    assert resp.status_code == 200
    frame = captured["frame"]
    assert frame.iloc[0]["commodity_Maize"] == 1
    assert frame.iloc[0]["market_Harare"] == 1
    assert frame.iloc[0]["category_Cereals"] == 1
    assert frame.iloc[0]["unit_KG"] == 1
    assert frame.iloc[0]["currency_USD"] == 1
    assert frame.iloc[0]["priceflag_actual"] == 1
    assert frame.iloc[0]["pricetype_Retail"] == 1
    assert frame.iloc[0]["admin1_Manicaland"] == 1
    assert frame.iloc[0]["admin2_Mutare"] == 1


def test_predict_price_maps_request_values_to_model_vocabulary(monkeypatch):
    captured = {}

    class CapturePricingModel:
        def predict(self, X):
            captured["frame"] = X.copy()
            return np.array([2.5] * len(X))

    monkeypatch.setattr(
        pricing_router, "dynamic_pricing_model", CapturePricingModel())
    monkeypatch.setattr(
        pricing_router,
        "model_columns",
        [
            "month",
            "latitude",
            "longitude",
            "commodity_Maize",
            "market_Mbare",
            "category_cereals and tubers",
            "unit_KG",
            "currency_USD",
            "priceflag_actual",
            "admin1_Manicaland",
        ],
    )

    payload = {
        "commodity": "maize",
        "market": "mbare",
        "category": "cereals",
        "unit": "kg",
        "month": 11,
        "latitude": -17.8,
        "longitude": 31.0,
        "currency": "usd",
        "priceflag": "actual",
        "admin1": "manicaland",
    }

    resp = client.post("/predict-price", json=payload)
    assert resp.status_code == 200
    frame = captured["frame"]
    assert frame.iloc[0]["commodity_Maize"] == 1
    assert frame.iloc[0]["market_Mbare"] == 1
    assert frame.iloc[0]["category_cereals and tubers"] == 1
    assert frame.iloc[0]["unit_KG"] == 1
    assert frame.iloc[0]["currency_USD"] == 1
    assert frame.iloc[0]["priceflag_actual"] == 1
    assert frame.iloc[0]["admin1_Manicaland"] == 1


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


def test_forecast_endpoint_uses_real_model_artifacts(monkeypatch):
    forecast_model, commodity_cols = _load_real_forecast_assets()
    monkeypatch.setattr(forecasting_router, "forecast_model", forecast_model)
    monkeypatch.setattr(forecasting_router, "commodity_cols", commodity_cols)

    maize = client.get("/forecast/Maize?periods=2&visual=false").json()
    beans = client.get("/forecast/Beans?periods=2&visual=false").json()
    wheat = client.get("/forecast/Wheat?periods=2&visual=false").json()

    maize_values = [point["value"] for point in maize["forecast"]]
    beans_values = [point["value"] for point in beans["forecast"]]
    wheat_values = [point["value"] for point in wheat["forecast"]]

    assert len(maize_values) == 2
    assert len(beans_values) == 2
    assert len(wheat_values) == 2
    assert maize_values != beans_values or beans_values != wheat_values
    assert all(value >= 0 for value in maize_values +
               beans_values + wheat_values)


def test_forecast_endpoint_writes_results_for_multiple_commodities(monkeypatch):
    commodities = [
        "Beans",
        "Beans (sugar)",
        "Beans (sugar, biofortified)",
        "Cowpeas",
        "Groundnuts (shelled)",
        "Maize",
        "Maize meal",
        "Maize meal (white, fortified)",
        "Rice",
        "Wheat",
    ]
    forecast_model, commodity_cols = _load_real_forecast_assets()
    results = []

    monkeypatch.setattr(forecasting_router, "forecast_model", forecast_model)
    monkeypatch.setattr(forecasting_router, "commodity_cols", commodity_cols)

    for commodity in commodities:
        response = client.get(
            f"/forecast/{commodity}?periods=3&visual=false"
        )
        assert response.status_code == 200
        body = response.json()
        results.append(
            {
                "commodity": commodity,
                "status": response.status_code,
                "forecast": body["forecast"],
                "first_value": body["forecast"][0]["value"],
                "last_value": body["forecast"][-1]["value"],
            }
        )

    out_file = Path(__file__).resolve().with_name(
        "forecasting_endpoint_results.json")
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")

    assert out_file.exists()
    assert len(results) == 10
    assert all(len(item["forecast"]) == 3 for item in results)


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


def test_demand_supply_forecast_surfaces_stale_market_signal():
    payload = {
        "region": "Manicaland",
        "season": "rainy",
        "periods": 2,
        "historical_sales": [
            {"date": "2024-01-01", "commodity": "maize",
                "quantity": 120, "region": "Manicaland"},
            {"date": "2024-02-01", "commodity": "maize",
                "quantity": 135, "region": "Manicaland"},
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
    assert body["market_signals"][0]["commodity"] == "maize"
    assert body["market_signals"][0]["impact"] == 0.0
    assert body["market_signals"][0]["is_stale"] is True
    assert body["warnings"]
    assert body["warnings"][0].startswith("market_data_stale:")


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


def test_trust_score_surfaces_review_service_config_gap(monkeypatch):
    monkeypatch.setattr(trust_service.config, "SPRING_BOOT_BASE_URL", "")

    resp = client.get("/trust-score/user-123")
    assert resp.status_code == 200
    body = resp.json()
    assert body["user_id"] == "user-123"
    assert body["scale"] == 5
    assert body["source"] == "spring_boot_error"
    assert body["review_count"] == 0
    assert body["warnings"] == ["spring_boot_base_url_not_set"]


def test_auto_pricing_without_live_signals():
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
        "use_live_signals": False,
        "demand_signal": 80,
        "supply_volume": 40,
    }
    resp = client.post("/pricing/auto", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "success"
    assert body["suggested_price"] >= 0
    assert "adjustment_pct" in body


def test_auto_pricing_with_live_signals(monkeypatch):
    async def _fake_signals(commodity, window_days):
        return {
            "demand_count": 10.0,
            "demand_qty": 120.0,
            "supply_count": 5.0,
            "supply_qty": 60.0,
        }, []

    monkeypatch.setattr(
        pricing_service, "resolve_platform_signals", _fake_signals)
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
        "use_live_signals": True,
        "signal_window_days": 30,
        "max_adjustment": 0.2,
    }
    resp = client.post("/pricing/auto", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["signals"]["platform_signals"]["demand_qty"] == 120.0
    assert body["sources"] == ["platform_api"]


def test_logistics_match_with_payload():
    payload = {
        "logistics_request": {
            "origin_lat": -17.8,
            "origin_lon": 31.0,
            "destination_lat": -18.2,
            "destination_lon": 31.6,
            "quantity": 5,
        },
        "providers": [
            {"id": "prov-1", "latitude": -17.9, "longitude": 31.1,
                "cost_per_km": 0.8, "capacity": 8},
            {"id": "prov-2", "latitude": -19.2, "longitude": 32.1,
                "cost_per_km": 0.5, "capacity": 3},
        ],
        "top_n": 2,
    }
    resp = client.post("/logistics/match", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["matches"]
    assert len(body["matches"]) == 2


def test_integrations_weather(monkeypatch):
    async def _fake_weather(lat, lon, days=7):
        return (
            [{"date": "2026-03-01", "rainfall_mm": 10, "temperature_c": 24}],
            "open_meteo",
            None,
        )

    monkeypatch.setattr(integrations_service,
                        "fetch_weather_open_meteo", _fake_weather)
    resp = client.get(
        "/integrations/weather?latitude=-17.8&longitude=31.0&days=3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "open_meteo"
    assert len(body["weather"]) == 1


def test_integrations_market_prices(monkeypatch):
    async def _fake_platform(region, commodity=None):
        return ([{"date": "2026-03-01", "commodity": "maize", "price": 0.4, "market": "Harare"}], "platform", None)

    async def _fake_external(region, commodity=None):
        return ([], "external", None)

    monkeypatch.setattr(integrations_service,
                        "fetch_platform_market_prices", _fake_platform)
    monkeypatch.setattr(integrations_service,
                        "fetch_external_market_prices", _fake_external)
    resp = client.get(
        "/integrations/market-prices?region=Manicaland&commodity=maize")
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"] == ["platform"]
    assert len(body["market_prices"]) == 1
