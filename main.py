from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import httpx
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple, Iterable
from datetime import UTC, datetime, timedelta
import numpy as np
import math

app = FastAPI(title="eHarvest AI API",
              description="API for eHarvest AI services", version="1.0.0")

CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "false").strip().lower()


def _parse_cors_origins(raw_value: str) -> List[str]:
    if not raw_value:
        return ["*"]
    value = raw_value.strip()
    if value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


cors_origins = _parse_cors_origins(CORS_ALLOW_ORIGINS)
allow_credentials = CORS_ALLOW_CREDENTIALS in ("1", "true", "yes")

if cors_origins == ["*"]:
    allow_credentials = False

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

dynamic_pricing_model = joblib.load('ai_training/dynamic_pricing_model.pkl')
model_columns = joblib.load('ai_training/model_columns.pkl')
forecast_model = joblib.load('ai_training/demand_forecast_model.pkl')
commodity_cols = joblib.load('ai_training/forecast_features.pkl')

# ['commodity', 'market', 'category', 'unit', 'month', 'latitude', 'longitude', 'currency', 'priceflag']


class PricePredictionRequest(BaseModel):
    commodity: str
    market: str
    category: str
    unit: str
    month: int
    latitude: float
    longitude: float
    currency: str
    priceflag: str
    admin1: Optional[str] = None
    admin2: Optional[str] = None
    pricetype: Optional[str] = None
    market_id: Optional[int] = None
    commodity_id: Optional[int] = None
    year: Optional[int] = None


class BatchPricePredictionRequest(BaseModel):
    items: List[PricePredictionRequest]


class HistoricalSalesEntry(BaseModel):
    date: str
    commodity: str
    quantity: float
    region: Optional[str] = None


class HistoricalSupplyEntry(BaseModel):
    date: str
    commodity: str
    quantity: float
    region: Optional[str] = None


class WeatherEntry(BaseModel):
    date: str
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    region: Optional[str] = None


class MarketEntry(BaseModel):
    date: str
    commodity: str
    price: float
    market: Optional[str] = None
    region: Optional[str] = None


class ReviewEntry(BaseModel):
    rating: float
    comment: Optional[str] = None
    verified_purchase: Optional[bool] = None
    helpful_votes: Optional[int] = None
    reported: Optional[bool] = None
    review_date: Optional[str] = None


class DemandSupplyForecastRequest(BaseModel):
    region: str
    season: Optional[str] = None
    periods: int = 6
    historical_sales: List[HistoricalSalesEntry]
    historical_supply: Optional[List[HistoricalSupplyEntry]] = None
    weather: Optional[List[WeatherEntry]] = None
    market_data: Optional[List[MarketEntry]] = None
    supply_multiplier: float = 1.05
    commodities: Optional[List[str]] = None
    auto_fetch_external: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ClimateSnapshot(BaseModel):
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None


class DemandSignal(BaseModel):
    commodity: str
    expected_demand: float


class PrescriptiveRecommendationRequest(BaseModel):
    region: str
    season: Optional[str] = None
    month: Optional[int] = None
    budget_usd: float
    climate: ClimateSnapshot
    demand_forecast: Optional[List[DemandSignal]] = None
    market_data: Optional[List[MarketEntry]] = None
    top_n: int = 3
    auto_fetch_external: bool = False
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class AutoPricingRequest(PricePredictionRequest):
    use_live_signals: bool = True
    demand_signal: Optional[float] = None
    supply_volume: Optional[float] = None
    searches: Optional[int] = None
    carts: Optional[int] = None
    orders: Optional[int] = None
    active_listings: Optional[int] = None
    signal_window_days: int = 30
    max_adjustment: float = 0.3

    model_config = {
        "json_schema_extra": {
            "examples": [
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
                    "use_live_signals": True,
                    "signal_window_days": 30,
                    "max_adjustment": 0.3,
                }
            ]
        }
    }


class LogisticsMatchRequest(BaseModel):
    request_id: Optional[str] = None
    logistics_request: Optional[Dict[str, Any]] = None
    providers: Optional[List[Dict[str, Any]]] = None
    top_n: int = 3
    max_distance_km: Optional[float] = None
    weights: Optional[Dict[str, float]] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "request_id": "123",
                    "top_n": 3,
                    "weights": {"cost": 0.4, "distance": 0.4, "capacity": 0.2},
                },
                {
                    "logistics_request": {
                        "origin_lat": -17.8,
                        "origin_lon": 31.0,
                        "destination_lat": -18.2,
                        "destination_lon": 31.6,
                        "quantity": 5,
                    },
                    "providers": [
                        {
                            "id": "prov-1",
                            "latitude": -17.9,
                            "longitude": 31.1,
                            "cost_per_km": 0.8,
                            "capacity": 8,
                        }
                    ],
                    "top_n": 2,
                },
            ]
        }
    }


@app.post("/predict-price")
async def predict_price(request: PricePredictionRequest):
    # Convert request to a dict and then to a dataframe
    input_data = pd.DataFrame([request.model_dump()])

    # One-Hot Encode the categorical features
    input_encoded = pd.get_dummies(input_data)

    # Reindex the columns to match the model's expected input
    # this adds any missing columns with 0 values and ensures the order is correct
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction using the model
    prediction = dynamic_pricing_model.predict(final_input)

    return {
        "suggested_price": round(float(prediction[0]), 2),
        "currency": request.currency,
        "status": "success"
    }


@app.get("/pricing/schema")
def pricing_schema():
    return {
        "required_fields": [
            "commodity",
            "market",
            "category",
            "unit",
            "month",
            "latitude",
            "longitude",
            "currency",
            "priceflag"
        ],
        "optional_fields": [
            "admin1",
            "admin2",
            "pricetype",
            "market_id",
            "commodity_id",
            "year"
        ],
        "model_columns": model_columns,
        "notes": [
            "month is 1-12",
            "priceflag is a categorical flag from the source data (e.g., actual)",
            "categorical fields are one-hot encoded server-side"
        ]
    }


@app.post("/pricing/batch")
async def predict_price_batch(request: BatchPricePredictionRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")

    input_df = pd.DataFrame([item.model_dump() for item in request.items])
    input_encoded = pd.get_dummies(input_df)
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = dynamic_pricing_model.predict(final_input)

    results = []
    for item, pred in zip(request.items, predictions):
        results.append({
            "commodity": item.commodity,
            "market": item.market,
            "suggested_price": round(float(pred), 2),
            "currency": item.currency
        })

    return {
        "status": "success",
        "count": len(results),
        "predictions": results
    }


@app.post("/pricing/auto")
async def auto_pricing(request: AutoPricingRequest):
    input_data = pd.DataFrame([request.model_dump()])
    input_encoded = pd.get_dummies(input_data)
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)
    prediction = dynamic_pricing_model.predict(final_input)
    base_price = round(float(prediction[0]), 2)

    warnings: List[str] = []
    sources: List[str] = []

    demand_signal = request.demand_signal
    supply_volume = request.supply_volume

    platform_signals: Dict[str, float] = {}
    if request.use_live_signals:
        platform_signals, platform_warnings = await _resolve_platform_signals(
            request.commodity, max(1, request.signal_window_days)
        )
        if platform_warnings:
            warnings.extend(platform_warnings)
        sources.append("platform_api")

    if demand_signal is None:
        if platform_signals:
            demand_signal = platform_signals.get("demand_qty") or platform_signals.get("demand_count")
    if supply_volume is None:
        if platform_signals:
            supply_volume = platform_signals.get("supply_qty") or platform_signals.get("supply_count")

    if request.searches:
        demand_signal = (demand_signal or 0.0) + float(request.searches)
    if request.carts:
        demand_signal = (demand_signal or 0.0) + float(request.carts)
    if request.orders:
        demand_signal = (demand_signal or 0.0) + float(request.orders)

    if request.active_listings:
        supply_volume = (supply_volume or 0.0) + float(request.active_listings)

    pressure = _compute_price_pressure(demand_signal, supply_volume)
    max_adjustment = _clamp(request.max_adjustment, 0.0, 1.0)
    adjustment = pressure * max_adjustment
    adjusted_price = round(base_price * (1 + adjustment), 2)

    return {
        "status": "success",
        "base_price": base_price,
        "suggested_price": adjusted_price,
        "currency": request.currency,
        "adjustment_pct": round(adjustment * 100, 2),
        "signals": {
            "demand_signal": demand_signal,
            "supply_volume": supply_volume,
            "platform_signals": platform_signals,
        },
        "sources": sources,
        "warnings": warnings,
    }


@app.get("/forecast/{commodity}")
async def get_forecast(
    commodity: str,
    periods: int = 30,
    region: Optional[str] = None,
    visual: bool = True
):
    # 1. Create future dates
    future = forecast_model.make_future_dataframe(periods=periods)

    # 2. Set the "Switch" for the requested commodity
    target_col = f"commodity_{commodity}"
    region_col = f"admin1_{region}" if region else None

    for col in commodity_cols:
        if col == target_col:
            future[col] = 1
        elif region_col and col == region_col:
            future[col] = 1
        else:
            future[col] = 0

    # 3. Predict
    forecast = forecast_model.predict(future)

    # Return last 'n' days
    result = forecast[['ds', 'yhat']].tail(periods)
    records = [
        {"date": row["ds"].strftime("%Y-%m-%d"), "value": float(row["yhat"])}
        for _, row in result.iterrows()
    ]
    response = {"commodity": commodity, "region": region, "forecast": records}

    if visual:
        response["visual"] = {
            "type": "line",
            "x": [r["date"] for r in records],
            "series": [{"name": "Demand Forecast", "data": [r["value"] for r in records]}]
        }

    return response


ZWE_SEASONS = {
    "rainy": [11, 12, 1, 2, 3],
    "post_harvest": [4, 5],
    "cool_dry": [6, 7, 8],
    "hot_dry": [9, 10]
}


CROP_PROFILES = {
    "maize": {"rain_min": 500, "rain_max": 1200, "temp_min": 18, "temp_max": 30, "plant_months": [10, 11, 12]},
    "sorghum": {"rain_min": 300, "rain_max": 800, "temp_min": 20, "temp_max": 35, "plant_months": [11, 12, 1]},
    "millet": {"rain_min": 250, "rain_max": 700, "temp_min": 20, "temp_max": 35, "plant_months": [11, 12, 1]},
    "groundnuts": {"rain_min": 400, "rain_max": 900, "temp_min": 18, "temp_max": 30, "plant_months": [11, 12]},
    "beans": {"rain_min": 350, "rain_max": 800, "temp_min": 16, "temp_max": 28, "plant_months": [11, 12]},
    "soybeans": {"rain_min": 450, "rain_max": 1000, "temp_min": 18, "temp_max": 32, "plant_months": [11, 12]},
    "sunflower": {"rain_min": 350, "rain_max": 900, "temp_min": 18, "temp_max": 32, "plant_months": [11, 12, 1]},
    "tomatoes": {"rain_min": 400, "rain_max": 800, "temp_min": 18, "temp_max": 30, "plant_months": [8, 9, 10]},
    "onions": {"rain_min": 300, "rain_max": 700, "temp_min": 13, "temp_max": 28, "plant_months": [6, 7, 8]},
    "potatoes": {"rain_min": 400, "rain_max": 1000, "temp_min": 12, "temp_max": 25, "plant_months": [6, 7, 8]}
}


CROP_COST_USD_PER_HA = {
    "maize": 380,
    "sorghum": 260,
    "millet": 240,
    "groundnuts": 420,
    "beans": 300,
    "soybeans": 450,
    "sunflower": 310,
    "tomatoes": 1500,
    "onions": 1200,
    "potatoes": 900
}

SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL", "").rstrip("/")
SPRING_BOOT_REVIEWS_PATH = os.getenv(
    "SPRING_BOOT_REVIEWS_PATH", "/api/reviews/user/{user_id}")
USE_REVIEW_PLACEHOLDER = os.getenv(
    "USE_REVIEW_PLACEHOLDER", "true").lower() in ("1", "true", "yes")
PLATFORM_API_BASE_URL = os.getenv(
    "PLATFORM_API_BASE_URL", "http://localhost:8080").rstrip("/")
PLATFORM_API_KEY = os.getenv(
    "PLATFORM_API_KEY", "eharvest-ai-secret-key-12345")
PLATFORM_API_TIMEOUT = float(os.getenv("PLATFORM_API_TIMEOUT", "6"))
WEATHER_API_URL = os.getenv(
    "WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast").strip()
MARKET_DATA_API_URL = os.getenv("MARKET_DATA_API_URL", "").strip()


def _season_from_month(month: int) -> str:
    for season, months in ZWE_SEASONS.items():
        if month in months:
            return season
    return "unknown"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    return float(series.mean())


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    if isinstance(value, (int, float)):
        return value != 0
    return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _extract_list_payload(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("content", "items", "data", "results", "records", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _get_first_present(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def _get_nested_dict(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def _parse_date(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if isinstance(parsed, pd.Timestamp):
        return parsed.to_pydatetime()
    return None


def _within_days(date_value: Optional[datetime], days: int) -> bool:
    if date_value is None:
        return True
    try:
        cutoff = datetime.now(UTC) - timedelta(days=days)
    except Exception:
        return True
    if date_value.tzinfo is None:
        date_value = date_value.replace(tzinfo=UTC)
    return date_value >= cutoff


def _extract_commodity(item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    direct = _get_first_present(item, (
        "commodity",
        "produce",
        "product",
        "item",
        "name",
        "produceName",
        "productName",
        "itemName",
    ))
    if isinstance(direct, str):
        return direct
    if isinstance(direct, dict):
        nested_name = _get_first_present(direct, (
            "commodity",
            "name",
            "produceName",
            "productName",
            "itemName",
        ))
        if isinstance(nested_name, str):
            return nested_name
    nested = _get_nested_dict(item, ("produce", "product", "item"))
    if isinstance(nested, dict):
        nested_name = _get_first_present(nested, (
            "commodity",
            "name",
            "produceName",
            "productName",
            "itemName",
        ))
        if isinstance(nested_name, str):
            return nested_name
    return None


def _extract_quantity(item: Dict[str, Any]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    value = _get_first_present(item, (
        "quantity",
        "qty",
        "amount",
        "volume",
        "weight",
        "units",
    ))
    return _coerce_float(value)


def _extract_cost(item: Dict[str, Any]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    value = _get_first_present(item, (
        "cost",
        "price",
        "rate",
        "price_per_km",
        "cost_per_km",
        "costPerKm",
        "ratePerKm",
        "basePrice",
        "base_fee",
    ))
    return _coerce_float(value)


def _extract_date(item: Dict[str, Any]) -> Optional[datetime]:
    if not isinstance(item, dict):
        return None
    value = _get_first_present(item, (
        "created_at",
        "createdAt",
        "date",
        "orderDate",
        "requestedAt",
        "updated_at",
        "updatedAt",
    ))
    return _parse_date(value)


def _extract_lat_lon(payload: Dict[str, Any], lat_keys: Iterable[str], lon_keys: Iterable[str]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(payload, dict):
        return None, None
    for lat_key in lat_keys:
        if lat_key in payload:
            lat_val = _coerce_float(payload.get(lat_key))
            if lat_val is None:
                continue
            for lon_key in lon_keys:
                if lon_key in payload:
                    lon_val = _coerce_float(payload.get(lon_key))
                    if lon_val is not None:
                        return lat_val, lon_val
    return None, None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return radius * c


def _normalize_inverse(value: Optional[float], min_value: float, max_value: float) -> Optional[float]:
    if value is None:
        return None
    if max_value <= min_value:
        return 1.0
    scaled = (value - min_value) / (max_value - min_value)
    return _clamp(1.0 - scaled, 0.0, 1.0)


def _extract_location(payload: Dict[str, Any], lat_keys: Iterable[str], lon_keys: Iterable[str], nested_keys: Iterable[str]) -> Tuple[Optional[float], Optional[float]]:
    lat, lon = _extract_lat_lon(payload, lat_keys, lon_keys)
    if lat is not None and lon is not None:
        return lat, lon
    nested = _get_nested_dict(payload, nested_keys)
    if isinstance(nested, dict):
        lat, lon = _extract_lat_lon(nested, lat_keys, lon_keys)
        if lat is not None and lon is not None:
            return lat, lon
    return None, None


def _extract_origin_destination(payload: Dict[str, Any]) -> Tuple[Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]:
    origin_lat_keys = (
        "origin_lat", "pickup_lat", "from_lat", "source_lat", "start_lat",
        "originLatitude", "pickupLatitude",
    )
    origin_lon_keys = (
        "origin_lon", "origin_lng", "pickup_lon", "pickup_lng", "from_lon", "from_lng",
        "source_lon", "source_lng", "start_lon", "start_lng",
        "originLongitude", "pickupLongitude",
    )
    dest_lat_keys = (
        "destination_lat", "dropoff_lat", "to_lat", "end_lat", "delivery_lat",
        "destinationLatitude", "dropoffLatitude",
    )
    dest_lon_keys = (
        "destination_lon", "destination_lng", "dropoff_lon", "dropoff_lng", "to_lon", "to_lng",
        "end_lon", "end_lng", "delivery_lon", "delivery_lng",
        "destinationLongitude", "dropoffLongitude",
    )
    origin = _extract_location(payload, origin_lat_keys, origin_lon_keys, ("origin", "pickup", "from", "source", "start", "location"))
    destination = _extract_location(payload, dest_lat_keys, dest_lon_keys, ("destination", "dropoff", "to", "end"))
    return origin, destination


def _extract_provider_location(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat_keys = ("latitude", "lat", "current_lat", "currentLat", "base_lat", "baseLat")
    lon_keys = ("longitude", "lon", "lng", "current_lon", "current_lng", "currentLng", "base_lon", "base_lng", "baseLng")
    return _extract_location(payload, lat_keys, lon_keys, ("location", "currentLocation", "baseLocation"))


def _extract_capacity(payload: Dict[str, Any]) -> Optional[float]:
    return _coerce_float(_get_first_present(payload, ("capacity", "maxLoad", "max_capacity", "maxCapacity")))


def _extract_rate_per_km(payload: Dict[str, Any]) -> Optional[float]:
    return _coerce_float(_get_first_present(payload, ("rate_per_km", "price_per_km", "cost_per_km", "ratePerKm", "costPerKm")))


def _availability_flag(payload: Dict[str, Any]) -> Optional[bool]:
    flag = _coerce_bool(_get_first_present(payload, ("available", "isAvailable", "active", "enabled")))
    if flag is not None:
        return flag
    status = _get_first_present(payload, ("status", "state"))
    if isinstance(status, str):
        return status.strip().lower() in ("available", "active", "online", "ready")
    return None


def _compute_price_pressure(demand: Optional[float], supply: Optional[float]) -> float:
    demand_value = demand or 0.0
    supply_value = supply or 0.0
    denom = abs(demand_value) + abs(supply_value) + 1.0
    pressure = (demand_value - supply_value) / denom
    return _clamp(pressure, -1.0, 1.0)


async def _resolve_platform_signals(commodity: str, window_days: int) -> Tuple[Dict[str, float], List[str]]:
    warnings: List[str] = []
    order_items, warn_items = await _fetch_platform_list("/api/v1/order_items")
    if warn_items:
        warnings.append(warn_items)
    orders, warn_orders = await _fetch_platform_list("/api/v1/orders")
    if warn_orders:
        warnings.append(warn_orders)
    produce, warn_produce = await _fetch_platform_list("/api/v1/produce")
    if warn_produce:
        warnings.append(warn_produce)

    demand_count = 0
    demand_qty = 0.0
    for item in order_items:
        if commodity and not _match_commodity(item, commodity):
            continue
        if not _within_days(_extract_date(item), window_days):
            continue
        qty = _extract_quantity(item)
        demand_qty += qty if qty is not None else 1.0
        demand_count += 1

    if demand_count == 0:
        for order in orders:
            if not _within_days(_extract_date(order), window_days):
                continue
            demand_count += 1

    supply_count = 0
    supply_qty = 0.0
    for item in produce:
        if commodity and not _match_commodity(item, commodity):
            continue
        status = _get_first_present(item, ("status", "state"))
        if isinstance(status, str) and status.strip().lower() in ("inactive", "archived", "sold"):
            continue
        qty = _extract_quantity(item)
        supply_qty += qty if qty is not None else 1.0
        supply_count += 1

    return {
        "demand_count": float(demand_count),
        "demand_qty": float(demand_qty),
        "supply_count": float(supply_count),
        "supply_qty": float(supply_qty),
    }, warnings


def _score_logistics_candidates(
    request_payload: Dict[str, Any],
    providers: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    max_distance_km: Optional[float] = None,
) -> Dict[str, Any]:
    weights = weights or {}
    weight_cost = float(weights.get("cost", 0.4))
    weight_distance = float(weights.get("distance", 0.4))
    weight_capacity = float(weights.get("capacity", 0.2))

    origin, destination = _extract_origin_destination(request_payload)
    route_distance = None
    if origin[0] is not None and destination[0] is not None:
        route_distance = _haversine_km(origin[0], origin[1], destination[0], destination[1])

    demand_qty = _extract_quantity(request_payload) or _coerce_float(
        _get_first_present(request_payload, ("weight", "load", "volume"))
    )

    scored: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    distances: List[float] = []
    costs: List[float] = []

    for provider in providers:
        provider_location = _extract_provider_location(provider)
        pickup_distance = None
        if origin[0] is not None and provider_location[0] is not None:
            pickup_distance = _haversine_km(origin[0], origin[1], provider_location[0], provider_location[1])

        if max_distance_km is not None and pickup_distance is not None:
            if pickup_distance > max_distance_km:
                rejected.append({
                    "provider": provider,
                    "reason": "pickup_distance_exceeds_max",
                    "pickup_distance_km": round(pickup_distance, 2),
                })
                continue

        base_cost = _extract_cost(provider)
        rate_per_km = _extract_rate_per_km(provider)
        distance_for_cost = route_distance or pickup_distance or 0.0
        estimated_cost = None
        if base_cost is not None or rate_per_km is not None:
            estimated_cost = (base_cost or 0.0) + (rate_per_km or 0.0) * distance_for_cost

        capacity = _extract_capacity(provider)
        capacity_score = None
        if capacity is not None and demand_qty is not None:
            capacity_score = 1.0 if capacity >= demand_qty else 0.4
        elif capacity is not None:
            capacity_score = 0.8

        availability = _availability_flag(provider)
        availability_penalty = 1.0
        if availability is False:
            availability_penalty = 0.6

        if pickup_distance is not None:
            distances.append(pickup_distance)
        if estimated_cost is not None:
            costs.append(estimated_cost)

        scored.append({
            "provider": provider,
            "pickup_distance_km": pickup_distance,
            "route_distance_km": route_distance,
            "estimated_cost": estimated_cost,
            "capacity_score": capacity_score,
            "availability_penalty": availability_penalty,
        })

    min_distance = min(distances) if distances else 0.0
    max_distance = max(distances) if distances else 0.0
    min_cost = min(costs) if costs else 0.0
    max_cost = max(costs) if costs else 0.0

    results: List[Dict[str, Any]] = []
    for item in scored:
        distance_score = _normalize_inverse(item["pickup_distance_km"], min_distance, max_distance)
        cost_score = _normalize_inverse(item["estimated_cost"], min_cost, max_cost)
        capacity_score = item["capacity_score"]

        total_weight = 0.0
        score = 0.0
        if distance_score is not None:
            score += weight_distance * distance_score
            total_weight += weight_distance
        if cost_score is not None:
            score += weight_cost * cost_score
            total_weight += weight_cost
        if capacity_score is not None:
            score += weight_capacity * capacity_score
            total_weight += weight_capacity
        if total_weight > 0:
            score = score / total_weight

        score *= item["availability_penalty"]

        results.append({
            "provider": item["provider"],
            "score": round(score, 4),
            "pickup_distance_km": None if item["pickup_distance_km"] is None else round(item["pickup_distance_km"], 2),
            "route_distance_km": None if item["route_distance_km"] is None else round(item["route_distance_km"], 2),
            "estimated_cost": None if item["estimated_cost"] is None else round(item["estimated_cost"], 2),
            "capacity_score": item["capacity_score"],
            "availability_penalty": item["availability_penalty"],
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "route_distance_km": None if route_distance is None else round(route_distance, 2),
        "matches": results,
        "rejected": rejected,
    }


_VADER_ANALYZER = None


def _get_vader_analyzer():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is False:
        return None
    if _VADER_ANALYZER is not None:
        return _VADER_ANALYZER

    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
    except Exception:
        _VADER_ANALYZER = False
        return None

    try:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
        return _VADER_ANALYZER
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except Exception:
            _VADER_ANALYZER = False

    return _VADER_ANALYZER


def _comment_sentiment_score(comment: Optional[str]) -> Optional[float]:
    if comment is None or not str(comment).strip():
        return None
    analyzer = _get_vader_analyzer()
    if analyzer is None:
        return None
    scores = analyzer.polarity_scores(str(comment))
    return float(scores.get("compound", 0.0))


def _coerce_review_payload(payload: Dict[str, Any]) -> Optional[ReviewEntry]:
    rating = payload.get("rating")
    if rating is None:
        rating = payload.get("score")
    verified = payload.get("verified_purchase")
    if verified is None:
        verified = payload.get("verifiedPurchase")

    helpful = payload.get("helpful_votes")
    if helpful is None:
        helpful = payload.get("helpfulVotes")

    reported = payload.get("reported")
    if reported is None:
        reported = payload.get("flagged")

    review_date = payload.get("review_date")
    if review_date is None:
        review_date = payload.get("createdAt")

    comment = payload.get("comment")
    if comment is None:
        comment = payload.get("review")
    if comment is None:
        comment = payload.get("reviewText")
    if comment is None:
        comment = payload.get("text")

    rating_value = None
    if rating is not None:
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            rating_value = None

    sentiment = _comment_sentiment_score(comment)
    text_rating = None
    if sentiment is not None:
        text_rating = _clamp(3.0 + (2.0 * sentiment), 1.0, 5.0)

    if rating_value is None and text_rating is None:
        return None

    if rating_value is None:
        rating_value = text_rating
    elif text_rating is not None:
        rating_value = (0.7 * rating_value) + (0.3 * text_rating)

    helpful_value = None
    if helpful is not None:
        try:
            helpful_value = int(helpful)
        except (TypeError, ValueError):
            helpful_value = None

    return ReviewEntry(
        rating=rating_value,
        comment=comment,
        verified_purchase=_coerce_bool(verified),
        helpful_votes=helpful_value,
        reported=_coerce_bool(reported),
        review_date=review_date,
    )


def _placeholder_reviews(user_id: str) -> List[ReviewEntry]:
    return [
        ReviewEntry(
            rating=4.6,
            comment="Quick delivery and the produce quality was excellent.",
            verified_purchase=True,
            helpful_votes=12,
            reported=False,
            review_date="2025-06-12",
        ),
        ReviewEntry(
            rating=4.2,
            comment="Good experience overall, but packaging could improve.",
            verified_purchase=True,
            helpful_votes=5,
            reported=False,
            review_date="2025-07-03",
        ),
        ReviewEntry(
            rating=3.8,
            comment="Decent service, average quality, nothing special.",
            verified_purchase=False,
            helpful_votes=2,
            reported=False,
            review_date="2025-09-01",
        ),
        ReviewEntry(
            rating=4.9,
            comment="Fantastic support and very fresh produce!",
            verified_purchase=True,
            helpful_votes=18,
            reported=False,
            review_date="2025-11-15",
        ),
        ReviewEntry(
            rating=2.6,
            comment="Late delivery and items were damaged.",
            verified_purchase=False,
            helpful_votes=1,
            reported=True,
            review_date="2026-01-10",
        ),
    ]


def _compute_trust_score(reviews: List[ReviewEntry]) -> Dict[str, Any]:
    if not reviews:
        return {
            "trust_score": 2.5,
            "review_count": 0,
            "average_rating": 0.0,
            "weighted_average": 0.0,
            "reported_ratio": 0.0,
            "verified_ratio": 0.0,
            "note": "no reviews available; returning neutral trust score",
        }

    ratings = []
    weights = []
    reported_count = 0
    verified_count = 0

    for review in reviews:
        rating = _clamp(float(review.rating), 1.0, 5.0)
        helpful_votes = review.helpful_votes or 0
        helpful_votes = max(0, helpful_votes)
        weight = 1.0 + (min(helpful_votes, 20) / 20.0)

        if review.verified_purchase:
            weight += 0.25
            verified_count += 1

        if review.reported:
            weight *= 0.6
            reported_count += 1

        ratings.append(rating)
        weights.append(weight)

    weighted_avg = float(np.average(ratings, weights=weights)) if weights else 0.0
    reported_ratio = reported_count / len(reviews)
    trust = weighted_avg - (reported_ratio * 0.8)
    trust = _clamp(trust, 1.0, 5.0)
    average_rating = float(np.mean(ratings)) if ratings else 0.0
    verified_ratio = verified_count / len(reviews)

    return {
        "trust_score": round(trust, 2),
        "review_count": len(reviews),
        "average_rating": round(average_rating, 2),
        "weighted_average": round(weighted_avg, 2),
        "reported_ratio": round(reported_ratio, 2),
        "verified_ratio": round(verified_ratio, 2),
    }


async def _fetch_user_reviews(user_id: str) -> Tuple[List[ReviewEntry], str, Optional[str]]:
    if USE_REVIEW_PLACEHOLDER or not SPRING_BOOT_BASE_URL:
        return _placeholder_reviews(user_id), "placeholder", None

    path = SPRING_BOOT_REVIEWS_PATH or "/api/reviews/user/{user_id}"
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{SPRING_BOOT_BASE_URL}{path.format(user_id=user_id)}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return _placeholder_reviews(user_id), "placeholder_fallback", f"spring_boot_error: {exc.__class__.__name__}"

    raw_reviews = data
    if isinstance(data, dict) and "reviews" in data:
        raw_reviews = data["reviews"]

    reviews: List[ReviewEntry] = []
    if isinstance(raw_reviews, list):
        for item in raw_reviews:
            if not isinstance(item, dict):
                continue
            review = _coerce_review_payload(item)
            if review is not None:
                reviews.append(review)

    return reviews, "spring_boot", None


def _platform_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if PLATFORM_API_KEY:
        headers["X-API-KEY"] = PLATFORM_API_KEY
    return headers


async def _platform_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[str]]:
    if not PLATFORM_API_BASE_URL:
        return None, "platform_base_url_not_set"
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{PLATFORM_API_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=PLATFORM_API_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=_platform_headers())
            resp.raise_for_status()
        return resp.json(), None
    except httpx.HTTPError as exc:
        return None, f"platform_api_error: {exc.__class__.__name__}"


async def _fetch_platform_list(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, warning = await _platform_get(path, params=params)
    if data is None:
        return [], warning
    return _extract_list_payload(data), warning


def _match_commodity(item: Dict[str, Any], commodity: str) -> bool:
    if not commodity:
        return False
    found = _extract_commodity(item)
    if not found:
        return False
    return commodity.strip().lower() in found.strip().lower()


async def _fetch_platform_market_prices(
    region: Optional[str],
    commodity: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    items, warning = await _fetch_platform_list("/api/v1/produce")
    results: List[Dict[str, Any]] = []
    for item in items:
        name = _extract_commodity(item)
        if not name:
            continue
        if commodity and commodity.lower() not in name.lower():
            continue
        price = _extract_cost(item)
        if price is None:
            continue
        market = _get_first_present(item, ("market", "location", "city", "town"))
        region_value = _get_first_present(item, ("region", "province", "admin1"))
        if region and isinstance(region_value, str):
            if region.lower() != region_value.lower():
                continue
        date_value = _extract_date(item) or datetime.now(UTC)
        results.append({
            "date": date_value.strftime("%Y-%m-%d"),
            "commodity": name,
            "price": float(price),
            "market": market,
            "region": region_value,
        })
    return results, "platform_produce", warning


async def _fetch_external_market_prices(
    region: Optional[str],
    commodity: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    if not MARKET_DATA_API_URL:
        return [], "not_configured", "market_data_api_url_not_set"
    try:
        async with httpx.AsyncClient(timeout=PLATFORM_API_TIMEOUT) as client:
            resp = await client.get(MARKET_DATA_API_URL, params={"region": region, "commodity": commodity})
            resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return [], "external_market_api", f"external_market_error: {exc.__class__.__name__}"

    items = _extract_list_payload(data)
    results: List[Dict[str, Any]] = []
    for item in items:
        name = _extract_commodity(item) or commodity
        price = _extract_cost(item)
        if name is None or price is None:
            continue
        date_value = _extract_date(item) or datetime.now(UTC)
        results.append({
            "date": date_value.strftime("%Y-%m-%d"),
            "commodity": name,
            "price": float(price),
            "market": _get_first_present(item, ("market", "location", "city")),
            "region": _get_first_present(item, ("region", "province", "admin1")),
        })
    return results, "external_market_api", None


async def _fetch_weather_open_meteo(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    if not WEATHER_API_URL:
        return [], "not_configured", "weather_api_url_not_set"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_mean,precipitation_sum",
        "forecast_days": max(1, min(days, 14)),
        "timezone": "UTC",
    }
    try:
        async with httpx.AsyncClient(timeout=PLATFORM_API_TIMEOUT) as client:
            resp = await client.get(WEATHER_API_URL, params=params)
            resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return [], "open_meteo", f"weather_api_error: {exc.__class__.__name__}"

    daily = data.get("daily", {}) if isinstance(data, dict) else {}
    dates = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []
    rains = daily.get("precipitation_sum") or []
    results: List[Dict[str, Any]] = []
    for idx, date_str in enumerate(dates):
        results.append({
            "date": date_str,
            "rainfall_mm": rains[idx] if idx < len(rains) else None,
            "temperature_c": temps[idx] if idx < len(temps) else None,
        })
    return results, "open_meteo", None


def _apply_region_filter(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if df is None or df.empty or "region" not in df.columns:
        return df
    filtered = df[df["region"].fillna("").str.lower() == region.lower()].copy()
    return filtered if not filtered.empty else df.copy()


def _compute_weather_impact(weather_df: Optional[pd.DataFrame], season: Optional[str]) -> float:
    if weather_df is None or weather_df.empty:
        return 0.0
    if "date" in weather_df.columns:
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        weather_df["season"] = weather_df["date"].dt.month.apply(
            _season_from_month)

    seasonal = weather_df
    if season:
        seasonal = weather_df[weather_df["season"] == season]
        if seasonal.empty:
            seasonal = weather_df

    rain_avg = _safe_mean(seasonal.get("rainfall_mm"))
    rain_all = _safe_mean(weather_df.get("rainfall_mm"))
    temp_avg = _safe_mean(seasonal.get("temperature_c"))
    temp_all = _safe_mean(weather_df.get("temperature_c"))

    rain_factor = 0.0 if rain_all == 0 else (rain_avg - rain_all) / rain_all
    temp_factor = 0.0 if temp_all == 0 else (temp_avg - temp_all) / temp_all

    return _clamp(0.4 * rain_factor + 0.2 * temp_factor, -0.3, 0.3)


def _compute_market_impact(market_df: Optional[pd.DataFrame], commodity: str) -> float:
    if market_df is None or market_df.empty:
        return 0.0
    commodity_df = market_df[market_df["commodity"].str.lower(
    ) == commodity.lower()].copy()
    if commodity_df.empty:
        return 0.0
    commodity_df["date"] = pd.to_datetime(commodity_df["date"])
    overall = _safe_mean(commodity_df["price"])
    recent_cutoff = commodity_df["date"].max() - timedelta(days=90)
    recent = commodity_df[commodity_df["date"] >= recent_cutoff]
    recent_avg = _safe_mean(recent["price"])
    if overall == 0:
        return 0.0
    price_trend = (recent_avg - overall) / overall
    return _clamp(-0.2 * price_trend, -0.2, 0.2)


def _forecast_monthly_series(
    monthly_df: pd.DataFrame,
    periods: int,
    weather_impact: float,
    market_impact: float
) -> List[Dict[str, Any]]:
    if monthly_df.empty:
        return []

    monthly_df = monthly_df.sort_values("year_month")
    overall_avg = _safe_mean(monthly_df["quantity"])
    monthly_df["month"] = monthly_df["year_month"].dt.month
    seasonal_means = monthly_df.groupby("month")["quantity"].mean()

    if len(monthly_df) >= 2:
        x = np.arange(len(monthly_df))
        y = monthly_df["quantity"].values
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    last_month = monthly_df["year_month"].max()
    results = []
    for i in range(1, periods + 1):
        future_month = last_month + pd.DateOffset(months=i)
        seasonal_base = seasonal_means.get(future_month.month, overall_avg)
        baseline = seasonal_base + slope * i
        adjusted = baseline * (1 + weather_impact + market_impact)
        results.append({
            "date": future_month.strftime("%Y-%m-%d"),
            "value": max(0.0, float(adjusted)),
            "season": _season_from_month(future_month.month)
        })
    return results


@app.post("/forecast/demand-supply")
async def demand_supply_forecast(request: DemandSupplyForecastRequest):
    if request.periods <= 0:
        raise HTTPException(
            status_code=400, detail="periods must be greater than 0")

    warnings: List[str] = []
    sources: List[str] = []

    sales_df = pd.DataFrame([item.model_dump()
                            for item in request.historical_sales])
    if sales_df.empty:
        raise HTTPException(
            status_code=400, detail="historical_sales cannot be empty")
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    sales_df = _apply_region_filter(sales_df, request.region)

    if request.historical_supply:
        supply_df = pd.DataFrame([item.model_dump()
                                 for item in request.historical_supply])
        supply_df["date"] = pd.to_datetime(supply_df["date"])
        supply_df = _apply_region_filter(supply_df, request.region)
    else:
        supply_df = pd.DataFrame()

    weather_df = None
    if request.weather:
        weather_df = pd.DataFrame([item.model_dump()
                                  for item in request.weather])
        weather_df = _apply_region_filter(weather_df, request.region)

    market_df = None
    if request.market_data:
        market_df = pd.DataFrame([item.model_dump()
                                 for item in request.market_data])
        market_df = _apply_region_filter(market_df, request.region)

    if request.auto_fetch_external:
        if (weather_df is None or weather_df.empty) and request.latitude is not None and request.longitude is not None:
            weather_payload, weather_source, warning = await _fetch_weather_open_meteo(
                request.latitude, request.longitude, days=max(request.periods, 7)
            )
            if weather_payload:
                weather_df = pd.DataFrame(weather_payload)
                sources.append(weather_source)
            if warning:
                warnings.append(warning)

        if market_df is None or market_df.empty:
            market_payload, market_source, warning = await _fetch_platform_market_prices(
                request.region
            )
            if market_payload:
                market_df = pd.DataFrame(market_payload)
                sources.append(market_source)
            if warning:
                warnings.append(warning)

            if market_df is None or market_df.empty:
                external_payload, external_source, warning = await _fetch_external_market_prices(
                    request.region
                )
                if external_payload:
                    market_df = pd.DataFrame(external_payload)
                    sources.append(external_source)
                if warning:
                    warnings.append(warning)

    commodities = request.commodities or sorted(
        sales_df["commodity"].dropna().unique().tolist())
    if not commodities:
        raise HTTPException(
            status_code=400, detail="no commodities found in historical_sales")

    weather_impact = _compute_weather_impact(weather_df, request.season)
    forecasts = []
    visual_series = []

    for commodity in commodities:
        commodity_sales = sales_df[sales_df["commodity"].str.lower(
        ) == commodity.lower()].copy()
        if commodity_sales.empty:
            continue
        commodity_sales["year_month"] = commodity_sales["date"].dt.to_period(
            "M").dt.to_timestamp()
        monthly_sales = commodity_sales.groupby(
            "year_month")["quantity"].sum().reset_index()
        demand_market_impact = _compute_market_impact(market_df, commodity)
        demand_forecast = _forecast_monthly_series(
            monthly_sales,
            request.periods,
            weather_impact,
            demand_market_impact
        )

        if not supply_df.empty:
            commodity_supply = supply_df[supply_df["commodity"].str.lower(
            ) == commodity.lower()].copy()
            commodity_supply["year_month"] = commodity_supply["date"].dt.to_period(
                "M").dt.to_timestamp()
            monthly_supply = commodity_supply.groupby(
                "year_month")["quantity"].sum().reset_index()
            supply_forecast = _forecast_monthly_series(
                monthly_supply,
                request.periods,
                weather_impact,
                0.0
            )
        else:
            supply_forecast = [
                {
                    "date": point["date"],
                    "value": max(0.0, point["value"] * request.supply_multiplier),
                    "season": point["season"]
                }
                for point in demand_forecast
            ]

        forecasts.append({
            "commodity": commodity,
            "demand": demand_forecast,
            "supply": supply_forecast
        })

        visual_series.append({
            "name": f"{commodity} demand",
            "data": [point["value"] for point in demand_forecast],
            "x": [point["date"] for point in demand_forecast]
        })

    if not forecasts:
        raise HTTPException(
            status_code=404, detail="no forecasts generated for provided commodities")

    return {
        "region": request.region,
        "season": request.season,
        "periods": request.periods,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecasts": forecasts,
        "visual": {
            "type": "line",
            "series": visual_series,
            "notes": "visual data only; frontend renders charts"
        },
        "sources": sources,
        "warnings": warnings,
        "assumptions": [
            "seasonal averages and recent trend are used for forecasting",
            "weather and market impacts are applied as bounded adjustments",
            "supply forecast uses provided supply data or demand * supply_multiplier"
        ]
    }


@app.post("/recommendations/prescriptive")
async def prescriptive_recommendations(request: PrescriptiveRecommendationRequest):
    if request.top_n <= 0:
        raise HTTPException(
            status_code=400, detail="top_n must be greater than 0")

    warnings: List[str] = []
    sources: List[str] = []

    target_month = request.month or datetime.now(UTC).month
    season = request.season or _season_from_month(target_month)

    demand_map = {item.commodity.lower(): item.expected_demand for item in (
        request.demand_forecast or [])}

    climate_rain = request.climate.rainfall_mm
    climate_temp = request.climate.temperature_c

    market_df = None
    if request.market_data:
        market_df = pd.DataFrame([item.model_dump()
                                 for item in request.market_data])
        market_df = _apply_region_filter(market_df, request.region)

    if request.auto_fetch_external:
        if (climate_rain is None or climate_temp is None) and request.latitude is not None and request.longitude is not None:
            weather_payload, weather_source, warning = await _fetch_weather_open_meteo(
                request.latitude, request.longitude, days=14
            )
            if weather_payload:
                weather_df = pd.DataFrame(weather_payload)
                if climate_rain is None:
                    climate_rain = _safe_mean(weather_df.get("rainfall_mm"))
                if climate_temp is None:
                    climate_temp = _safe_mean(weather_df.get("temperature_c"))
                sources.append(weather_source)
            if warning:
                warnings.append(warning)

        if market_df is None or market_df.empty:
            market_payload, market_source, warning = await _fetch_platform_market_prices(
                request.region
            )
            if market_payload:
                market_df = pd.DataFrame(market_payload)
                sources.append(market_source)
            if warning:
                warnings.append(warning)

            if market_df is None or market_df.empty:
                external_payload, external_source, warning = await _fetch_external_market_prices(
                    request.region
                )
                if external_payload:
                    market_df = pd.DataFrame(external_payload)
                    sources.append(external_source)
                if warning:
                    warnings.append(warning)

    scored = []
    for crop, profile in CROP_PROFILES.items():
        plant_ok = target_month in profile["plant_months"]

        rain_ok = True if climate_rain is None else (
            profile["rain_min"] <= climate_rain <= profile["rain_max"])
        temp_ok = True if climate_temp is None else (
            profile["temp_min"] <= climate_temp <= profile["temp_max"])
        climate_score = 1.0 if (rain_ok and temp_ok) else 0.0

        demand_score = 0.5
        if crop in demand_map:
            demand_score = min(
                1.0, max(0.0, demand_map[crop] / max(demand_map.values() or [1])))

        cost = CROP_COST_USD_PER_HA.get(crop, 400)
        budget_score = 1.0 if request.budget_usd >= cost else 0.4

        total = (0.45 * climate_score) + \
            (0.35 * demand_score) + (0.20 * budget_score)
        if plant_ok:
            total += 0.1

        scored.append({
            "commodity": crop,
            "score": round(total, 3),
            "estimated_cost_usd_per_ha": cost,
            "planting_months": profile["plant_months"],
            "climate_fit": climate_score == 1.0
        })

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)[
        : request.top_n]

    recommendations = []
    for item in scored:
        markets = []
        if market_df is not None and not market_df.empty:
            crop_prices = market_df[market_df["commodity"].str.lower(
            ) == item["commodity"]]
            if not crop_prices.empty and "market" in crop_prices.columns:
                best = crop_prices.groupby("market")["price"].mean(
                ).sort_values(ascending=False).head(3)
                markets = [{"market": idx, "avg_price": round(
                    float(val), 2)} for idx, val in best.items()]

        recommendations.append({
            "commodity": item["commodity"],
            "score": item["score"],
            "why": {
                "climate_fit": item["climate_fit"],
                "demand_signal": demand_map.get(item["commodity"], None),
                "estimated_cost_usd_per_ha": item["estimated_cost_usd_per_ha"],
                "planting_months": item["planting_months"]
            },
            "market_targets": markets
        })

    return {
        "region": request.region,
        "season": season,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recommendations": recommendations,
        "sources": sources,
        "warnings": warnings,
        "assumptions": [
            "crop profiles reflect typical Zimbabwe growing ranges",
            "budget score uses a rough cost-per-hectare estimate",
            "market targets use average prices in provided market_data"
        ]
    }


@app.post("/logistics/match")
async def logistics_match(request: LogisticsMatchRequest):
    warnings: List[str] = []
    sources: List[str] = []

    logistics_request = request.logistics_request
    if logistics_request is None and request.request_id:
        data, warning = await _platform_get(f"/api/v1/logistics/{request.request_id}")
        if data is not None:
            logistics_request = data if isinstance(data, dict) else None
            sources.append("platform_logistics")
        if warning:
            warnings.append(warning)

    providers = request.providers
    if not providers:
        providers, warning = await _fetch_platform_list("/api/v1/logistics-providers")
        if warning:
            warnings.append(warning)
        if providers:
            sources.append("platform_logistics_providers")

    if not logistics_request:
        raise HTTPException(status_code=400, detail="logistics_request or request_id is required")
    if not providers:
        raise HTTPException(status_code=400, detail="no logistics providers available for matching")

    result = _score_logistics_candidates(
        logistics_request,
        providers,
        weights=request.weights,
        max_distance_km=request.max_distance_km,
    )
    result["matches"] = result["matches"][: max(1, request.top_n)]

    return {
        "request": logistics_request,
        "route_distance_km": result["route_distance_km"],
        "matches": result["matches"],
        "rejected": result["rejected"],
        "sources": sources,
        "warnings": warnings,
    }


@app.get("/integrations/weather")
async def integrations_weather(
    latitude: float = Query(
        ...,
        examples={"default": {"value": -17.8}},
        description="Latitude in decimal degrees",
    ),
    longitude: float = Query(
        ...,
        examples={"default": {"value": 31.0}},
        description="Longitude in decimal degrees",
    ),
    days: int = Query(
        7,
        examples={"default": {"value": 7}},
        ge=1,
        le=14,
        description="Number of forecast days (1-14)",
    ),
):
    weather_payload, source, warning = await _fetch_weather_open_meteo(latitude, longitude, days=days)
    response = {
        "source": source,
        "weather": weather_payload,
    }
    if warning:
        response["warnings"] = [warning]
    return response


@app.get("/integrations/market-prices")
async def integrations_market_prices(
    region: Optional[str] = Query(None, examples={"default": {"value": "Manicaland"}}),
    commodity: Optional[str] = Query(None, examples={"default": {"value": "maize"}}),
):
    warnings: List[str] = []
    sources: List[str] = []

    market_payload, market_source, warning = await _fetch_platform_market_prices(region, commodity)
    if market_payload:
        sources.append(market_source)
    if warning:
        warnings.append(warning)

    if not market_payload:
        external_payload, external_source, warning = await _fetch_external_market_prices(region, commodity)
        if external_payload:
            market_payload = external_payload
            sources.append(external_source)
        if warning:
            warnings.append(warning)

    response = {
        "sources": sources,
        "market_prices": market_payload,
    }
    if warnings:
        response["warnings"] = warnings
    return response


@app.get("/trust-score/{user_id}")
async def trust_score(user_id: str):
    reviews, source, warning = await _fetch_user_reviews(user_id)
    details = _compute_trust_score(reviews)

    response = {
        "user_id": user_id,
        "trust_score": details["trust_score"],
        "scale": 5,
        "review_count": details["review_count"],
        "source": source,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "details": {
            "average_rating": details["average_rating"],
            "weighted_average": details["weighted_average"],
            "reported_ratio": details["reported_ratio"],
            "verified_ratio": details["verified_ratio"],
        },
    }

    if "note" in details:
        response["details"]["note"] = details["note"]

    if warning:
        response["warnings"] = [warning]

    return response


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def read_root():
    return {"message": "API is running. Use /predict-price for price predictions and /forecast/commodity for demand forecasts."}


'''
Datasets Links:

-https://data.humdata.org/dataset/global-wfp-food-prices
-https://bulks-faostat.fao.org/production/production_crops_E_All_Data_(Normalized).zip
-https://fenixservices.fao.org/faostat/static/bulkdownloads/FoodBalanceSheets_E_All_Data_(Normalized).zip
-https://data.chc.ucsb.edu/products/CHIRPS/v3.0/
-https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
-https://zimstat.co.zw/agriculture-statistics/
'''
