from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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
    region: Optional[str] = None
    periods: int = 6
    commodities: Optional[List[str]] = None


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
    climate: Optional[ClimateSnapshot] = None
    demand_forecast: Optional[List[DemandSignal]] = None
    market_data: Optional[List[MarketEntry]] = None
    top_n: int = 3
    auto_fetch_external: bool = True
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "region": "harare",
                    "month": 11,
                    "budget_usd": 500,
                    "top_n": 3
                },
                {
                    "region": "bulawayo",
                    "budget_usd": 350,
                    "season": "rainy"
                }
            ]
        }
    }


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
