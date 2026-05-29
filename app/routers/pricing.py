from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.models import AutoPricingRequest, BatchPricePredictionRequest, PricePredictionRequest
from app.services import pricing as pricing_service
from app.services.common import clamp
from app.state import dynamic_pricing_model, model_columns

router = APIRouter()


def _predict_price_value(data: Dict[str, Any]) -> float:
    return pricing_service.predict_price_value(
        data,
        dynamic_pricing_model,
        model_columns,
    )


@router.post("/predict-price")
async def predict_price(request: PricePredictionRequest):
    return {
        "suggested_price": _predict_price_value(request.model_dump()),
        "currency": request.currency,
        "status": "success",
    }


@router.get("/pricing/schema")
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
            "priceflag",
        ],
        "optional_fields": [
            "admin1",
            "admin2",
            "pricetype",
            "market_id",
            "commodity_id",
            "year",
        ],
        "model_columns": model_columns,
        "notes": [
            "month is 1-12",
            "priceflag is a categorical flag from the source data (e.g., actual)",
            "categorical fields are one-hot encoded server-side",
        ],
    }


@router.post("/pricing/batch")
async def predict_price_batch(request: BatchPricePredictionRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")

    results = []
    for item in request.items:
        results.append({
            "commodity": item.commodity,
            "market": item.market,
            "suggested_price": _predict_price_value(item.model_dump()),
            "currency": item.currency,
        })

    return {
        "status": "success",
        "count": len(results),
        "predictions": results,
    }


@router.post("/pricing/auto")
async def auto_pricing(request: AutoPricingRequest):
    base_price = _predict_price_value(request.model_dump())

    warnings: List[str] = []
    sources: List[str] = []

    demand_signal = request.demand_signal
    supply_volume = request.supply_volume

    platform_signals: Dict[str, float] = {}
    if request.use_live_signals:
        platform_signals, platform_warnings = await pricing_service.resolve_platform_signals(
            request.commodity, max(1, request.signal_window_days)
        )
        if platform_warnings:
            warnings.extend(platform_warnings)
        sources.append("platform_api")

    if demand_signal is None and platform_signals:
        demand_signal = platform_signals.get(
            "demand_qty") or platform_signals.get("demand_count")
    if supply_volume is None and platform_signals:
        supply_volume = platform_signals.get(
            "supply_qty") or platform_signals.get("supply_count")

    if request.searches:
        demand_signal = (demand_signal or 0.0) + float(request.searches)
    if request.carts:
        demand_signal = (demand_signal or 0.0) + float(request.carts)
    if request.orders:
        demand_signal = (demand_signal or 0.0) + float(request.orders)

    if request.active_listings:
        supply_volume = (supply_volume or 0.0) + float(request.active_listings)

    pressure = pricing_service.compute_price_pressure(
        demand_signal, supply_volume)
    max_adjustment = clamp(request.max_adjustment, 0.0, 1.0)
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


