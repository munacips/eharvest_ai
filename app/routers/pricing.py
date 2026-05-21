from difflib import get_close_matches
from typing import Any, Dict, List

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.models import AutoPricingRequest, BatchPricePredictionRequest, PricePredictionRequest
from app.services import pricing as pricing_service
from app.services.common import clamp
from app.state import dynamic_pricing_model, model_columns

router = APIRouter()

CATEGORICAL_MODEL_FIELDS = [
    "commodity",
    "market",
    "category",
    "unit",
    "currency",
    "priceflag",
    "pricetype",
    "admin1",
    "admin2",
]

PRICING_CATEGORY_ALIASES = {
    "cereals": "cereals and tubers",
    "cereal": "cereals and tubers",
    "pulses": "pulses and nuts",
    "pulse": "pulses and nuts",
    "nuts": "pulses and nuts",
    "oil": "oil and fats",
    "oils": "oil and fats",
    "fat": "oil and fats",
    "fats": "oil and fats",
    "meat": "meat, fish and eggs",
    "fish": "meat, fish and eggs",
    "eggs": "meat, fish and eggs",
}


def _normalize_pricing_text(value: str) -> str:
    return value.strip().title()


def _normalize_pricing_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(data)
    for field in ["commodity", "market", "category", "admin1", "admin2", "pricetype"]:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = _normalize_pricing_text(value)
    for field in ["unit", "currency"]:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = value.strip().upper()
    value = normalized.get("priceflag")
    if isinstance(value, str):
        normalized["priceflag"] = value.strip().lower()
    return normalized


def _extract_allowed_model_values(columns: List[str], field: str) -> List[str]:
    prefix = f"{field}_"
    return [column[len(prefix):] for column in columns if column.startswith(prefix)]


def _resolve_model_value(field: str, value: Any, columns: List[str]) -> Any:
    if not isinstance(value, str):
        return value

    allowed_values = _extract_allowed_model_values(columns, field)
    if not allowed_values:
        return value

    raw_value = value.strip()
    if not raw_value:
        return value

    candidates = [raw_value]
    lowered = raw_value.casefold()
    if field == "category" and lowered in PRICING_CATEGORY_ALIASES:
        candidates.insert(0, PRICING_CATEGORY_ALIASES[lowered])

    exact_lookup = {allowed.casefold(): allowed for allowed in allowed_values}
    for candidate in candidates:
        exact_match = exact_lookup.get(candidate.casefold())
        if exact_match:
            return exact_match

    for candidate in candidates:
        candidate_text = candidate.casefold()
        substring_matches = [
            allowed for allowed in allowed_values
            if candidate_text in allowed.casefold() or allowed.casefold() in candidate_text
        ]
        if len(substring_matches) == 1:
            return substring_matches[0]

    close_match = get_close_matches(
        candidates[0].casefold(),
        [allowed.casefold() for allowed in allowed_values],
        n=1,
        cutoff=0.6,
    )
    if close_match:
        return exact_lookup[close_match[0]]

    return value


def _build_pricing_model_input(data: Dict[str, Any]) -> pd.DataFrame:
    normalized = _normalize_pricing_request_data(data)

    # Match request categories to the trained model vocabulary so we keep signal
    # even when clients vary casing or use a simplified label like "cereals".
    for field in CATEGORICAL_MODEL_FIELDS:
        normalized[field] = _resolve_model_value(
            field,
            normalized.get(field),
            model_columns,
        )

    input_data = pd.DataFrame([normalized])
    input_encoded = pd.get_dummies(input_data)
    return input_encoded.reindex(columns=model_columns, fill_value=0)


def _predict_price_value(data: Dict[str, Any]) -> float:
    final_input = _build_pricing_model_input(data)
    prediction = dynamic_pricing_model.predict(final_input)
    return round(max(0.0, float(prediction[0])), 2)


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
