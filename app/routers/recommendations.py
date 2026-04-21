from datetime import UTC, datetime

import pandas as pd
from fastapi import APIRouter, HTTPException

from app import config
from app.models import PrescriptiveRecommendationRequest
from app.services.common import safe_mean, season_from_month
from app.services.forecasting import apply_region_filter
from app.services.integrations import (
    fetch_external_market_prices,
    fetch_platform_market_prices,
    fetch_weather_open_meteo,
)

router = APIRouter()


@router.post("/recommendations/prescriptive")
async def prescriptive_recommendations(request: PrescriptiveRecommendationRequest):
    if request.top_n <= 0:
        raise HTTPException(
            status_code=400, detail="top_n must be greater than 0")

    warnings = []
    sources = []

    target_month = request.month or datetime.now(UTC).month
    season = request.season or season_from_month(target_month)

    demand_map = {item.commodity.lower(): item.expected_demand for item in (
        request.demand_forecast or [])}

    climate_rain = request.climate.rainfall_mm
    climate_temp = request.climate.temperature_c

    market_df = None
    if request.market_data:
        market_df = pd.DataFrame([item.model_dump()
                                 for item in request.market_data])
        market_df = apply_region_filter(market_df, request.region)

    if request.auto_fetch_external:
        if (climate_rain is None or climate_temp is None) and request.latitude is not None and request.longitude is not None:
            weather_payload, weather_source, warning = await fetch_weather_open_meteo(
                request.latitude, request.longitude, days=14
            )
            if weather_payload:
                weather_df = pd.DataFrame(weather_payload)
                if climate_rain is None:
                    climate_rain = safe_mean(weather_df.get("rainfall_mm"))
                if climate_temp is None:
                    climate_temp = safe_mean(weather_df.get("temperature_c"))
                sources.append(weather_source)
            if warning:
                warnings.append(warning)

        if market_df is None or market_df.empty:
            market_payload, market_source, warning = await fetch_platform_market_prices(request.region)
            if market_payload:
                market_df = pd.DataFrame(market_payload)
                sources.append(market_source)
            if warning:
                warnings.append(warning)

            if market_df is None or market_df.empty:
                external_payload, external_source, warning = await fetch_external_market_prices(request.region)
                if external_payload:
                    market_df = pd.DataFrame(external_payload)
                    sources.append(external_source)
                if warning:
                    warnings.append(warning)

    scored = []
    for crop, profile in config.CROP_PROFILES.items():
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

        cost = config.CROP_COST_USD_PER_HA.get(crop, 400)
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
            "climate_fit": climate_score == 1.0,
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
                "planting_months": item["planting_months"],
            },
            "market_targets": markets,
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
            "market targets use average prices in provided market_data",
        ],
    }
