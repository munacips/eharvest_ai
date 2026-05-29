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


def _normalize_region(value: str) -> str:
    if not value:
        return ""
    normalized = value.strip().lower()
    return config.REGION_ALIASES.get(normalized, normalized)


def _season_to_month(season: str) -> int | None:
    if not season:
        return None
    months = config.ZWE_SEASONS.get(season)
    if not months:
        return None
    return months[len(months) // 2]


@router.post("/recommendations/prescriptive")
async def prescriptive_recommendations(request: PrescriptiveRecommendationRequest):
    if request.top_n <= 0:
        raise HTTPException(
            status_code=400, detail="top_n must be greater than 0")

    warnings = []
    sources = []

    region_key = _normalize_region(request.region)

    if request.season and request.season not in config.ZWE_SEASONS:
        raise HTTPException(
            status_code=400,
            detail="season must be one of: " +
            ", ".join(config.ZWE_SEASONS.keys()),
        )

    if request.month is not None and not 1 <= request.month <= 12:
        raise HTTPException(
            status_code=400, detail="month must be between 1 and 12")

    target_month = request.month
    if target_month is None and request.season:
        target_month = _season_to_month(request.season)
    if target_month is None:
        target_month = datetime.now(UTC).month
    season = request.season or season_from_month(target_month)

    demand_map = {item.commodity.lower(): item.expected_demand for item in (
        request.demand_forecast or [])}

    climate_snapshot = request.climate
    climate_rain = climate_snapshot.rainfall_mm if climate_snapshot else None
    climate_temp = climate_snapshot.temperature_c if climate_snapshot else None

    latitude = request.latitude
    longitude = request.longitude
    if latitude is None or longitude is None:
        preset_coords = config.REGION_COORDS.get(region_key)
        if preset_coords:
            if latitude is None:
                latitude = preset_coords[0]
            if longitude is None:
                longitude = preset_coords[1]

    market_df = None
    if request.market_data:
        market_df = pd.DataFrame([item.model_dump()
                                 for item in request.market_data])
        market_df = apply_region_filter(market_df, request.region)

    preset_climate = config.REGION_SEASON_CLIMATE.get(
        region_key, {}).get(season)
    if preset_climate:
        used_preset = False
        if climate_rain is None and preset_climate.get("rainfall_mm") is not None:
            climate_rain = preset_climate.get("rainfall_mm")
            used_preset = True
        if climate_temp is None and preset_climate.get("temperature_c") is not None:
            climate_temp = preset_climate.get("temperature_c")
            used_preset = True
        if used_preset:
            sources.append("preset_climate")

    if request.auto_fetch_external:
        if (climate_rain is None or climate_temp is None) and (
            latitude is None or longitude is None
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "unable to infer climate for region; provide latitude/longitude, "
                    "climate snapshot, or a supported region preset"
                ),
            )
        if (climate_rain is None or climate_temp is None) and latitude is not None and longitude is not None:
            weather_payload, weather_source, warning = await fetch_weather_open_meteo(
                latitude, longitude, days=14
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
            "season is inferred from month when not provided",
            "preset climate values are used before fetching live weather",
        ],
    }
