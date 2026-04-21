from typing import List, Optional

from fastapi import APIRouter, Query

from app.services import integrations as integrations_service

router = APIRouter()


@router.get("/integrations/weather")
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
    weather_payload, source, warning = await integrations_service.fetch_weather_open_meteo(latitude, longitude, days=days)
    response = {
        "source": source,
        "weather": weather_payload,
    }
    if warning:
        response["warnings"] = [warning]
    return response


@router.get("/integrations/market-prices")
async def integrations_market_prices(
    region: Optional[str] = Query(
        None, examples={"default": {"value": "Manicaland"}}),
    commodity: Optional[str] = Query(
        None, examples={"default": {"value": "maize"}}),
):
    warnings: List[str] = []
    sources: List[str] = []

    market_payload, market_source, warning = await integrations_service.fetch_platform_market_prices(region, commodity)
    if market_payload:
        sources.append(market_source)
    if warning:
        warnings.append(warning)

    if not market_payload:
        external_payload, external_source, warning = await integrations_service.fetch_external_market_prices(region, commodity)
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
