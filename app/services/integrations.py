from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app import config
from app.services.common import (
    extract_commodity,
    extract_cost,
    extract_date,
    extract_list_payload,
    get_first_present,
)


def platform_headers() -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if config.PLATFORM_API_KEY:
        headers["X-API-KEY"] = config.PLATFORM_API_KEY
    return headers


async def platform_get(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Optional[Any], Optional[str]]:
    if not config.PLATFORM_API_BASE_URL:
        return None, "platform_base_url_not_set"
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{config.PLATFORM_API_BASE_URL}{path}"
    try:
        async with httpx.AsyncClient(timeout=config.PLATFORM_API_TIMEOUT) as client:
            resp = await client.get(url, params=params, headers=platform_headers())
            resp.raise_for_status()
        return resp.json(), None
    except httpx.HTTPError as exc:
        return None, f"platform_api_error: {exc.__class__.__name__}"


async def fetch_platform_list(path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    data, warning = await platform_get(path, params=params)
    if data is None:
        return [], warning
    return extract_list_payload(data), warning


def match_commodity(item: Dict[str, Any], commodity: str) -> bool:
    if not commodity:
        return False
    found = extract_commodity(item)
    if not found:
        return False
    return commodity.strip().lower() in found.strip().lower()


async def fetch_platform_market_prices(
    region: Optional[str],
    commodity: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    items, warning = await fetch_platform_list("/api/v1/produce")
    results: List[Dict[str, Any]] = []
    for item in items:
        name = extract_commodity(item)
        if not name:
            continue
        if commodity and commodity.lower() not in name.lower():
            continue
        price = extract_cost(item)
        if price is None:
            continue
        market = get_first_present(
            item, ("market", "location", "city", "town"))
        region_value = get_first_present(
            item, ("region", "province", "admin1"))
        if region and isinstance(region_value, str):
            if region.lower() != region_value.lower():
                continue
        date_value = extract_date(item) or datetime.now(UTC)
        results.append({
            "date": date_value.strftime("%Y-%m-%d"),
            "commodity": name,
            "price": float(price),
            "market": market,
            "region": region_value,
        })
    return results, "platform_produce", warning


async def fetch_external_market_prices(
    region: Optional[str],
    commodity: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    if not config.MARKET_DATA_API_URL:
        return [], "not_configured", "market_data_api_url_not_set"
    try:
        async with httpx.AsyncClient(timeout=config.PLATFORM_API_TIMEOUT) as client:
            resp = await client.get(config.MARKET_DATA_API_URL, params={"region": region, "commodity": commodity})
            resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return [], "external_market_api", f"external_market_error: {exc.__class__.__name__}"

    items = extract_list_payload(data)
    results: List[Dict[str, Any]] = []
    for item in items:
        name = extract_commodity(item) or commodity
        price = extract_cost(item)
        if name is None or price is None:
            continue
        date_value = extract_date(item) or datetime.now(UTC)
        results.append({
            "date": date_value.strftime("%Y-%m-%d"),
            "commodity": name,
            "price": float(price),
            "market": get_first_present(item, ("market", "location", "city")),
            "region": get_first_present(item, ("region", "province", "admin1")),
        })
    return results, "external_market_api", None


async def fetch_weather_open_meteo(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    if not config.WEATHER_API_URL:
        return [], "not_configured", "weather_api_url_not_set"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_mean,precipitation_sum",
        "forecast_days": max(1, min(days, 14)),
        "timezone": "UTC",
    }
    try:
        async with httpx.AsyncClient(timeout=config.PLATFORM_API_TIMEOUT) as client:
            resp = await client.get(config.WEATHER_API_URL, params=params)
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
