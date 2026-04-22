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
    """Build authentication headers for the Spring Boot platform."""
    headers: Dict[str, str] = {}
    if config.PLATFORM_API_KEY:
        headers["X-API-KEY"] = config.PLATFORM_API_KEY
    return headers


def _normalize_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split()).strip().casefold()
    return normalized or None


def _matches_text_filter(value: Any, expected: Optional[str]) -> bool:
    expected_text = _normalize_text(expected)
    if expected_text is None:
        return True
    value_text = _normalize_text(value)
    if value_text is None:
        return False
    return expected_text in value_text


def _extract_market_name(item: Dict[str, Any]) -> Optional[str]:
    market = get_first_present(item, ("market", "location", "city", "town"))
    if isinstance(market, str):
        return market
    if isinstance(market, dict):
        nested_market = get_first_present(
            market, ("name", "market", "location", "city", "town")
        )
        if isinstance(nested_market, str):
            return nested_market
    return None


def _extract_region_name(item: Dict[str, Any]) -> Optional[str]:
    region = get_first_present(item, ("region", "province", "admin1"))
    if isinstance(region, str):
        return region
    if isinstance(region, dict):
        nested_region = get_first_present(
            region, ("name", "region", "province", "admin1")
        )
        if isinstance(nested_region, str):
            return nested_region
    return None


def _is_active_market_record(item: Dict[str, Any]) -> bool:
    status = get_first_present(item, ("status", "state"))
    if not isinstance(status, str):
        return True
    return status.strip().lower() not in {
        "inactive",
        "archived",
        "sold",
        "unavailable",
        "deleted",
    }


def _matches_region(
    item: Dict[str, Any],
    region: Optional[str],
    *,
    region_value: Optional[str] = None,
    market_value: Optional[str] = None,
) -> bool:
    if not region:
        return True

    region_text = region_value or _extract_region_name(item)
    market_text = market_value or _extract_market_name(item)

    # When a caller explicitly asks for a region, only keep rows we can tie to
    # that geography through a region field or a market/location field.
    return _matches_text_filter(region_text, region) or _matches_text_filter(
        market_text, region
    )


async def _http_get_json(
    url: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    error_prefix: str,
) -> Tuple[Optional[Any], Optional[str]]:
    try:
        async with httpx.AsyncClient(
            timeout=config.PLATFORM_API_TIMEOUT,
            follow_redirects=True,
        ) as client:
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
    except httpx.TimeoutException:
        return None, f"{error_prefix}: TimeoutException"
    except httpx.HTTPStatusError as exc:
        status_code = (
            exc.response.status_code if exc.response is not None else "unknown"
        )
        return None, f"{error_prefix}: HTTPStatusError:{status_code}"
    except httpx.RequestError as exc:
        return None, f"{error_prefix}: {exc.__class__.__name__}"

    if resp.status_code == httpx.codes.NO_CONTENT or not resp.content:
        return None, None

    try:
        return resp.json(), None
    except ValueError:
        return None, f"{error_prefix}: InvalidJSON"


async def platform_get(
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """Perform a GET against the platform API and normalize its response shape."""
    if not config.PLATFORM_API_BASE_URL:
        return None, "platform_base_url_not_set"
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{config.PLATFORM_API_BASE_URL}{path}"
    return await _http_get_json(
        url,
        params=params,
        headers=platform_headers(),
        error_prefix="platform_api_error",
    )


async def fetch_platform_list(
    path: str,
    params: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Fetch a platform endpoint and unwrap common list envelope shapes."""
    data, warning = await platform_get(path, params=params)
    if data is None:
        return [], warning
    return extract_list_payload(data), warning


def match_commodity(item: Dict[str, Any], commodity: str) -> bool:
    """Return True when the item commodity loosely matches the requested text."""
    if not commodity:
        return False
    found = extract_commodity(item)
    if not found:
        return False
    return _matches_text_filter(found, commodity)


async def fetch_platform_market_prices(
    region: Optional[str],
    commodity: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Derive normalized market price records from platform produce listings."""
    items, warning = await fetch_platform_list("/api/v1/produce")
    results: List[Dict[str, Any]] = []
    retrieved_at = datetime.now(UTC)
    for item in items:
        # Produce records act as live market signals, so inactive listings
        # should not contribute to price recommendations or forecasting.
        if not _is_active_market_record(item):
            continue
        name = extract_commodity(item)
        if not name:
            continue
        if commodity and not match_commodity(item, commodity):
            continue
        price = extract_cost(item)
        if price is None:
            continue
        market = _extract_market_name(item)
        region_value = _extract_region_name(item)
        if not _matches_region(
            item,
            region,
            region_value=region_value,
            market_value=market,
        ):
            continue
        date_value = extract_date(item) or retrieved_at
        results.append({
            "date": date_value.strftime("%Y-%m-%d"),
            "commodity": name,
            "price": float(price),
            "market": market,
            "region": region_value,
        })
    results.sort(key=lambda entry: entry["date"], reverse=True)
    return results, "platform_produce", warning


async def fetch_external_market_prices(
    region: Optional[str],
    commodity: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Fetch normalized market prices from the configured external market API."""
    if not config.MARKET_DATA_API_URL:
        return [], "external_market_api", "market_data_api_url_not_set"

    params = {
        key: value
        for key, value in {"region": region, "commodity": commodity}.items()
        if value is not None
    }
    data, warning = await _http_get_json(
        config.MARKET_DATA_API_URL,
        params=params or None,
        error_prefix="external_market_error",
    )
    if data is None:
        return [], "external_market_api", warning

    items = extract_list_payload(data)
    results: List[Dict[str, Any]] = []
    retrieved_at = datetime.now(UTC)
    for item in items:
        name = extract_commodity(item) or commodity
        price = extract_cost(item)
        if name is None or price is None:
            continue
        if commodity and not _matches_text_filter(name, commodity):
            continue
        market = _extract_market_name(item)
        region_value = _extract_region_name(item)
        if not _matches_region(
            item,
            region,
            region_value=region_value,
            market_value=market,
        ):
            continue
        date_value = extract_date(item) or retrieved_at
        results.append({
            "date": date_value.strftime("%Y-%m-%d"),
            "commodity": name,
            "price": float(price),
            "market": market,
            "region": region_value,
        })
    results.sort(key=lambda entry: entry["date"], reverse=True)
    return results, "external_market_api", warning


async def fetch_weather_open_meteo(
    latitude: float,
    longitude: float,
    days: int = 7,
) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
    """Fetch a daily weather forecast from Open-Meteo in a normalized format."""
    if not config.WEATHER_API_URL:
        return [], "open_meteo", "weather_api_url_not_set"
    if not (-90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0):
        return [], "open_meteo", "weather_api_invalid_coordinates"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_mean,precipitation_sum",
        "forecast_days": max(1, min(days, 14)),
        "timezone": "UTC",
    }
    data, warning = await _http_get_json(
        config.WEATHER_API_URL,
        params=params,
        error_prefix="weather_api_error",
    )
    if data is None:
        return [], "open_meteo", warning

    daily = data.get("daily", {}) if isinstance(data, dict) else {}
    dates = daily.get("time") or []
    temps = daily.get("temperature_2m_mean") or []
    rains = daily.get("precipitation_sum") or []
    results: List[Dict[str, Any]] = []
    for idx, date_str in enumerate(dates):
        # Open-Meteo returns aligned daily arrays, so the index is the join key.
        results.append({
            "date": date_str,
            "rainfall_mm": rains[idx] if idx < len(rains) else None,
            "temperature_c": temps[idx] if idx < len(temps) else None,
        })
    return results, "open_meteo", warning
