import math
from datetime import UTC, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from app.config import ZWE_SEASONS


def season_from_month(month: int) -> str:
    for season, months in ZWE_SEASONS.items():
        if month in months:
            return season
    return "unknown"


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    return float(series.mean())


def coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    if isinstance(value, (int, float)):
        return value != 0
    return None


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def extract_list_payload(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("content", "items", "data", "results", "records", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def get_first_present(payload: Dict[str, Any], keys: Iterable[str]) -> Any:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return None


def get_nested_dict(payload: Dict[str, Any], keys: Iterable[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    for key in keys:
        value = payload.get(key)
        if isinstance(value, dict):
            return value
    return None


def parse_date(value: Any) -> Optional[datetime]:
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


def within_days(date_value: Optional[datetime], days: int) -> bool:
    if date_value is None:
        return True
    try:
        cutoff = datetime.now(UTC) - timedelta(days=days)
    except Exception:
        return True
    if date_value.tzinfo is None:
        date_value = date_value.replace(tzinfo=UTC)
    return date_value >= cutoff


def extract_commodity(item: Dict[str, Any]) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    direct = get_first_present(item, (
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
        nested_name = get_first_present(direct, (
            "commodity",
            "name",
            "produceName",
            "productName",
            "itemName",
        ))
        if isinstance(nested_name, str):
            return nested_name
    nested = get_nested_dict(item, ("produce", "product", "item"))
    if isinstance(nested, dict):
        nested_name = get_first_present(nested, (
            "commodity",
            "name",
            "produceName",
            "productName",
            "itemName",
        ))
        if isinstance(nested_name, str):
            return nested_name
    return None


def extract_quantity(item: Dict[str, Any]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    value = get_first_present(item, (
        "quantity",
        "qty",
        "amount",
        "volume",
        "weight",
        "units",
    ))
    return coerce_float(value)


def extract_cost(item: Dict[str, Any]) -> Optional[float]:
    if not isinstance(item, dict):
        return None
    value = get_first_present(item, (
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
    return coerce_float(value)


def extract_date(item: Dict[str, Any]) -> Optional[datetime]:
    if not isinstance(item, dict):
        return None
    value = get_first_present(item, (
        "created_at",
        "createdAt",
        "date",
        "orderDate",
        "requestedAt",
        "updated_at",
        "updatedAt",
    ))
    return parse_date(value)


def extract_lat_lon(payload: Dict[str, Any], lat_keys: Iterable[str], lon_keys: Iterable[str]) -> Tuple[Optional[float], Optional[float]]:
    if not isinstance(payload, dict):
        return None, None
    for lat_key in lat_keys:
        if lat_key in payload:
            lat_val = coerce_float(payload.get(lat_key))
            if lat_val is None:
                continue
            for lon_key in lon_keys:
                if lon_key in payload:
                    lon_val = coerce_float(payload.get(lon_key))
                    if lon_val is not None:
                        return lat_val, lon_val
    return None, None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * \
        math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return radius * c


def normalize_inverse(value: Optional[float], min_value: float, max_value: float) -> Optional[float]:
    if value is None:
        return None
    if max_value <= min_value:
        return 1.0
    scaled = (value - min_value) / (max_value - min_value)
    return clamp(1.0 - scaled, 0.0, 1.0)


def extract_location(payload: Dict[str, Any], lat_keys: Iterable[str], lon_keys: Iterable[str], nested_keys: Iterable[str]) -> Tuple[Optional[float], Optional[float]]:
    lat, lon = extract_lat_lon(payload, lat_keys, lon_keys)
    if lat is not None and lon is not None:
        return lat, lon
    nested = get_nested_dict(payload, nested_keys)
    if isinstance(nested, dict):
        lat, lon = extract_lat_lon(nested, lat_keys, lon_keys)
        if lat is not None and lon is not None:
            return lat, lon
    return None, None


def extract_origin_destination(payload: Dict[str, Any]) -> Tuple[Tuple[Optional[float], Optional[float]], Tuple[Optional[float], Optional[float]]]:
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
    origin = extract_location(payload, origin_lat_keys, origin_lon_keys,
                              ("origin", "pickup", "from", "source", "start", "location"))
    destination = extract_location(
        payload, dest_lat_keys, dest_lon_keys, ("destination", "dropoff", "to", "end"))
    return origin, destination


def extract_provider_location(payload: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    lat_keys = ("latitude", "lat", "current_lat",
                "currentLat", "base_lat", "baseLat")
    lon_keys = ("longitude", "lon", "lng", "current_lon",
                "current_lng", "currentLng", "base_lon", "base_lng", "baseLng")
    return extract_location(payload, lat_keys, lon_keys, ("location", "currentLocation", "baseLocation"))


def extract_capacity(payload: Dict[str, Any]) -> Optional[float]:
    return coerce_float(get_first_present(payload, ("capacity", "maxLoad", "max_capacity", "maxCapacity")))


def extract_rate_per_km(payload: Dict[str, Any]) -> Optional[float]:
    return coerce_float(get_first_present(payload, ("rate_per_km", "price_per_km", "cost_per_km", "ratePerKm", "costPerKm")))


def availability_flag(payload: Dict[str, Any]) -> Optional[bool]:
    flag = coerce_bool(get_first_present(
        payload, ("available", "isAvailable", "active", "enabled")))
    if flag is not None:
        return flag
    status = get_first_present(payload, ("status", "state"))
    if isinstance(status, str):
        return status.strip().lower() in ("available", "active", "online", "ready")
    return None
