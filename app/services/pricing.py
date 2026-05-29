import asyncio
from difflib import get_close_matches
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from app.services.common import clamp, extract_date, extract_quantity, get_first_present, within_days
from app.services.integrations import fetch_platform_list, match_commodity

INACTIVE_SUPPLY_STATUSES = {"inactive",
                            "archived", "sold", "unavailable", "deleted"}
ORDER_LINE_ITEM_KEYS = ("items", "order_items",
                        "orderItems", "line_items", "lineItems", "lines")

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


def normalize_pricing_request_data(data: Dict[str, Any]) -> Dict[str, Any]:
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


def build_pricing_model_input(data: Dict[str, Any], columns: List[str]) -> pd.DataFrame:
    normalized = normalize_pricing_request_data(data)

    for field in CATEGORICAL_MODEL_FIELDS:
        normalized[field] = _resolve_model_value(
            field,
            normalized.get(field),
            columns,
        )

    input_data = pd.DataFrame([normalized])
    input_encoded = pd.get_dummies(input_data)
    return input_encoded.reindex(columns=columns, fill_value=0)


def predict_price_value(data: Dict[str, Any], model, columns: List[str]) -> float:
    final_input = build_pricing_model_input(data, columns)
    prediction = model.predict(final_input)
    return round(max(0.0, float(prediction[0])), 2)


def _normalize_window_days(window_days: int) -> int:
    """Keep the lookback window usable even if an invalid value slips through."""
    try:
        return max(1, int(window_days))
    except (TypeError, ValueError):
        return 30


def _append_warning(warnings: List[str], warning: Optional[str]) -> None:
    if warning and warning not in warnings:
        warnings.append(warning)


def _iter_nested_items(payload: Dict[str, Any], keys: Iterable[str]) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in keys:
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _is_active_supply_record(item: Dict[str, Any]) -> bool:
    status = get_first_present(item, ("status", "state"))
    if not isinstance(status, str):
        return True
    return status.strip().lower() not in INACTIVE_SUPPLY_STATUSES


def _matches_order_commodity(order: Dict[str, Any], commodity: str) -> bool:
    """Check both top-level order fields and nested line items for a commodity match."""
    if not commodity:
        return True
    if match_commodity(order, commodity):
        return True
    return any(match_commodity(item, commodity) for item in _iter_nested_items(order, ORDER_LINE_ITEM_KEYS))


def _extract_order_quantity(order: Dict[str, Any], commodity: str) -> Optional[float]:
    """Prefer commodity-specific line-item quantities before falling back to order totals."""
    line_items = _iter_nested_items(order, ORDER_LINE_ITEM_KEYS)
    if line_items:
        relevant_items = (
            [item for item in line_items if match_commodity(item, commodity)]
            if commodity
            else line_items
        )
        if relevant_items:
            return sum(extract_quantity(item) or 1.0 for item in relevant_items)

    direct_qty = extract_quantity(order)
    if direct_qty is not None:
        return direct_qty
    return None


def _aggregate_records(
    records: List[Dict[str, Any]],
    commodity: str,
    window_days: int,
    *,
    active_only: bool = False,
) -> Tuple[int, float]:
    """Count records and sum their quantities, using 1.0 when a quantity is missing."""
    count = 0
    quantity_total = 0.0
    for record in records:
        if commodity and not match_commodity(record, commodity):
            continue
        if not within_days(extract_date(record), window_days):
            continue
        if active_only and not _is_active_supply_record(record):
            continue
        quantity_total += extract_quantity(record) or 1.0
        count += 1
    return count, quantity_total


def compute_price_pressure(demand: Optional[float], supply: Optional[float]) -> float:
    """Return a bounded demand-vs-supply pressure score in the range [-1, 1]."""
    demand_value = demand or 0.0
    supply_value = supply or 0.0

    # Add a small constant so missing or zero-valued signals do not produce a
    # divide-by-zero error. This keeps the score stable near zero activity.
    denom = abs(demand_value) + abs(supply_value) + 1.0
    pressure = (demand_value - supply_value) / denom
    return clamp(pressure, -1.0, 1.0)


async def resolve_platform_signals(commodity: str, window_days: int) -> Tuple[Dict[str, float], List[str]]:
    """Build recent demand and supply signals from the platform APIs.

    Demand is derived from order items first because those records are usually
    the most specific source for commodity-level quantity. When they are absent
    or do not match the requested commodity, the function falls back to orders.
    Supply comes from active produce listings that fall within the same recent
    lookback window.
    """
    normalized_window_days = _normalize_window_days(window_days)
    warnings: List[str] = []

    responses = await asyncio.gather(
        fetch_platform_list("/api/v1/order_items"),
        fetch_platform_list("/api/v1/orders"),
        fetch_platform_list("/api/v1/produce"),
    )
    (order_items, warn_items), (orders,
                                warn_orders), (produce, warn_produce) = responses

    _append_warning(warnings, warn_items)
    _append_warning(warnings, warn_orders)
    _append_warning(warnings, warn_produce)

    demand_count, demand_qty = _aggregate_records(
        order_items,
        commodity,
        normalized_window_days,
    )

    # Some platform deployments expose only top-level orders. In that case, try
    # to recover a commodity-aware demand signal from the order payload.
    if demand_count == 0:
        for order in orders:
            if not within_days(extract_date(order), normalized_window_days):
                continue
            if commodity and not _matches_order_commodity(order, commodity):
                continue
            demand_qty += _extract_order_quantity(order, commodity) or 1.0
            demand_count += 1

    supply_count, supply_qty = _aggregate_records(
        produce,
        commodity,
        normalized_window_days,
        active_only=True,
    )

    return {
        "demand_count": float(demand_count),
        "demand_qty": float(demand_qty),
        "supply_count": float(supply_count),
        "supply_qty": float(supply_qty),
    }, warnings
