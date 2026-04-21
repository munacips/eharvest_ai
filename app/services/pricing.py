from typing import Dict, List, Optional, Tuple

from app.services.common import clamp, extract_date, extract_quantity, get_first_present, within_days
from app.services.integrations import fetch_platform_list, match_commodity


def compute_price_pressure(demand: Optional[float], supply: Optional[float]) -> float:
    demand_value = demand or 0.0
    supply_value = supply or 0.0
    denom = abs(demand_value) + abs(supply_value) + 1.0
    pressure = (demand_value - supply_value) / denom
    return clamp(pressure, -1.0, 1.0)


async def resolve_platform_signals(commodity: str, window_days: int) -> Tuple[Dict[str, float], List[str]]:
    warnings: List[str] = []
    order_items, warn_items = await fetch_platform_list("/api/v1/order_items")
    if warn_items:
        warnings.append(warn_items)
    orders, warn_orders = await fetch_platform_list("/api/v1/orders")
    if warn_orders:
        warnings.append(warn_orders)
    produce, warn_produce = await fetch_platform_list("/api/v1/produce")
    if warn_produce:
        warnings.append(warn_produce)

    demand_count = 0
    demand_qty = 0.0
    for item in order_items:
        if commodity and not match_commodity(item, commodity):
            continue
        if not within_days(extract_date(item), window_days):
            continue
        qty = extract_quantity(item)
        demand_qty += qty if qty is not None else 1.0
        demand_count += 1

    if demand_count == 0:
        for order in orders:
            if not within_days(extract_date(order), window_days):
                continue
            demand_count += 1

    supply_count = 0
    supply_qty = 0.0
    for item in produce:
        if commodity and not match_commodity(item, commodity):
            continue
        status = get_first_present(item, ("status", "state"))
        if isinstance(status, str) and status.strip().lower() in ("inactive", "archived", "sold"):
            continue
        qty = extract_quantity(item)
        supply_qty += qty if qty is not None else 1.0
        supply_count += 1

    return {
        "demand_count": float(demand_count),
        "demand_qty": float(demand_qty),
        "supply_count": float(supply_count),
        "supply_qty": float(supply_qty),
    }, warnings
