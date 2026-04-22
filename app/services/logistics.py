from typing import Any, Dict, List, Optional, Tuple

from app.services.common import (
    availability_flag,
    coerce_float,
    extract_capacity,
    extract_origin_destination,
    extract_provider_location,
    extract_quantity,
    extract_rate_per_km,
    get_first_present,
    haversine_km,
    normalize_inverse,
)


DEFAULT_SCORE_WEIGHTS = {
    "cost": 0.4,
    "distance": 0.4,
    "capacity": 0.2,
}


def _has_complete_location(location: Tuple[Optional[float], Optional[float]]) -> bool:
    return location[0] is not None and location[1] is not None


def _distance_between(
    start: Tuple[Optional[float], Optional[float]],
    end: Tuple[Optional[float], Optional[float]],
) -> Optional[float]:
    if not (_has_complete_location(start) and _has_complete_location(end)):
        return None

    start_lat, start_lon = start
    end_lat, end_lon = end
    return haversine_km(
        float(start_lat),
        float(start_lon),
        float(end_lat),
        float(end_lon),
    )


def _resolve_weights(weights: Optional[Dict[str, float]]) -> Dict[str, float]:
    resolved: Dict[str, float] = {}
    for key, default in DEFAULT_SCORE_WEIGHTS.items():
        raw_value = None if weights is None else coerce_float(weights.get(key))
        resolved[key] = raw_value if raw_value is not None and raw_value >= 0 else default

    if sum(resolved.values()) <= 0:
        return DEFAULT_SCORE_WEIGHTS.copy()
    return resolved


def _extract_base_fee(
    provider: Dict[str, Any],
    rate_per_km: Optional[float],
) -> Optional[float]:
    explicit_base_fee = coerce_float(
        get_first_present(
            provider,
            (
                "base_fee",
                "baseFee",
                "base_cost",
                "baseCost",
                "flat_fee",
                "flatFee",
                "flat_cost",
                "flatCost",
                "basePrice",
                "base_price",
                "cost",
            ),
        )
    )
    if explicit_base_fee is not None:
        return explicit_base_fee

    # Plain "price" / "rate" fields are ambiguous, so only treat them as a flat
    # fee when there is no explicit per-km rate in the payload.
    if rate_per_km is not None:
        return None
    return coerce_float(get_first_present(provider, ("price", "rate")))


def _capacity_fit_score(
    capacity: Optional[float],
    demand_qty: Optional[float],
) -> Optional[float]:
    if capacity is None:
        return None
    if capacity <= 0:
        return 0.0
    if demand_qty is None or demand_qty <= 0:
        return 0.8
    return min(1.0, capacity / demand_qty)


def _round_optional(value: Optional[float], digits: int = 2) -> Optional[float]:
    return None if value is None else round(value, digits)


def score_logistics_candidates(
    request_payload: Dict[str, Any],
    providers: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    max_distance_km: Optional[float] = None,
) -> Dict[str, Any]:
    """Rank logistics providers using distance, cost, capacity, and availability.

    The scorer is intentionally tolerant of partial payloads: if a provider is
    missing enough data for one factor, that factor is skipped rather than
    failing the whole match.
    """
    resolved_weights = _resolve_weights(weights)
    weight_cost = resolved_weights["cost"]
    weight_distance = resolved_weights["distance"]
    weight_capacity = resolved_weights["capacity"]

    distance_limit_km = coerce_float(max_distance_km)
    if distance_limit_km is not None and distance_limit_km < 0:
        distance_limit_km = None

    origin, destination = extract_origin_destination(request_payload)
    route_distance = _distance_between(origin, destination)

    # Quantity can arrive under different names depending on the caller.
    demand_qty = extract_quantity(request_payload) or coerce_float(
        get_first_present(request_payload, ("weight", "load", "volume"))
    )

    scored: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    distances: List[float] = []
    costs: List[float] = []

    for provider in providers:
        if not isinstance(provider, dict):
            rejected.append({
                "provider": provider,
                "reason": "invalid_provider_payload",
            })
            continue

        provider_location = extract_provider_location(provider)
        pickup_distance = _distance_between(origin, provider_location)

        if distance_limit_km is not None and pickup_distance is not None:
            if pickup_distance > distance_limit_km:
                rejected.append({
                    "provider": provider,
                    "reason": "pickup_distance_exceeds_max",
                    "pickup_distance_km": round(pickup_distance, 2),
                })
                continue

        rate_per_km = extract_rate_per_km(provider)
        base_cost = _extract_base_fee(provider, rate_per_km)
        estimated_cost = None
        if base_cost is not None or rate_per_km is not None:
            # When possible, include both the deadhead trip to the pickup point
            # and the actual delivery route in the billable distance estimate.
            distance_for_cost = 0.0
            if pickup_distance is not None:
                distance_for_cost += pickup_distance
            if route_distance is not None:
                distance_for_cost += route_distance
            estimated_cost = (base_cost or 0.0) + (rate_per_km or 0.0) * distance_for_cost

        capacity = extract_capacity(provider)
        capacity_score = _capacity_fit_score(capacity, demand_qty)

        availability = availability_flag(provider)
        availability_penalty = 1.0
        if availability is False:
            availability_penalty = 0.6

        if pickup_distance is not None:
            distances.append(pickup_distance)
        if estimated_cost is not None:
            costs.append(estimated_cost)

        scored.append({
            "provider": provider,
            "pickup_distance_km": pickup_distance,
            "route_distance_km": route_distance,
            "estimated_cost": estimated_cost,
            "capacity_score": capacity_score,
            "availability_penalty": availability_penalty,
        })

    min_distance = min(distances) if distances else 0.0
    max_distance = max(distances) if distances else 0.0
    min_cost = min(costs) if costs else 0.0
    max_cost = max(costs) if costs else 0.0

    results: List[Dict[str, Any]] = []
    for item in scored:
        distance_score = normalize_inverse(
            item["pickup_distance_km"], min_distance, max_distance)
        cost_score = normalize_inverse(
            item["estimated_cost"], min_cost, max_cost)
        capacity_score = item["capacity_score"]

        # Only blend together signals we were actually able to compute.
        total_weight = 0.0
        score = 0.0
        if distance_score is not None:
            score += weight_distance * distance_score
            total_weight += weight_distance
        if cost_score is not None:
            score += weight_cost * cost_score
            total_weight += weight_cost
        if capacity_score is not None:
            score += weight_capacity * capacity_score
            total_weight += weight_capacity
        if total_weight > 0:
            score = score / total_weight

        score *= item["availability_penalty"]

        results.append({
            "provider": item["provider"],
            "score": round(score, 4),
            "pickup_distance_km": _round_optional(item["pickup_distance_km"]),
            "route_distance_km": _round_optional(item["route_distance_km"]),
            "estimated_cost": _round_optional(item["estimated_cost"]),
            "capacity_score": _round_optional(item["capacity_score"], 4),
            "availability_penalty": item["availability_penalty"],
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "route_distance_km": _round_optional(route_distance),
        "matches": results,
        "rejected": rejected,
    }
