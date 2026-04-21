from typing import Any, Dict, List, Optional

from app.services.common import (
    availability_flag,
    coerce_float,
    extract_capacity,
    extract_cost,
    extract_origin_destination,
    extract_provider_location,
    extract_quantity,
    extract_rate_per_km,
    get_first_present,
    haversine_km,
    normalize_inverse,
)


def score_logistics_candidates(
    request_payload: Dict[str, Any],
    providers: List[Dict[str, Any]],
    weights: Optional[Dict[str, float]] = None,
    max_distance_km: Optional[float] = None,
) -> Dict[str, Any]:
    weights = weights or {}
    weight_cost = float(weights.get("cost", 0.4))
    weight_distance = float(weights.get("distance", 0.4))
    weight_capacity = float(weights.get("capacity", 0.2))

    origin, destination = extract_origin_destination(request_payload)
    route_distance = None
    if origin[0] is not None and destination[0] is not None:
        route_distance = haversine_km(
            origin[0], origin[1], destination[0], destination[1])

    demand_qty = extract_quantity(request_payload) or coerce_float(
        get_first_present(request_payload, ("weight", "load", "volume"))
    )

    scored: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    distances: List[float] = []
    costs: List[float] = []

    for provider in providers:
        provider_location = extract_provider_location(provider)
        pickup_distance = None
        if origin[0] is not None and provider_location[0] is not None:
            pickup_distance = haversine_km(
                origin[0], origin[1], provider_location[0], provider_location[1])

        if max_distance_km is not None and pickup_distance is not None:
            if pickup_distance > max_distance_km:
                rejected.append({
                    "provider": provider,
                    "reason": "pickup_distance_exceeds_max",
                    "pickup_distance_km": round(pickup_distance, 2),
                })
                continue

        base_cost = extract_cost(provider)
        rate_per_km = extract_rate_per_km(provider)
        distance_for_cost = route_distance or pickup_distance or 0.0
        estimated_cost = None
        if base_cost is not None or rate_per_km is not None:
            estimated_cost = (base_cost or 0.0) + \
                (rate_per_km or 0.0) * distance_for_cost

        capacity = extract_capacity(provider)
        capacity_score = None
        if capacity is not None and demand_qty is not None:
            capacity_score = 1.0 if capacity >= demand_qty else 0.4
        elif capacity is not None:
            capacity_score = 0.8

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
            "pickup_distance_km": None if item["pickup_distance_km"] is None else round(item["pickup_distance_km"], 2),
            "route_distance_km": None if item["route_distance_km"] is None else round(item["route_distance_km"], 2),
            "estimated_cost": None if item["estimated_cost"] is None else round(item["estimated_cost"], 2),
            "capacity_score": item["capacity_score"],
            "availability_penalty": item["availability_penalty"],
        })

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    return {
        "route_distance_km": None if route_distance is None else round(route_distance, 2),
        "matches": results,
        "rejected": rejected,
    }
