import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.common import haversine_km
from app.services.logistics import score_logistics_candidates


def test_score_logistics_candidates_uses_total_travel_for_rate_based_cost():
    request_payload = {
        "origin_lat": -17.8,
        "origin_lon": 31.0,
        "destination_lat": -18.2,
        "destination_lon": 31.6,
        "quantity": 5,
    }
    providers = [
        {
            "id": "prov-1",
            "latitude": -17.9,
            "longitude": 31.1,
            "cost_per_km": 0.8,
            "capacity": 8,
        }
    ]

    result = score_logistics_candidates(request_payload, providers)
    match = result["matches"][0]

    pickup_distance = haversine_km(-17.8, 31.0, -17.9, 31.1)
    route_distance = haversine_km(-17.8, 31.0, -18.2, 31.6)
    expected_cost = round(0.8 * (pickup_distance + route_distance), 2)

    assert match["pickup_distance_km"] == round(pickup_distance, 2)
    assert match["route_distance_km"] == round(route_distance, 2)
    assert match["estimated_cost"] == expected_cost


def test_score_logistics_candidates_tolerates_partial_coordinates_and_invalid_weights():
    request_payload = {
        "origin_lat": -17.8,
        "quantity": 5,
    }
    providers = [
        {
            "id": "prov-1",
            "latitude": -17.9,
            "cost": 10,
            "capacity": 5,
        }
    ]

    result = score_logistics_candidates(
        request_payload,
        providers,
        weights={"cost": "bad", "distance": -1, "capacity": 0},
    )
    match = result["matches"][0]

    assert result["route_distance_km"] is None
    assert match["pickup_distance_km"] is None
    assert match["estimated_cost"] == 10.0
    assert match["capacity_score"] == 1.0


def test_score_logistics_candidates_rejects_invalid_provider_payloads():
    result = score_logistics_candidates(
        {"origin_lat": -17.8, "origin_lon": 31.0},
        ["bad-provider-payload"],
    )

    assert result["matches"] == []
    assert result["rejected"] == [
        {
            "provider": "bad-provider-payload",
            "reason": "invalid_provider_payload",
        }
    ]
