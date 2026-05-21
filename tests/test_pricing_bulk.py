import json
from pathlib import Path

from tests.test_api import client
from app.services import pricing as pricing_service


def _make_base_payload(i: int) -> dict:
    crops = ["maize", "beans", "sorghum", "groundnuts", "rice"]
    return {
        "commodity": f"{crops[i % len(crops)]}",
        "market": f"market{i % 4}",
        "category": "cereals",
        "unit": "KG",
        "month": (i % 12) + 1,
        "latitude": -17.8 + (i * 0.01),
        "longitude": 31.0 + (i * 0.01),
        "currency": "USD",
        "priceflag": "actual",
        "admin1": "Manicaland",
    }


async def _fake_platform_signals(commodity, window_days):
    return ({"demand_qty": 100.0, "supply_qty": 50.0}, [])


def test_pricing_endpoints_bulk(monkeypatch, tmp_path: Path):
    # Ensure platform signals are predictable for auto pricing tests
    monkeypatch.setattr(
        pricing_service, "resolve_platform_signals", _fake_platform_signals)

    results = {"predict_price": [], "batch": [], "auto": []}

    # Up to 10 individual predict-price requests
    for i in range(10):
        payload = _make_base_payload(i)
        resp = client.post("/predict-price", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        results["predict_price"].append({"request": payload, "response": body})

    # Up to 10 batch requests (each with 5 items)
    for i in range(10):
        items = [_make_base_payload(j + i * 5) for j in range(5)]
        resp = client.post("/pricing/batch", json={"items": items})
        assert resp.status_code == 200
        body = resp.json()
        results["batch"].append(
            {"request_count": len(items), "response": body})

    # Up to 10 auto pricing requests
    for i in range(10):
        payload = _make_base_payload(i)
        # vary signals and options
        if i % 2 == 0:
            payload["use_live_signals"] = True
            payload["signal_window_days"] = 7
            payload["max_adjustment"] = 0.15
        else:
            payload["use_live_signals"] = False
            payload["demand_signal"] = 80 + i
            payload["supply_volume"] = 40 + i

        resp = client.post("/pricing/auto", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        results["auto"].append({"request": payload, "response": body})

    out_file = tmp_path / "pricing_test_results.json"
    out_file.write_text(json.dumps(results, indent=2))

    # also write to repository tests folder for the user's quick inspection
    repo_out = Path.cwd() / "tests" / "pricing_test_results.json"
    repo_out.write_text(json.dumps(results, indent=2))

    assert repo_out.exists()
