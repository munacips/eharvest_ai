import main
import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

# Add project root to sys.path so `import main` works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


client = TestClient(main.app)


def make_pricing_payload(i: int):
    crops = ["maize", "beans", "sorghum", "groundnuts", "rice"]
    return {
        "commodity": crops[i % len(crops)],
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


def call_predict_price(i: int):
    payload = make_pricing_payload(i)
    resp = client.post("/predict-price", json=payload)
    return payload, resp.status_code, resp.json()


def call_batch(i: int):
    items = [make_pricing_payload(j + i * 5) for j in range(5)]
    resp = client.post("/pricing/batch", json={"items": items})
    return items, resp.status_code, resp.json()


def call_auto(i: int):
    payload = make_pricing_payload(i)
    if i % 2 == 0:
        payload["use_live_signals"] = True
        payload["signal_window_days"] = 7
        payload["max_adjustment"] = 0.15
    else:
        payload["use_live_signals"] = False
        payload["demand_signal"] = 80 + i
        payload["supply_volume"] = 40 + i
    resp = client.post("/pricing/auto", json=payload)
    return payload, resp.status_code, resp.json()


def generate_pricing_results():
    results = {"predict_price": [], "batch": [], "auto": []}

    for i in range(10):
        req, status, resp = call_predict_price(i)
        results["predict_price"].append(
            {"request": req, "status": status, "response": resp})

    for i in range(10):
        items, status, resp = call_batch(i)
        results["batch"].append({"request_count": len(
            items), "status": status, "response": resp})

    for i in range(10):
        req, status, resp = call_auto(i)
        results["auto"].append(
            {"request": req, "status": status, "response": resp})

    out = Path("tests") / "pricing_real_model_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out


def generate_forecasting_results():
    results = {"forecasts": [],
               "exact_vocab_forecasts": [], "demand_supply": None}

    commodities = ["maize", "beans", "sorghum", "groundnuts", "rice"]
    exact_commodities = [
        "Maize",
        "Maize meal",
        "Maize meal (white, fortified)",
        "Beans",
        "Groundnuts (shelled)",
    ]

    for c in commodities:
        resp = client.get(
            f"/forecast/{c}", params={"periods": 5, "region": "Manicaland", "visual": "false"})
        results["forecasts"].append(
            {"commodity": c, "status": resp.status_code, "response": resp.json()})

    for c in exact_commodities:
        resp = client.get(
            f"/forecast/{c}", params={"periods": 5, "region": "Manicaland", "visual": "false"})
        results["exact_vocab_forecasts"].append(
            {"commodity": c, "status": resp.status_code, "response": resp.json()})

    payload = {
        "region": "Manicaland",
        "periods": 3,
        "commodities": ["maize", "beans"],
    }
    resp = client.post("/forecast/demand-supply", json=payload)
    results["demand_supply"] = {
        "status": resp.status_code, "response": resp.json()}

    out = Path("tests") / "forecasting_real_model_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return out


def main():
    # give app a moment
    time.sleep(0.2)
    p = generate_pricing_results()
    f = generate_forecasting_results()
    print("Wrote:", p, f)


if __name__ == "__main__":
    main()
