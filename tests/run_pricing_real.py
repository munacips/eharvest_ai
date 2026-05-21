import json
import time
import urllib.request

BASE = "http://127.0.0.1:8001"

crops = ["maize", "beans", "sorghum", "groundnuts", "rice"]


def make_payload(i: int):
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


def _post_json(path: str, data: dict, timeout=10):
    url = f"{BASE}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
                                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.load(resp)


def call_predict_price(i):
    payload = make_payload(i)
    status, body = _post_json("/predict-price", payload)
    return payload, status, body


def call_batch(i):
    items = [make_payload(j + i * 5) for j in range(5)]
    status, body = _post_json("/pricing/batch", {"items": items})
    return items, status, body


def call_auto(i):
    payload = make_payload(i)
    if i % 2 == 0:
        payload["use_live_signals"] = True
        payload["signal_window_days"] = 7
        payload["max_adjustment"] = 0.15
    else:
        payload["use_live_signals"] = False
        payload["demand_signal"] = 80 + i
        payload["supply_volume"] = 40 + i
    status, body = _post_json("/pricing/auto", payload)
    return payload, status, body


def main():
    # wait a moment for server to be ready
    time.sleep(1)

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

    out = "tests/pricing_real_model_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote:", out)


if __name__ == "__main__":
    main()
