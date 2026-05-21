import json
import time
import urllib.request
import urllib.parse

BASE = "http://127.0.0.1:8001"

commodities = ["maize", "beans", "sorghum", "groundnuts", "rice"]
exact_commodities = [
    "Maize",
    "Maize meal",
    "Maize meal (white, fortified)",
    "Beans",
    "Groundnuts (shelled)",
]


def _get_json(path: str, params: dict | None = None, timeout=10):
    url = f"{BASE}{path}"
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.getcode(), json.load(resp)


def _post_json(path: str, data: dict, timeout=10):
    url = f"{BASE}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={
                                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), json.load(resp)


def call_forecasts():
    results = []
    for c in commodities:
        status, body = _get_json(f"/forecast/{urllib.parse.quote(c)}", {
                                 "periods": 5, "region": "Manicaland", "visual": "false"})
        results.append({"commodity": c, "status": status, "response": body})
    return results


def call_exact_forecasts():
    results = []
    for c in exact_commodities:
        status, body = _get_json(
            f"/forecast/{urllib.parse.quote(c)}",
            {"periods": 5, "region": "Manicaland", "visual": "false"},
        )
        results.append({"commodity": c, "status": status, "response": body})
    return results


def call_demand_supply():
    payload = {
        "region": "Manicaland",
        "season": "rainy",
        "periods": 3,
        "historical_sales": [
            {"date": "2024-01-01", "commodity": "maize",
                "quantity": 120, "region": "Manicaland"},
            {"date": "2024-02-01", "commodity": "maize",
                "quantity": 135, "region": "Manicaland"},
            {"date": "2024-01-01", "commodity": "beans",
                "quantity": 80, "region": "Manicaland"},
        ],
        "weather": [
            {"date": "2024-01-01", "rainfall_mm": 85,
                "temperature_c": 24, "region": "Manicaland"},
            {"date": "2024-02-01", "rainfall_mm": 90,
                "temperature_c": 25, "region": "Manicaland"},
        ],
        "market_data": [
            {"date": "2024-01-01", "commodity": "maize", "price": 0.35,
                "market": "Harare", "region": "Manicaland"},
            {"date": "2024-02-01", "commodity": "maize", "price": 0.38,
                "market": "Harare", "region": "Manicaland"},
        ],
    }
    return _post_json("/forecast/demand-supply", payload)


def main():
    time.sleep(1)
    out = {
        "forecasts": call_forecasts(),
        "exact_vocab_forecasts": call_exact_forecasts(),
    }
    status, ds = call_demand_supply()
    out["demand_supply"] = {"status": status, "response": ds}
    out_file = "tests/forecasting_real_model_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote:", out_file)


if __name__ == "__main__":
    main()
