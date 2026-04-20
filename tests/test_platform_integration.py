# How to run:
#   RUN_PLATFORM_TESTS=1 PLATFORM_API_BASE_URL=http://localhost:8080 PLATFORM_API_KEY=eharvest-ai-secret-key-12345 python -m pytest -q

import os

import httpx
import pytest


def _platform_base_url() -> str:
    return os.getenv("PLATFORM_API_BASE_URL", "http://localhost:8080").rstrip("/")


def _platform_headers() -> dict:
    api_key = os.getenv("PLATFORM_API_KEY", "eharvest-ai-secret-key-12345")
    return {"X-API-KEY": api_key} if api_key else {}


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/produce",
        "/api/v1/logistics-providers",
        "/api/v1/orders",
    ],
)
def test_platform_endpoints_reachable(path):
    if os.getenv("RUN_PLATFORM_TESTS") != "1":
        pytest.skip("Set RUN_PLATFORM_TESTS=1 to run live platform integration tests.")

    url = f"{_platform_base_url()}{path}"
    with httpx.Client(timeout=6.0) as client:
        resp = client.get(url, headers=_platform_headers())

    assert resp.status_code in (200, 204), f"{path} returned {resp.status_code}: {resp.text[:200]}"
