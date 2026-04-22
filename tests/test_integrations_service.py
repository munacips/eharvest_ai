import asyncio

import httpx

from app.services import integrations as integrations_service


def _install_fake_async_client(monkeypatch, response_factory):
    class DummyAsyncClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, params=None, headers=None):
            return response_factory(url, params, headers)

    monkeypatch.setattr(integrations_service.httpx, "AsyncClient", DummyAsyncClient)


def test_fetch_platform_list_handles_no_content(monkeypatch):
    def response_factory(url, params, headers):
        request = httpx.Request("GET", url, params=params, headers=headers)
        return httpx.Response(204, request=request)

    _install_fake_async_client(monkeypatch, response_factory)

    items, warning = asyncio.run(
        integrations_service.fetch_platform_list("/api/v1/produce")
    )

    assert items == []
    assert warning is None


def test_fetch_platform_market_prices_filters_inactive_and_region(monkeypatch):
    async def fake_fetch_platform_list(path, params=None):
        return (
            [
                {
                    "commodity": "Maize Grain",
                    "price": 0.41,
                    "market": "Mutare",
                    "region": "Manicaland",
                    "status": "active",
                    "date": "2026-03-02",
                },
                {
                    "commodity": "Maize Grain",
                    "price": 0.55,
                    "market": "Mutare",
                    "region": "Manicaland",
                    "status": "sold",
                    "date": "2026-03-03",
                },
                {
                    "commodity": "Maize Meal",
                    "price": 0.39,
                    "market": "Manicaland Central",
                    "status": "active",
                    "date": "2026-03-01",
                },
                {
                    "commodity": "Beans",
                    "price": 0.62,
                    "market": "Mutare",
                    "region": "Manicaland",
                    "status": "active",
                    "date": "2026-03-04",
                },
                {
                    "commodity": "Maize Grain",
                    "price": 0.49,
                    "market": "Chinhoyi",
                    "region": "Mashonaland West",
                    "status": "active",
                    "date": "2026-03-05",
                },
            ],
            None,
        )

    monkeypatch.setattr(
        integrations_service, "fetch_platform_list", fake_fetch_platform_list
    )

    results, source, warning = asyncio.run(
        integrations_service.fetch_platform_market_prices("Manicaland", "maize")
    )

    assert source == "platform_produce"
    assert warning is None
    assert [item["commodity"] for item in results] == ["Maize Grain", "Maize Meal"]
    assert [item["date"] for item in results] == ["2026-03-02", "2026-03-01"]


def test_fetch_external_market_prices_filters_api_payload(monkeypatch):
    captured = {}

    def response_factory(url, params, headers):
        captured["url"] = url
        captured["params"] = params
        request = httpx.Request("GET", url, params=params, headers=headers)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "commodity": "Maize",
                        "price": 0.44,
                        "market": "Mutare",
                        "region": "Manicaland",
                        "date": "2026-03-02",
                    },
                    {
                        "commodity": "Beans",
                        "price": 0.66,
                        "market": "Mutare",
                        "region": "Manicaland",
                        "date": "2026-03-03",
                    },
                    {
                        "commodity": "Maize",
                        "price": 0.47,
                        "market": "Chinhoyi",
                        "region": "Mashonaland West",
                        "date": "2026-03-04",
                    },
                    {
                        "commodity": "Maize",
                        "market": "Mutare",
                        "region": "Manicaland",
                        "date": "2026-03-05",
                    },
                ]
            },
            request=request,
        )

    monkeypatch.setattr(
        integrations_service.config,
        "MARKET_DATA_API_URL",
        "https://example.com/market-data",
    )
    _install_fake_async_client(monkeypatch, response_factory)

    results, source, warning = asyncio.run(
        integrations_service.fetch_external_market_prices("Manicaland", "maize")
    )

    assert captured["url"] == "https://example.com/market-data"
    assert captured["params"] == {"region": "Manicaland", "commodity": "maize"}
    assert source == "external_market_api"
    assert warning is None
    assert results == [
        {
            "date": "2026-03-02",
            "commodity": "Maize",
            "price": 0.44,
            "market": "Mutare",
            "region": "Manicaland",
        }
    ]


def test_fetch_weather_open_meteo_rejects_invalid_coordinates():
    weather, source, warning = asyncio.run(
        integrations_service.fetch_weather_open_meteo(120.0, 31.0, days=7)
    )

    assert weather == []
    assert source == "open_meteo"
    assert warning == "weather_api_invalid_coordinates"
