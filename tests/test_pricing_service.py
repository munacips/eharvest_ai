import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import pricing as pricing_service


def _iso_days_ago(days: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days)).isoformat()


def test_compute_price_pressure_stays_bounded():
    neutral = pricing_service.compute_price_pressure(None, None)
    demand_heavy = pricing_service.compute_price_pressure(100.0, 0.0)
    supply_heavy = pricing_service.compute_price_pressure(0.0, 100.0)

    assert neutral == 0.0
    assert 0.0 < demand_heavy <= 1.0
    assert -1.0 <= supply_heavy < 0.0


def test_resolve_platform_signals_filters_by_window_status_and_commodity(monkeypatch):
    recent_date = _iso_days_ago(2)
    stale_date = _iso_days_ago(45)

    async def fake_fetch_platform_list(path, params=None):
        if path == "/api/v1/order_items":
            return (
                [
                    {"commodity": "Maize", "quantity": 5, "created_at": recent_date},
                    {"commodity": "Maize", "quantity": 9, "created_at": stale_date},
                    {"commodity": "Beans", "quantity": 4, "created_at": recent_date},
                ],
                None,
            )
        if path == "/api/v1/orders":
            return ([], None)
        if path == "/api/v1/produce":
            return (
                [
                    {"commodity": "Maize", "quantity": 7, "status": "active", "created_at": recent_date},
                    {"commodity": "Maize", "quantity": 3, "status": "sold", "created_at": recent_date},
                    {"commodity": "Maize", "quantity": 8, "status": "active", "created_at": stale_date},
                ],
                None,
            )
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(pricing_service, "fetch_platform_list", fake_fetch_platform_list)

    signals, warnings = asyncio.run(pricing_service.resolve_platform_signals("maize", 30))

    assert warnings == []
    assert signals == {
        "demand_count": 1.0,
        "demand_qty": 5.0,
        "supply_count": 1.0,
        "supply_qty": 7.0,
    }


def test_resolve_platform_signals_falls_back_to_orders_and_deduplicates_warnings(monkeypatch):
    recent_date = _iso_days_ago(1)
    stale_date = _iso_days_ago(40)

    async def fake_fetch_platform_list(path, params=None):
        warning = "platform_api_warning"
        if path == "/api/v1/order_items":
            return ([{"commodity": "Beans", "quantity": 2, "created_at": recent_date}], warning)
        if path == "/api/v1/orders":
            return (
                [
                    {
                        "created_at": recent_date,
                        "items": [
                            {"commodity": "Maize", "quantity": 4},
                            {"commodity": "Beans", "quantity": 1},
                        ],
                    },
                    {"commodity": "Maize", "quantity": 6, "created_at": recent_date},
                    {"commodity": "Maize", "quantity": 20, "created_at": stale_date},
                    {"items": [{"commodity": "Tomatoes", "quantity": 3}], "created_at": recent_date},
                ],
                warning,
            )
        if path == "/api/v1/produce":
            return ([], warning)
        raise AssertionError(f"Unexpected path: {path}")

    monkeypatch.setattr(pricing_service, "fetch_platform_list", fake_fetch_platform_list)

    signals, warnings = asyncio.run(pricing_service.resolve_platform_signals("maize", 30))

    assert signals["demand_count"] == 2.0
    assert signals["demand_qty"] == 10.0
    assert signals["supply_count"] == 0.0
    assert signals["supply_qty"] == 0.0
    assert warnings == ["platform_api_warning"]
