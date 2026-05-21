import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services import forecasting as forecasting_service


def test_summarize_market_signal_returns_recent_impact():
    now = pd.Timestamp.now(tz="UTC").normalize()
    market_df = pd.DataFrame(
        [
            {"date": (now - pd.Timedelta(days=150)).strftime("%Y-%m-%d"), "commodity": "Maize", "price": 0.30},
            {"date": (now - pd.Timedelta(days=20)).strftime("%Y-%m-%d"), "commodity": "Maize", "price": 0.36},
            {"date": (now - pd.Timedelta(days=5)).strftime("%Y-%m-%d"), "commodity": "Maize", "price": 0.39},
        ]
    )

    summary = forecasting_service.summarize_market_signal(market_df, "maize")

    assert summary["commodity"] == "maize"
    assert summary["is_stale"] is False
    assert summary["warning"] is None
    assert summary["recent_observation_count"] == 2
    assert summary["impact"] < 0.0


def test_summarize_market_signal_zeroes_stale_data():
    market_df = pd.DataFrame(
        [
            {"date": "2024-01-01", "commodity": "Maize", "price": 0.30},
            {"date": "2024-02-01", "commodity": "Maize", "price": 0.34},
        ]
    )

    summary = forecasting_service.summarize_market_signal(market_df, "maize")

    assert summary["impact"] == 0.0
    assert summary["price_trend"] == 0.0
    assert summary["is_stale"] is True
    assert summary["warning"].startswith("market_data_stale:")
