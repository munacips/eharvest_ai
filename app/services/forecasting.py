from datetime import UTC, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.common import clamp, safe_mean, season_from_month

MARKET_SIGNAL_STALE_AFTER_DAYS = 45


def apply_region_filter(df: pd.DataFrame, region: str) -> pd.DataFrame:
    """Filter records by region while failing open when region data is unavailable.

    If no `region` column exists, or filtering yields no rows, the original data is
    returned so upstream forecasting can still proceed.
    """
    if df is None or df.empty or "region" not in df.columns:
        return df
    if not isinstance(region, str) or not region.strip():
        return df.copy()
    filtered = df[df["region"].fillna("").str.lower() == region.lower()].copy()
    return filtered if not filtered.empty else df.copy()


def compute_weather_impact(weather_df: Optional[pd.DataFrame], season: Optional[str]) -> float:
    """Estimate weather impact as a bounded proportional adjustment.

    Positive impact implies better-than-average growing conditions and negative
    impact implies worse-than-average conditions.
    """
    if weather_df is None or weather_df.empty:
        return 0.0
    working_df = weather_df.copy()

    # Enrich with season labels only when valid dates are available.
    if "season" not in working_df.columns:
        working_df["season"] = None
    if "date" in weather_df.columns:
        parsed_dates = pd.to_datetime(working_df["date"], errors="coerce")
        working_df["date"] = parsed_dates
        valid_dates = parsed_dates.notna()
        working_df.loc[valid_dates, "season"] = working_df.loc[valid_dates, "date"].dt.month.apply(
            season_from_month)

    seasonal = working_df
    if season:
        season_key = str(season).strip().lower()
        seasonal = working_df[working_df["season"].fillna(
            "").str.lower() == season_key]
        if seasonal.empty:
            seasonal = working_df

    rain_seasonal = pd.to_numeric(seasonal.get("rainfall_mm"), errors="coerce")
    rain_overall = pd.to_numeric(
        working_df.get("rainfall_mm"), errors="coerce")
    temp_seasonal = pd.to_numeric(
        seasonal.get("temperature_c"), errors="coerce")
    temp_overall = pd.to_numeric(
        working_df.get("temperature_c"), errors="coerce")

    rain_avg = safe_mean(rain_seasonal)
    rain_all = safe_mean(rain_overall)
    temp_avg = safe_mean(temp_seasonal)
    temp_all = safe_mean(temp_overall)

    rain_factor = 0.0 if rain_all == 0 else (rain_avg - rain_all) / rain_all
    temp_factor = 0.0 if temp_all == 0 else (temp_avg - temp_all) / temp_all

    return clamp(0.4 * rain_factor + 0.2 * temp_factor, -0.3, 0.3)


def summarize_market_signal(market_df: Optional[pd.DataFrame], commodity: str) -> Dict[str, Any]:
    """Summarize market signal freshness and trend for a commodity.

    Rising prices reduce demand-side projections (negative impact), while falling
    prices can mildly support demand (positive impact). Stale inputs are surfaced
    explicitly so callers do not mistake old static data for a live trend.
    """
    summary: Dict[str, Any] = {
        "commodity": commodity,
        "impact": 0.0,
        "price_trend": 0.0,
        "latest_observation_date": None,
        "recent_observation_count": 0,
        "is_stale": False,
        "warning": None,
    }
    if market_df is None or market_df.empty:
        summary["warning"] = "market_data_unavailable"
        return summary
    if "commodity" not in market_df.columns or "price" not in market_df.columns or "date" not in market_df.columns:
        summary["warning"] = "market_data_missing_required_columns"
        return summary
    if not isinstance(commodity, str) or not commodity.strip():
        summary["warning"] = "market_commodity_not_provided"
        return summary

    commodity_df = market_df[
        market_df["commodity"].fillna("").astype(
            str).str.lower() == commodity.lower()
    ].copy()
    if commodity_df.empty:
        summary["warning"] = "market_commodity_not_found"
        return summary

    commodity_df["date"] = pd.to_datetime(
        commodity_df["date"], errors="coerce", utc=True)
    commodity_df["price"] = pd.to_numeric(
        commodity_df["price"], errors="coerce")
    commodity_df = commodity_df.dropna(subset=["date", "price"])
    if commodity_df.empty:
        summary["warning"] = "market_commodity_has_no_valid_rows"
        return summary

    overall = safe_mean(commodity_df["price"])
    latest_observation = commodity_df["date"].max()
    if pd.isna(latest_observation):
        summary["warning"] = "market_commodity_has_no_valid_rows"
        return summary

    summary["latest_observation_date"] = latest_observation.strftime("%Y-%m-%d")
    age_days = max(0, (datetime.now(UTC) - latest_observation.to_pydatetime()).days)
    if age_days > MARKET_SIGNAL_STALE_AFTER_DAYS:
        summary["is_stale"] = True
        summary["warning"] = f"market_data_stale:{age_days}d_old"
        return summary

    recent_cutoff = latest_observation - timedelta(days=90)
    recent = commodity_df[commodity_df["date"] >= recent_cutoff]
    summary["recent_observation_count"] = int(len(recent))
    recent_avg = safe_mean(recent["price"])
    if overall == 0:
        summary["warning"] = "market_price_average_is_zero"
        return summary
    price_trend = (recent_avg - overall) / overall
    summary["price_trend"] = float(price_trend)
    summary["impact"] = float(clamp(-0.2 * price_trend, -0.2, 0.2))
    return summary


def compute_market_impact(market_df: Optional[pd.DataFrame], commodity: str) -> float:
    return float(summarize_market_signal(market_df, commodity)["impact"])


def forecast_monthly_series(
    monthly_df: pd.DataFrame,
    periods: int,
    weather_impact: float,
    market_impact: float,
) -> List[Dict[str, Any]]:
    """Forecast future monthly quantities using seasonal profile + linear trend.

    The forecast baseline is a month-of-year seasonal mean adjusted by a simple
    linear trend, then scaled by exogenous weather/market impacts.
    """
    if monthly_df.empty:
        return []
    if "year_month" not in monthly_df.columns or "quantity" not in monthly_df.columns:
        return []

    monthly_df = monthly_df.copy()
    monthly_df["year_month"] = pd.to_datetime(
        monthly_df["year_month"], errors="coerce")
    monthly_df["quantity"] = pd.to_numeric(
        monthly_df["quantity"], errors="coerce")
    monthly_df = monthly_df.dropna(subset=["year_month", "quantity"])
    if monthly_df.empty:
        return []

    monthly_df = monthly_df.sort_values("year_month")
    overall_avg = safe_mean(monthly_df["quantity"])
    monthly_df["month"] = monthly_df["year_month"].dt.month
    seasonal_means = monthly_df.groupby("month")["quantity"].mean()

    if len(monthly_df) >= 2:
        x = np.arange(len(monthly_df))
        y = monthly_df["quantity"].values
        if np.allclose(y, y[0]):
            slope = 0.0
        else:
            slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    last_month = monthly_df["year_month"].max()
    results: List[Dict[str, Any]] = []
    impact_multiplier = max(0.0, 1 + weather_impact + market_impact)
    for i in range(1, periods + 1):
        future_month = last_month + pd.DateOffset(months=i)
        seasonal_base = seasonal_means.get(future_month.month, overall_avg)
        baseline = seasonal_base + slope * i
        adjusted = baseline * impact_multiplier
        results.append({
            "date": future_month.strftime("%Y-%m-%d"),
            "value": max(0.0, float(adjusted)),
            "season": season_from_month(future_month.month),
        })
    return results
