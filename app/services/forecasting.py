from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.services.common import clamp, safe_mean, season_from_month


def apply_region_filter(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if df is None or df.empty or "region" not in df.columns:
        return df
    filtered = df[df["region"].fillna("").str.lower() == region.lower()].copy()
    return filtered if not filtered.empty else df.copy()


def compute_weather_impact(weather_df: Optional[pd.DataFrame], season: Optional[str]) -> float:
    if weather_df is None or weather_df.empty:
        return 0.0
    if "date" in weather_df.columns:
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        weather_df["season"] = weather_df["date"].dt.month.apply(
            season_from_month)

    seasonal = weather_df
    if season:
        seasonal = weather_df[weather_df["season"] == season]
        if seasonal.empty:
            seasonal = weather_df

    rain_avg = safe_mean(seasonal.get("rainfall_mm"))
    rain_all = safe_mean(weather_df.get("rainfall_mm"))
    temp_avg = safe_mean(seasonal.get("temperature_c"))
    temp_all = safe_mean(weather_df.get("temperature_c"))

    rain_factor = 0.0 if rain_all == 0 else (rain_avg - rain_all) / rain_all
    temp_factor = 0.0 if temp_all == 0 else (temp_avg - temp_all) / temp_all

    return clamp(0.4 * rain_factor + 0.2 * temp_factor, -0.3, 0.3)


def compute_market_impact(market_df: Optional[pd.DataFrame], commodity: str) -> float:
    if market_df is None or market_df.empty:
        return 0.0
    commodity_df = market_df[market_df["commodity"].str.lower(
    ) == commodity.lower()].copy()
    if commodity_df.empty:
        return 0.0
    commodity_df["date"] = pd.to_datetime(commodity_df["date"])
    overall = safe_mean(commodity_df["price"])
    recent_cutoff = commodity_df["date"].max() - timedelta(days=90)
    recent = commodity_df[commodity_df["date"] >= recent_cutoff]
    recent_avg = safe_mean(recent["price"])
    if overall == 0:
        return 0.0
    price_trend = (recent_avg - overall) / overall
    return clamp(-0.2 * price_trend, -0.2, 0.2)


def forecast_monthly_series(
    monthly_df: pd.DataFrame,
    periods: int,
    weather_impact: float,
    market_impact: float,
) -> List[Dict[str, Any]]:
    if monthly_df.empty:
        return []

    monthly_df = monthly_df.sort_values("year_month")
    overall_avg = safe_mean(monthly_df["quantity"])
    monthly_df["month"] = monthly_df["year_month"].dt.month
    seasonal_means = monthly_df.groupby("month")["quantity"].mean()

    if len(monthly_df) >= 2:
        x = np.arange(len(monthly_df))
        y = monthly_df["quantity"].values
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    last_month = monthly_df["year_month"].max()
    results: List[Dict[str, Any]] = []
    for i in range(1, periods + 1):
        future_month = last_month + pd.DateOffset(months=i)
        seasonal_base = seasonal_means.get(future_month.month, overall_avg)
        baseline = seasonal_base + slope * i
        adjusted = baseline * (1 + weather_impact + market_impact)
        results.append({
            "date": future_month.strftime("%Y-%m-%d"),
            "value": max(0.0, float(adjusted)),
            "season": season_from_month(future_month.month),
        })
    return results
