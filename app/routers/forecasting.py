from datetime import UTC, datetime
from difflib import get_close_matches
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.models import DemandSupplyForecastRequest
from app.services.forecasting import (
    apply_region_filter,
    compute_weather_impact,
    forecast_monthly_series,
    summarize_market_signal,
)
from app.services.integrations import (
    fetch_external_market_prices,
    fetch_platform_market_prices,
    fetch_weather_open_meteo,
)
from app.state import commodity_cols, forecast_model

router = APIRouter()


def _resolve_forecast_feature_name(prefix: str, raw_value: Optional[str]) -> Optional[str]:
    if not raw_value:
        return None
    target = raw_value.strip()
    if not target:
        return None

    # allowed suffixes for this prefix (e.g., columns after 'commodity_')
    allowed = [col[len(prefix) + 1:]
               for col in commodity_cols if col.startswith(f"{prefix}_")]
    if not allowed:
        return None

    # exact case-insensitive match
    exact_lookup = {a.casefold(): a for a in allowed}
    if target.casefold() in exact_lookup:
        return f"{prefix}_{exact_lookup[target.casefold()]}"

    # helpful aliases for common short names -> full vocabulary labels
    FORECAST_ALIASES = {
        "maize": "Maize",
        "beans": "Beans",
        "groundnuts": "Groundnuts (shelled)",
        "rice": "Rice",
        "sorghum": "Sorghum",
        "millet": "Millet",
    }

    lowered = target.casefold()
    # try alias mapping first (commodity-specific)
    if prefix == "commodity" and lowered in FORECAST_ALIASES:
        candidate = FORECAST_ALIASES[lowered]
        if candidate.casefold() in exact_lookup:
            return f"{prefix}_{exact_lookup[candidate.casefold()]}"

    # substring unique match (e.g., 'maize meal' vs 'maize')
    substring_matches = [
        a for a in allowed if lowered in a.casefold() or a.casefold() in lowered]
    if len(substring_matches) == 1:
        return f"{prefix}_{substring_matches[0]}"

    # close match fallback
    close = get_close_matches(
        lowered, [a.casefold() for a in allowed], n=1, cutoff=0.6)
    if close:
        return f"{prefix}_{exact_lookup[close[0]]}"

    return None


@router.get("/forecast/{commodity}")
async def get_forecast(
    commodity: str,
    periods: int = 30,
    region: Optional[str] = None,
    visual: bool = True,
    ):
    future = forecast_model.make_future_dataframe(periods=periods)

    target_col = _resolve_forecast_feature_name("commodity", commodity)
    region_col = _resolve_forecast_feature_name("admin1", region)

    for col in commodity_cols:
        if col == target_col:
            future[col] = 1
        elif region_col and col == region_col:
            future[col] = 1
        else:
            future[col] = 0

    forecast = forecast_model.predict(future)

    result = forecast[["ds", "yhat"]].tail(periods)
    records = [
        {"date": row["ds"].strftime("%Y-%m-%d"),
         "value": max(0.0, float(row["yhat"]))}
        for _, row in result.iterrows()
    ]
    response = {"commodity": commodity, "region": region, "forecast": records}

    if visual:
        response["visual"] = {
            "type": "line",
            "x": [r["date"] for r in records],
            "series": [{"name": "Demand Forecast", "data": [r["value"] for r in records]}],
        }

    return response


@router.post("/forecast/demand-supply")
async def demand_supply_forecast(request: DemandSupplyForecastRequest):
    if request.periods <= 0:
        raise HTTPException(
            status_code=400, detail="periods must be greater than 0")

    warnings: List[str] = []
    sources: List[str] = []

    sales_df = pd.DataFrame([item.model_dump()
                            for item in request.historical_sales])
    if sales_df.empty:
        raise HTTPException(
            status_code=400, detail="historical_sales cannot be empty")
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    sales_df = apply_region_filter(sales_df, request.region)

    if request.historical_supply:
        supply_df = pd.DataFrame([item.model_dump()
                                 for item in request.historical_supply])
        supply_df["date"] = pd.to_datetime(supply_df["date"])
        supply_df = apply_region_filter(supply_df, request.region)
    else:
        supply_df = pd.DataFrame()

    weather_df = None
    if request.weather:
        weather_df = pd.DataFrame([item.model_dump()
                                  for item in request.weather])
        weather_df = apply_region_filter(weather_df, request.region)

    market_df = None
    if request.market_data:
        market_df = pd.DataFrame([item.model_dump()
                                 for item in request.market_data])
        market_df = apply_region_filter(market_df, request.region)

    if request.auto_fetch_external:
        if (weather_df is None or weather_df.empty) and request.latitude is not None and request.longitude is not None:
            weather_payload, weather_source, warning = await fetch_weather_open_meteo(
                request.latitude, request.longitude, days=max(
                    request.periods, 7)
            )
            if weather_payload:
                weather_df = pd.DataFrame(weather_payload)
                sources.append(weather_source)
            if warning:
                warnings.append(warning)

        if market_df is None or market_df.empty:
            market_payload, market_source, warning = await fetch_platform_market_prices(request.region)
            if market_payload:
                market_df = pd.DataFrame(market_payload)
                sources.append(market_source)
            if warning:
                warnings.append(warning)

            if market_df is None or market_df.empty:
                external_payload, external_source, warning = await fetch_external_market_prices(request.region)
                if external_payload:
                    market_df = pd.DataFrame(external_payload)
                    sources.append(external_source)
                if warning:
                    warnings.append(warning)

    commodities = request.commodities or sorted(
        sales_df["commodity"].dropna().unique().tolist())
    if not commodities:
        raise HTTPException(
            status_code=400, detail="no commodities found in historical_sales")

    weather_impact = compute_weather_impact(weather_df, request.season)
    forecasts = []
    visual_series = []
    market_signals = []

    for commodity in commodities:
        commodity_sales = sales_df[sales_df["commodity"].str.lower(
        ) == commodity.lower()].copy()
        if commodity_sales.empty:
            continue
        commodity_sales["year_month"] = commodity_sales["date"].dt.to_period(
            "M").dt.to_timestamp()
        monthly_sales = commodity_sales.groupby(
            "year_month")["quantity"].sum().reset_index()
        market_signal = summarize_market_signal(market_df, commodity)
        demand_market_impact = float(market_signal["impact"])
        market_signals.append(market_signal)
        if market_signal.get("warning") and market_signal["warning"] not in warnings:
            warnings.append(market_signal["warning"])
        demand_forecast = forecast_monthly_series(
            monthly_sales,
            request.periods,
            weather_impact,
            demand_market_impact,
        )

        if not supply_df.empty:
            commodity_supply = supply_df[supply_df["commodity"].str.lower(
            ) == commodity.lower()].copy()
            commodity_supply["year_month"] = commodity_supply["date"].dt.to_period(
                "M").dt.to_timestamp()
            monthly_supply = commodity_supply.groupby(
                "year_month")["quantity"].sum().reset_index()
            supply_forecast = forecast_monthly_series(
                monthly_supply,
                request.periods,
                weather_impact,
                0.0,
            )
        else:
            supply_forecast = [
                {
                    "date": point["date"],
                    "value": max(0.0, point["value"] * request.supply_multiplier),
                    "season": point["season"],
                }
                for point in demand_forecast
            ]

        forecasts.append({
            "commodity": commodity,
            "demand": demand_forecast,
            "supply": supply_forecast,
        })

        visual_series.append({
            "name": f"{commodity} demand",
            "data": [point["value"] for point in demand_forecast],
            "x": [point["date"] for point in demand_forecast],
        })

    if not forecasts:
        raise HTTPException(
            status_code=404, detail="no forecasts generated for provided commodities")

    return {
        "region": request.region,
        "season": request.season,
        "periods": request.periods,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecasts": forecasts,
        "visual": {
            "type": "line",
            "series": visual_series,
            "notes": "visual data only; frontend renders charts",
        },
        "market_signals": market_signals,
        "weather_impact": weather_impact,
        "sources": sources,
        "warnings": warnings,
        "assumptions": [
            "seasonal averages and recent trend are used for forecasting",
            "weather and market impacts are applied as bounded adjustments",
            "supply forecast uses provided supply data or demand * supply_multiplier",
        ],
    }
