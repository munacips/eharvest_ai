from datetime import UTC, datetime
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from app.models import DemandSupplyForecastRequest
from app.services.forecasting import (
    apply_region_filter,
    compute_market_impact,
    compute_weather_impact,
    forecast_monthly_series,
)
from app.services.integrations import (
    fetch_external_market_prices,
    fetch_platform_market_prices,
    fetch_weather_open_meteo,
)
from app.state import commodity_cols, forecast_model

router = APIRouter()


@router.get("/forecast/{commodity}")
async def get_forecast(
    commodity: str,
    periods: int = 30,
    region: Optional[str] = None,
    visual: bool = True,
):
    future = forecast_model.make_future_dataframe(periods=periods)

    target_col = f"commodity_{commodity}"
    region_col = f"admin1_{region}" if region else None

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
        {"date": row["ds"].strftime("%Y-%m-%d"), "value": float(row["yhat"])}
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

    for commodity in commodities:
        commodity_sales = sales_df[sales_df["commodity"].str.lower(
        ) == commodity.lower()].copy()
        if commodity_sales.empty:
            continue
        commodity_sales["year_month"] = commodity_sales["date"].dt.to_period(
            "M").dt.to_timestamp()
        monthly_sales = commodity_sales.groupby(
            "year_month")["quantity"].sum().reset_index()
        demand_market_impact = compute_market_impact(market_df, commodity)
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
        "sources": sources,
        "warnings": warnings,
        "assumptions": [
            "seasonal averages and recent trend are used for forecasting",
            "weather and market impacts are applied as bounded adjustments",
            "supply forecast uses provided supply data or demand * supply_multiplier",
        ],
    }
