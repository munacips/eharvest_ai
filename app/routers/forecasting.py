from typing import Optional

from fastapi import APIRouter, HTTPException

from app.models import DemandSupplyForecastRequest
from app.services import forecasting as forecasting_service

router = APIRouter()


@router.get("/forecast/{commodity}")
async def get_forecast(
    commodity: str,
    periods: int = 30,
    region: Optional[str] = None,
    visual: bool = True,
    ):
    if periods <= 0:
        raise HTTPException(
            status_code=400, detail="periods must be greater than 0")
    try:
        return forecasting_service.build_price_forecast_response(
            commodity,
            region=region,
            periods=periods,
            visual=visual,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/forecast/demand-supply")
async def demand_supply_forecast(request: DemandSupplyForecastRequest):
    if request.periods <= 0:
        raise HTTPException(
            status_code=400, detail="periods must be greater than 0")

    response = forecasting_service.build_demand_supply_response(
        request.commodities,
        region=request.region,
        periods=request.periods,
    )
    if not response["forecasts"]:
        raise HTTPException(status_code=404, detail="no forecasts generated")
    return response
