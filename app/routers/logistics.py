from fastapi import APIRouter, HTTPException

from app.models import LogisticsMatchRequest
from app.services.integrations import fetch_platform_list, platform_get
from app.services.logistics import score_logistics_candidates

router = APIRouter()


@router.post("/logistics/match")
async def logistics_match(request: LogisticsMatchRequest):
    warnings = []
    sources = []

    logistics_request = request.logistics_request
    if logistics_request is None and request.request_id:
        data, warning = await platform_get(f"/api/v1/logistics/{request.request_id}")
        if data is not None:
            logistics_request = data if isinstance(data, dict) else None
            sources.append("platform_logistics")
        if warning:
            warnings.append(warning)

    providers = request.providers
    if not providers:
        providers, warning = await fetch_platform_list("/api/v1/logistics-providers")
        if warning:
            warnings.append(warning)
        if providers:
            sources.append("platform_logistics_providers")

    if not logistics_request:
        raise HTTPException(
            status_code=400, detail="logistics_request or request_id is required")
    if not providers:
        raise HTTPException(
            status_code=400, detail="no logistics providers available for matching")

    result = score_logistics_candidates(
        logistics_request,
        providers,
        weights=request.weights,
        max_distance_km=request.max_distance_km,
    )
    result["matches"] = result["matches"][: max(1, request.top_n)]

    return {
        "request": logistics_request,
        "route_distance_km": result["route_distance_km"],
        "matches": result["matches"],
        "rejected": result["rejected"],
        "sources": sources,
        "warnings": warnings,
    }
