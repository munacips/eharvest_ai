from datetime import UTC, datetime
from typing import Any, Dict

import httpx
from fastapi import APIRouter, Response, status

from app import config, state
from app.services.integrations import platform_headers

router = APIRouter()


def _health_timeout_seconds() -> float:
    return max(0.5, min(config.PLATFORM_API_TIMEOUT, 2.0))


def _models_health_check() -> Dict[str, Any]:
    issues = []

    if getattr(state, "dynamic_pricing_model", None) is None:
        issues.append("dynamic_pricing_model_not_loaded")
    if not getattr(state, "model_columns", None):
        issues.append("model_columns_not_loaded")
    if getattr(state, "forecast_model", None) is None:
        issues.append("forecast_model_not_loaded")
    if not getattr(state, "commodity_cols", None):
        issues.append("forecast_features_not_loaded")

    if issues:
        return {"status": "error", "issues": issues}
    return {"status": "ok"}


async def _probe_http_endpoint(
    url: str,
    *,
    headers: Dict[str, str] | None = None,
    require_success_status: bool = True,
) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(
            timeout=_health_timeout_seconds(),
            follow_redirects=True,
        ) as client:
            response = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        return {"status": "error", "url": url, "detail": "timeout"}
    except httpx.RequestError as exc:
        return {
            "status": "error",
            "url": url,
            "detail": exc.__class__.__name__,
        }

    status_code = response.status_code
    if require_success_status:
        if status_code in (200, 204):
            return {"status": "ok", "url": url, "status_code": status_code}
        return {
            "status": "error",
            "url": url,
            "status_code": status_code,
            "detail": "unexpected_status_code",
        }

    if status_code < 500:
        return {"status": "ok", "url": url, "status_code": status_code}
    return {
        "status": "error",
        "url": url,
        "status_code": status_code,
        "detail": "upstream_server_error",
    }


async def _platform_api_health_check() -> Dict[str, Any]:
    if not config.PLATFORM_API_BASE_URL:
        return {"status": "error", "detail": "platform_api_base_url_not_set"}

    return await _probe_http_endpoint(
        f"{config.PLATFORM_API_BASE_URL}/api/v1/produce",
        headers=platform_headers(),
        require_success_status=True,
    )


async def _review_service_health_check() -> Dict[str, Any]:
    if config.USE_REVIEW_PLACEHOLDER:
        return {
            "status": "degraded",
            "detail": "trust_placeholder_enabled",
        }
    if not config.SPRING_BOOT_BASE_URL:
        return {"status": "error", "detail": "spring_boot_base_url_not_set"}

    return await _probe_http_endpoint(
        config.SPRING_BOOT_BASE_URL,
        require_success_status=False,
    )


def _combine_health_status(checks: Dict[str, Dict[str, Any]]) -> str:
    statuses = {check["status"] for check in checks.values()}
    if "error" in statuses:
        return "error"
    if "degraded" in statuses:
        return "degraded"
    return "ok"


@router.get("/health")
async def health_check(response: Response):
    checks = {
        "models": _models_health_check(),
        "platform_api": await _platform_api_health_check(),
        "review_service": await _review_service_health_check(),
    }
    overall_status = _combine_health_status(checks)
    if overall_status != "ok":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": overall_status,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checks": checks,
    }


@router.get("/")
def read_root():
    return {
        "message": "API is running. Use /predict-price for price predictions and /forecast/commodity for demand forecasts."
    }
