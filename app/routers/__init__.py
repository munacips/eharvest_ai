from app.routers.forecasting import router as forecasting_router
from app.routers.integrations import router as integrations_router
from app.routers.logistics import router as logistics_router
from app.routers.pricing import router as pricing_router
from app.routers.recommendations import router as recommendations_router
from app.routers.system import router as system_router
from app.routers.trust import router as trust_router

__all__ = [
    "forecasting_router",
    "integrations_router",
    "logistics_router",
    "pricing_router",
    "recommendations_router",
    "system_router",
    "trust_router",
]
