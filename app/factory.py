from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import allow_credentials, cors_origins

from app.routers import (
    forecasting_router,
    integrations_router,
    logistics_router,
    pricing_router,
    recommendations_router,
    system_router,
    trust_router,
)


def create_app() -> FastAPI:
    app = FastAPI(title="eHarvest AI API",
                  description="API for eHarvest AI services", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(pricing_router)
    app.include_router(forecasting_router)
    app.include_router(recommendations_router)
    app.include_router(logistics_router)
    app.include_router(integrations_router)
    app.include_router(trust_router)
    app.include_router(system_router)

    return app


app = create_app()
