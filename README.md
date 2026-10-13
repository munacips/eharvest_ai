# eHarvest AI

eHarvest AI is a FastAPI service for agricultural pricing, demand and supply forecasting, crop recommendations, logistics matching, live market and weather data, and buyer trust scoring.

## Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Architecture and Patterns](#architecture-and-patterns)
- [Data Models](#data-models)
- [Services and API Layer](#services-and-api-layer)
- [Navigation and Routing](#navigation-and-routing)
- [Admin Panel](#admin-panel)
- [Configuration](#configuration)
- [Getting Started](#getting-started)
- [Backend Requirements](#backend-requirements)
- [Deployment](#deployment)
- [Model Training](#model-training)
- [Testing](#testing)

## Features

### Pricing and market intelligence

- Predict a suggested commodity price from market, commodity, geographic, category, unit, currency, and source-data fields.
- Generate multiple price predictions in one request.
- Inspect the pricing model schema and its required, optional, and generated feature columns.
- Apply live platform demand and supply signals to a base price with a bounded adjustment.
- Fetch normalized market prices from platform produce listings, with an optional external market-data fallback.

### Forecasting

- Generate per-commodity price forecasts, optionally filtered by region.
- Generate demand and supply projections for selected commodities and regions.
- Return visual forecast data when requested by the price-forecast endpoint.

### Farm planning and recommendations

- Recommend crops using region, season or month, budget, climate, demand forecasts, and market data.
- Infer Zimbabwe seasonality, regional coordinates, and preset climate values for supported regions.
- Fetch Open-Meteo weather data when a recommendation request needs external climate inputs.

### Logistics

- Match a logistics request to available providers.
- Obtain a request by ID and provider data from the platform when those values are not supplied in the request.
- Rank candidates using cost, route distance, and capacity weights, with optional maximum-distance filtering.

### Trust and system operations

- Compute a user trust score from review data supplied by the Spring Boot reviews service.
- Expose a root endpoint with a short service description.
- Report model, platform API, and review-service health from `/health`.
- Serve interactive OpenAPI documentation through FastAPI at `/docs`.

## Tech Stack

Versions below are the locked versions resolved in `uv.lock` where available. Minimum constraints are declared in `pyproject.toml`.

| Concern | Package/Tool | Version |
| --- | --- | --- |
| Runtime | Python | 3.12 (`.python-version`; `requires-python >=3.12`) |
| API framework | FastAPI | 0.129.0 |
| ASGI server | Uvicorn | 0.41.0 |
| HTTP client | HTTPX | 0.28.1 |
| Configuration | python-dotenv | 1.2.2 |
| Data frames | pandas | 3.0.1 |
| Numerical computing | NumPy | 2.4.2 |
| Machine learning | scikit-learn | 1.8.0 |
| Time-series forecasting | Prophet | 1.3.0 |
| Model serialization | joblib | 1.5.3 |
| Sentiment analysis support | NLTK | 3.9.3 |
| Plotting and training utilities | Matplotlib | 3.10.8 |
| Test runner | pytest | 9.0.2 |
| Dependency management | uv | Uses `pyproject.toml` and `uv.lock` |
| Container runtime | Docker | Python 3.12 slim image |
| CI/CD | GitHub Actions | Docker Hub build/push and SSH deployment |

## Project Structure

```text
.
├── main.py                                      # Exposes the FastAPI application from the factory.
├── pyproject.toml                               # Project metadata, Python requirement, and dependencies.
├── uv.lock                                      # Reproducible dependency resolution.
├── .python-version                              # Selects Python 3.12.
├── .env                                         # Local environment values; do not commit secrets.
├── Dockerfile                                   # Builds and runs the API container on port 8000.
├── app/
│   ├── __init__.py                              # Application package marker.
│   ├── config.py                                # Environment settings and Zimbabwe crop/region constants.
│   ├── factory.py                               # FastAPI creation, CORS middleware, and router registration.
│   ├── models.py                                # Pydantic request models and OpenAPI examples.
│   ├── state.py                                 # Loads serialized pricing and forecasting models at import time.
│   ├── routers/
│   │   ├── forecasting.py                       # Price and demand/supply forecast routes.
│   │   ├── integrations.py                      # Weather and market-price routes.
│   │   ├── logistics.py                         # Logistics-provider matching route.
│   │   ├── pricing.py                           # Single, batch, schema, and live-signal pricing routes.
│   │   ├── recommendations.py                   # Prescriptive crop recommendation route.
│   │   ├── system.py                            # Root and dependency-aware health routes.
│   │   └── trust.py                             # User trust-score route.
│   └── services/
│       ├── common.py                            # Shared coercion, parsing, date, distance, and normalization helpers.
│       ├── forecasting.py                       # Forecast model selection and response construction.
│       ├── integrations.py                      # Async HTTP calls and normalization for external services.
│       ├── logistics.py                         # Provider filtering, distance calculation, and scoring.
│       ├── pricing.py                           # Price prediction, pressure, and platform-signal logic.
│       └── trust.py                             # Review retrieval, normalization, sentiment, and scoring.
├── ai_training/
│   ├── *.csv, *.xlsx                           # Source datasets used by training notebooks and scripts.
│   ├── *.pkl                                   # Dynamic pricing and legacy forecasting artifacts.
│   └── *.ipynb, forecasts.png                  # Exploratory/training notebooks and generated visualization.
├── new_ai_training/
│   ├── *.pkl                                   # Commodity-specific price, demand, and supply artifacts.
│   ├── *.csv                                   # Training and synthetic observation data.
│   └── *.ipynb, *.png                          # Training notebooks and generated visualizations.
├── scripts/
│   └── train_pipeline.py                        # Rebuilds pricing and demand-forecast artifacts from WFP data.
├── synthetic_data_generation/
│   └── *.py                                    # Generates synthetic demand, price, supply, and weather records.
├── tests/
│   ├── conftest.py                              # Loads local dotenv values and gates live platform tests.
│   ├── test_api.py                              # FastAPI route tests with dummy model loaders.
│   ├── test_*_service.py                        # Unit tests for forecasting, integrations, logistics, pricing, and trust.
│   ├── test_platform_integration.py             # Opt-in live platform reachability tests.
│   └── run_* / generate_*                       # Helpers for real-model test runs and result generation.
└── .github/workflows/deploy.yml                 # Builds/pushes the image and restarts the VM service.
```

Generated folders such as `.venv/`, `__pycache__/`, and `.pytest_cache/` are local tooling output and are not part of the application architecture.

## Architecture and Patterns

- **Application factory:** `app.factory.create_app()` creates the FastAPI instance, installs CORS middleware, and registers each domain router. `main.py` is intentionally thin.
- **Router/service separation:** Routers validate HTTP input and shape responses; domain services contain model inference, normalization, scoring, and integration logic.
- **Import-time model state:** `app.state` loads joblib artifacts when the application imports. The `/health` endpoint checks that the required model collections and feature metadata are present.
- **Pydantic contracts:** Request bodies are defined in `app.models`, providing validation and examples in generated OpenAPI documentation.
- **Async REST integrations:** `httpx.AsyncClient` is used for outbound HTTP GET calls. Responses are normalized from common envelope shapes and warnings are returned instead of silently hiding upstream failures.
- **Platform authentication:** When `PLATFORM_API_KEY` is set, platform requests include the `X-API-KEY` header.
- **Fallback data sources:** Market prices are attempted from the platform first and then from `MARKET_DATA_API_URL`. Weather uses the configured weather endpoint, defaulting to Open-Meteo.
- **Heuristic agricultural context:** Region aliases, Zimbabwe seasons, region coordinates, seasonal climate snapshots, crop climate ranges, planting months, and crop costs are centralized in `app.config`.
- **Bounded pricing adjustment:** Automatic pricing combines demand and supply signals with an adjustment capped by the request's `max_adjustment` value.
- **Geospatial logistics scoring:** Provider candidates are ranked using Haversine route distance plus cost and capacity normalization.
- **CORS:** Origins and credential handling are environment-controlled. Credentials are forcibly disabled when the origin is `*`.
- **No real-time transport:** The code uses HTTP REST calls only; no WebSocket or gRPC transport is implemented.
- **No frontend state/navigation layer:** This repository contains an API service, not a browser or mobile UI.

## Data Models

| Model | File | Key Fields |
| --- | --- | --- |
| `PricePredictionRequest` | `app/models.py` | `commodity`, `market`, `category`, `unit`, `month`, `latitude`, `longitude`, `currency`, `priceflag`; optional administrative and source IDs |
| `BatchPricePredictionRequest` | `app/models.py` | `items: List[PricePredictionRequest]` |
| `AutoPricingRequest` | `app/models.py` | Base price fields plus `use_live_signals`, demand/supply signals, activity counts, `signal_window_days`, `max_adjustment` |
| `HistoricalSalesEntry` | `app/models.py` | `date`, `commodity`, `quantity`, optional `region` |
| `HistoricalSupplyEntry` | `app/models.py` | `date`, `commodity`, `quantity`, optional `region` |
| `WeatherEntry` | `app/models.py` | `date`, rainfall, temperature, optional `region` |
| `MarketEntry` | `app/models.py` | `date`, `commodity`, `price`, optional market and region |
| `ReviewEntry` | `app/models.py` | `rating`, optional comment, verification, helpful votes, reported flag, and review date |
| `DemandSupplyForecastRequest` | `app/models.py` | Optional `region`, `periods`, and commodity list |
| `ClimateSnapshot` | `app/models.py` | Optional `rainfall_mm` and `temperature_c` |
| `DemandSignal` | `app/models.py` | `commodity`, `expected_demand` |
| `PrescriptiveRecommendationRequest` | `app/models.py` | `region`, season/month, `budget_usd`, climate, demand forecast, market data, `top_n`, and external-fetch controls |
| `LogisticsMatchRequest` | `app/models.py` | `request_id` or logistics payload, providers, `top_n`, distance limit, and scoring weights |

## Services and API Layer

| Service | Transport | Purpose |
| --- | --- | --- |
| Pricing service | In-process scikit-learn/joblib; async REST for live signals | Dynamic price prediction, batch inference, price pressure, and live adjustment |
| Forecasting service | In-process serialized forecasting models | Commodity price forecasts and demand/supply projections |
| Integrations service | Async REST via HTTPX | Platform produce/orders/logistics data, external market data, and weather |
| Logistics service | In-process calculations | Haversine distance, availability filtering, and weighted provider ranking |
| Trust service | Async REST via HTTPX plus optional NLTK VADER | Review retrieval, normalization, sentiment-aware trust scoring |
| Common service | In-process helpers | Shared parsing, coercion, normalization, and geographic utilities |

## Navigation and Routing

This is an API, so there are no UI screens or client-side routes. The HTTP route tree is:

```text
/
├── GET /                                  # Service-running message
├── GET /health                            # Model and dependency health
├── POST /predict-price                    # Single price prediction
├── GET /pricing/schema                    # Pricing input/model schema
├── POST /pricing/batch                    # Batch price predictions
├── POST /pricing/auto                     # Price prediction with live signals
├── GET /forecast/{commodity}              # Commodity price forecast
├── POST /forecast/demand-supply            # Demand and supply projections
├── POST /recommendations/prescriptive      # Crop recommendations
├── POST /logistics/match                  # Logistics provider matches
├── GET /integrations/weather              # Weather forecast by coordinates
├── GET /integrations/market-prices        # Platform/external market prices
└── GET /trust-score/{user_id}             # Review-derived trust score
```

Interactive API documentation is available at `/docs`; the generated schema is available at `/openapi.json`.

## Admin Panel

No admin panel or admin-specific routes are present in this repository. All exposed routes are service/API endpoints.

## Configuration

Create a `.env` file in the repository root for local values. `python-dotenv` loads it automatically when `app.config` is imported.

| Variable | Required | Default | Used for |
| --- | --- | --- | --- |
| `CORS_ALLOW_ORIGINS` | No | `*` | Comma-separated allowed origins |
| `CORS_ALLOW_CREDENTIALS` | No | `false` | Enables CORS credentials for non-wildcard origins |
| `SPRING_BOOT_BASE_URL` | For trust scores and review health | Empty | Base URL of the Spring Boot review service |
| `SPRING_BOOT_REVIEWS_PATH` | No | `/api/reviews/user/{user_id}` | Review path template; `{user_id}` is replaced at runtime |
| `PLATFORM_API_BASE_URL` | For live platform features | `http://34.206.207.121:8080` | Core platform API base URL |
| `PLATFORM_API_KEY` | Depends on platform | Empty | Sent as `X-API-KEY` to the platform |
| `PLATFORM_API_TIMEOUT` | No | `6` seconds | Timeout for platform requests and health probes |
| `WEATHER_API_URL` | No | `https://api.open-meteo.com/v1/forecast` | Weather forecast endpoint |
| `MARKET_DATA_API_URL` | No | Empty | Optional external market-data fallback |
| `RUN_PLATFORM_TESTS` | Only for live integration tests | Unset | Set to `1` to enable platform reachability tests |

Example local configuration:

```dotenv
CORS_ALLOW_ORIGINS=http://localhost:3000,http://localhost:5173
CORS_ALLOW_CREDENTIALS=false
SPRING_BOOT_BASE_URL=http://localhost:8080
SPRING_BOOT_REVIEWS_PATH=/api/reviews/user/{user_id}
PLATFORM_API_BASE_URL=http://localhost:8080
PLATFORM_API_KEY=replace-with-a-local-key
PLATFORM_API_TIMEOUT=6
WEATHER_API_URL=https://api.open-meteo.com/v1/forecast
MARKET_DATA_API_URL=
```

Do not commit real API keys or credentials. The service also requires the serialized files loaded by `app.state`, including the pricing artifacts under `ai_training/` and the commodity forecast artifacts under `new_ai_training/`.

## Getting Started

1. **Install Python 3.12 and uv.** The project selects Python 3.12 and uses uv for dependency management.

2. **Install locked dependencies.**

   ```bash
   uv sync
   ```

3. **Configure local integrations.** Copy the configuration pattern above into `.env` and set the platform/review URLs and API key required by the features you intend to exercise.

4. **Verify model assets.** Confirm the files referenced by `app/state.py` exist:

   ```text
   ai_training/dynamic_pricing_model.pkl
   ai_training/model_columns.pkl
   new_ai_training/forecast_models.pkl
   new_ai_training/forecast_features.pkl
   new_ai_training/forecast_commodity_list.pkl
   new_ai_training/demand_models.pkl
   new_ai_training/demand_features.pkl
   new_ai_training/demand_commodity_list.pkl
   new_ai_training/supply_models.pkl
   new_ai_training/supply_features.pkl
   new_ai_training/supply_commodity_list.pkl
   ```

5. **Start the API locally.**

   ```bash
   uv run uvicorn main:app --host 0.0.0.0 --port 8000
   ```

6. **Check the service.**

   ```bash
   curl http://localhost:8000/health
   ```

   Open `http://localhost:8000/docs` for the interactive API explorer.

7. **Run tests.**

   ```bash
   uv run pytest -q
   ```

   Live platform tests are skipped unless explicitly enabled:

   ```bash
   RUN_PLATFORM_TESTS=1 PLATFORM_API_BASE_URL=http://localhost:8080 PLATFORM_API_KEY=your-key uv run pytest -q tests/test_platform_integration.py
   ```

## Backend Requirements

The AI service itself is the backend documented by the route tree above. It also expects the following upstream HTTP areas when live integrations are used:

| Upstream area | Method | Path or URL | Used by |
| --- | --- | --- | --- |
| Platform produce listings | `GET` | `${PLATFORM_API_BASE_URL}/api/v1/produce` | Market prices, recommendations, pricing signals |
| Platform order items | `GET` | `${PLATFORM_API_BASE_URL}/api/v1/order_items` | Pricing demand signals |
| Platform orders | `GET` | `${PLATFORM_API_BASE_URL}/api/v1/orders` | Pricing demand signals and live integration tests |
| Platform logistics request | `GET` | `${PLATFORM_API_BASE_URL}/api/v1/logistics/{request_id}` | Logistics matching when only `request_id` is supplied |
| Platform logistics providers | `GET` | `${PLATFORM_API_BASE_URL}/api/v1/logistics-providers` | Logistics matching when providers are omitted |
| Spring Boot reviews | `GET` | `${SPRING_BOOT_BASE_URL}${SPRING_BOOT_REVIEWS_PATH}` | User trust scoring |
| Weather provider | `GET` | `WEATHER_API_URL` with latitude, longitude, daily fields, forecast days, and UTC timezone | Weather integration and recommendations |
| External market feed | `GET` | `MARKET_DATA_API_URL` with optional `region` and `commodity` query parameters | Market-price fallback |

Platform requests include `X-API-KEY` only when `PLATFORM_API_KEY` is non-empty. The service accepts several common JSON list/envelope shapes from upstream APIs and normalizes them internally.

## Deployment

### Docker

Build and run the image locally:

```bash
docker build -t eharvest-ai .
docker run --rm --env-file .env -p 8000:8000 eharvest-ai
```

The Dockerfile uses `python:3.12-slim`, installs uv, installs from the frozen `uv.lock`, copies the application and model assets, and starts Uvicorn on `0.0.0.0:8000`.

### GitHub Actions

`.github/workflows/deploy.yml` runs on pushes to `main` and:

1. Checks out the repository.
2. Logs in to Docker Hub.
3. Builds and pushes `munacips/eharvest_ai:latest`.
4. SSHes to an AWS VM.
5. Runs `docker compose pull ai-service` and `docker compose up -d ai-service` from `/home/ubuntu/eharvest_main`.

The workflow expects these GitHub Actions secrets:

| Secret | Purpose |
| --- | --- |
| `DOCKERHUB_USERNAME` | Docker Hub login username |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `AWS_HOST` | Deployment VM hostname |
| `AWS_USERNAME` | SSH username |
| `SSH_PRIVATE_KEY` | SSH private key |

The repository does not contain a `docker-compose.yml`; the deployment host must provide a compose service named `ai-service`, expose port `8000`, provide the required `.env` values, and make all model artifacts available in the image or mounted deployment directory.

For production hosting, keep the container behind a TLS-terminating reverse proxy or load balancer, configure a restricted `CORS_ALLOW_ORIGINS` value instead of `*`, provide upstream API credentials through secret storage, and monitor `/health`.

## Model Training

The training pipeline can rebuild the pricing and demand-forecast artifacts from the WFP Zimbabwe food-price dataset:

```bash
uv run python scripts/train_pipeline.py --wfp ai_training/wfp_food_prices_zwe.csv --output-dir ai_training
```

The pipeline writes or refreshes:

```text
ai_training/dynamic_pricing_model.pkl
ai_training/model_columns.pkl
ai_training/demand_forecast_model.pkl
ai_training/forecast_features.pkl
```

The runtime also consumes the newer commodity-specific artifacts under `new_ai_training/`; update those artifacts when changing the corresponding forecast models.

## Testing

Run the complete suite:

```bash
uv run pytest -q
```

The API tests replace model loading with deterministic dummy models, so route behavior can be tested without retraining. Service-level tests cover forecasting, integrations, logistics, pricing, and trust logic. The platform integration tests are opt-in and require a reachable platform API plus `RUN_PLATFORM_TESTS=1`.
