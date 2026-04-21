eHarvest AI

Overview
eHarvest AI is a FastAPI service that provides pricing predictions, demand forecasting, prescriptive crop recommendations, and buyer trust scoring for agricultural markets. It serves a trained dynamic pricing model, a Prophet-based demand forecast model, and heuristic analytics for demand-supply planning and recommendations.

Key Features
- Dynamic price prediction for commodities with batch support.
- Demand forecasting per commodity (with optional region conditioning).
- Demand vs supply projections using recent sales, weather, and market data.
- Prescriptive crop recommendations based on climate, budget, and demand signals.
- Trust score computation from review data (Spring Boot integration or placeholder data).

Project Structure
- main.py: Thin entrypoint that imports the app.
- app/factory.py: FastAPI app factory and router registration.
- app/models.py: Pydantic request/response models.
- app/services/: Business logic grouped by domain (pricing, forecasting, logistics, integrations, trust).
- app/routers/: Route handlers grouped by feature.
- scripts/train_pipeline.py: Training pipeline for pricing and demand forecast models.
- ai_training/: Training datasets and serialized model artifacts.
- tests/test_api.py: API test suite with dummy models.
- Dockerfile: Container build with uv for dependencies.

Requirements
- Python 3.12
- uv (used for dependency management and running)

Setup
Install dependencies with uv:

```bash
uv sync
```

Run the API
Start the service locally:

```bash
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Default URLs:
- API root: http://localhost:8000/
- Health check: http://localhost:8000/health
- Interactive docs: http://localhost:8000/docs

Docker
Build and run:

```bash
docker build -t eharvest-ai .
docker run --rm -p 8000:8000 eharvest-ai
```

Environment Variables
These control the trust score integration with a Spring Boot reviews service.
- `SPRING_BOOT_BASE_URL`: Base URL of the Spring Boot service (e.g., `http://localhost:8080`).
- `SPRING_BOOT_REVIEWS_PATH`: Reviews endpoint path. Default: `/api/reviews/user/{user_id}`.
- `USE_REVIEW_PLACEHOLDER`: If true, use placeholder reviews instead of calling Spring Boot. Default: `true`.

Platform + Live Data Integration
- `PLATFORM_API_BASE_URL`: Base URL of the core platform API (default `http://localhost:8080`).
- `PLATFORM_API_KEY`: API key for the platform (`X-API-KEY` header).
- `PLATFORM_API_TIMEOUT`: Request timeout in seconds (default `6`).
- `WEATHER_API_URL`: Weather API endpoint (default `https://api.open-meteo.com/v1/forecast`).
- `MARKET_DATA_API_URL`: Optional external market data feed (e.g., ZIMSTAT proxy).

API Endpoints
Pricing
- `POST /predict-price`: Predict a suggested price. Accepts a single request payload.
- `POST /pricing/batch`: Batch pricing predictions.
- `GET /pricing/schema`: Returns required fields and model columns.
- `POST /pricing/auto`: Predict price and apply supply/demand adjustments from live platform signals.

Forecasting
- `GET /forecast/{commodity}`: Demand forecast for a commodity. Query params: `periods`, `region`, `visual`.
- `POST /forecast/demand-supply`: Demand and supply projections from historical data.

Recommendations
- `POST /recommendations/prescriptive`: Recommend crops based on climate, budget, and demand signals.

Logistics
- `POST /logistics/match`: Match a logistics request to providers using route efficiency and cost signals.

Integrations
- `GET /integrations/weather`: Fetch live weather data for a latitude/longitude.
- `GET /integrations/market-prices`: Fetch market prices from platform produce data or external feed.

Trust Score
- `GET /trust-score/{user_id}`: Computes a trust score from reviews.

Example Requests
Auto pricing with live signals:
```json
{
  "commodity": "maize",
  "market": "harare",
  "category": "cereals",
  "unit": "KG",
  "month": 11,
  "latitude": -17.8,
  "longitude": 31.0,
  "currency": "USD",
  "priceflag": "actual",
  "use_live_signals": true,
  "signal_window_days": 30,
  "max_adjustment": 0.3
}
```

Logistics match using platform data:
```json
{
  "request_id": "123",
  "top_n": 3,
  "weights": {"cost": 0.4, "distance": 0.4, "capacity": 0.2}
}
```

Health
- `GET /health`: Health check.

Training Models
Train or refresh model artifacts from the WFP Zimbabwe food price dataset:

```bash
python scripts/train_pipeline.py --wfp ai_training/wfp_food_prices_zwe.csv --output-dir ai_training
```

Artifacts generated:
- `ai_training/dynamic_pricing_model.pkl`
- `ai_training/model_columns.pkl`
- `ai_training/demand_forecast_model.pkl`
- `ai_training/forecast_features.pkl`

Tests
Run the test suite:

```bash
pytest -q
```

Notes
- The pricing model is a RandomForest regressor trained on WFP price data.
- The demand forecast model uses Prophet with commodity and region regressors.
- Forecasting and recommendations include heuristics derived from Zimbabwe-specific seasonality and crop profiles.
