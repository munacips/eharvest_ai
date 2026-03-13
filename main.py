from fastapi import FastAPI, HTTPException
import httpx
import joblib
import os
import pandas as pd
from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Tuple
from datetime import UTC, datetime, timedelta
import numpy as np

app = FastAPI(title="eHarvest AI API",
              description="API for eHarvest AI services", version="1.0.0")

dynamic_pricing_model = joblib.load('ai_training/dynamic_pricing_model.pkl')
model_columns = joblib.load('ai_training/model_columns.pkl')
forecast_model = joblib.load('ai_training/demand_forecast_model.pkl')
commodity_cols = joblib.load('ai_training/forecast_features.pkl')

# ['commodity', 'market', 'category', 'unit', 'month', 'latitude', 'longitude', 'currency', 'priceflag']


class PricePredictionRequest(BaseModel):
    commodity: str
    market: str
    category: str
    unit: str
    month: int
    latitude: float
    longitude: float
    currency: str
    priceflag: str
    admin1: Optional[str] = None
    admin2: Optional[str] = None
    pricetype: Optional[str] = None
    market_id: Optional[int] = None
    commodity_id: Optional[int] = None
    year: Optional[int] = None


class BatchPricePredictionRequest(BaseModel):
    items: List[PricePredictionRequest]


class HistoricalSalesEntry(BaseModel):
    date: str
    commodity: str
    quantity: float
    region: Optional[str] = None


class HistoricalSupplyEntry(BaseModel):
    date: str
    commodity: str
    quantity: float
    region: Optional[str] = None


class WeatherEntry(BaseModel):
    date: str
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None
    region: Optional[str] = None


class MarketEntry(BaseModel):
    date: str
    commodity: str
    price: float
    market: Optional[str] = None
    region: Optional[str] = None


class ReviewEntry(BaseModel):
    rating: float
    comment: Optional[str] = None
    verified_purchase: Optional[bool] = None
    helpful_votes: Optional[int] = None
    reported: Optional[bool] = None
    review_date: Optional[str] = None


class DemandSupplyForecastRequest(BaseModel):
    region: str
    season: Optional[str] = None
    periods: int = 6
    historical_sales: List[HistoricalSalesEntry]
    historical_supply: Optional[List[HistoricalSupplyEntry]] = None
    weather: Optional[List[WeatherEntry]] = None
    market_data: Optional[List[MarketEntry]] = None
    supply_multiplier: float = 1.05
    commodities: Optional[List[str]] = None


class ClimateSnapshot(BaseModel):
    rainfall_mm: Optional[float] = None
    temperature_c: Optional[float] = None


class DemandSignal(BaseModel):
    commodity: str
    expected_demand: float


class PrescriptiveRecommendationRequest(BaseModel):
    region: str
    season: Optional[str] = None
    month: Optional[int] = None
    budget_usd: float
    climate: ClimateSnapshot
    demand_forecast: Optional[List[DemandSignal]] = None
    market_data: Optional[List[MarketEntry]] = None
    top_n: int = 3


@app.post("/predict-price")
async def predict_price(request: PricePredictionRequest):
    # Convert request to a dict and then to a dataframe
    input_data = pd.DataFrame([request.model_dump()])

    # One-Hot Encode the categorical features
    input_encoded = pd.get_dummies(input_data)

    # Reindex the columns to match the model's expected input
    # this adds any missing columns with 0 values and ensures the order is correct
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)

    # Make prediction using the model
    prediction = dynamic_pricing_model.predict(final_input)

    return {
        "suggested_price": round(float(prediction[0]), 2),
        "currency": request.currency,
        "status": "success"
    }


@app.get("/pricing/schema")
def pricing_schema():
    return {
        "required_fields": [
            "commodity",
            "market",
            "category",
            "unit",
            "month",
            "latitude",
            "longitude",
            "currency",
            "priceflag"
        ],
        "optional_fields": [
            "admin1",
            "admin2",
            "pricetype",
            "market_id",
            "commodity_id",
            "year"
        ],
        "model_columns": model_columns,
        "notes": [
            "month is 1-12",
            "priceflag is a categorical flag from the source data (e.g., actual)",
            "categorical fields are one-hot encoded server-side"
        ]
    }


@app.post("/pricing/batch")
async def predict_price_batch(request: BatchPricePredictionRequest):
    if not request.items:
        raise HTTPException(status_code=400, detail="items cannot be empty")

    input_df = pd.DataFrame([item.model_dump() for item in request.items])
    input_encoded = pd.get_dummies(input_df)
    final_input = input_encoded.reindex(columns=model_columns, fill_value=0)
    predictions = dynamic_pricing_model.predict(final_input)

    results = []
    for item, pred in zip(request.items, predictions):
        results.append({
            "commodity": item.commodity,
            "market": item.market,
            "suggested_price": round(float(pred), 2),
            "currency": item.currency
        })

    return {
        "status": "success",
        "count": len(results),
        "predictions": results
    }


@app.get("/forecast/{commodity}")
async def get_forecast(
    commodity: str,
    periods: int = 30,
    region: Optional[str] = None,
    visual: bool = True
):
    # 1. Create future dates
    future = forecast_model.make_future_dataframe(periods=periods)

    # 2. Set the "Switch" for the requested commodity
    target_col = f"commodity_{commodity}"
    region_col = f"admin1_{region}" if region else None

    for col in commodity_cols:
        if col == target_col:
            future[col] = 1
        elif region_col and col == region_col:
            future[col] = 1
        else:
            future[col] = 0

    # 3. Predict
    forecast = forecast_model.predict(future)

    # Return last 'n' days
    result = forecast[['ds', 'yhat']].tail(periods)
    records = [
        {"date": row["ds"].strftime("%Y-%m-%d"), "value": float(row["yhat"])}
        for _, row in result.iterrows()
    ]
    response = {"commodity": commodity, "region": region, "forecast": records}

    if visual:
        response["visual"] = {
            "type": "line",
            "x": [r["date"] for r in records],
            "series": [{"name": "Demand Forecast", "data": [r["value"] for r in records]}]
        }

    return response


ZWE_SEASONS = {
    "rainy": [11, 12, 1, 2, 3],
    "post_harvest": [4, 5],
    "cool_dry": [6, 7, 8],
    "hot_dry": [9, 10]
}


CROP_PROFILES = {
    "maize": {"rain_min": 500, "rain_max": 1200, "temp_min": 18, "temp_max": 30, "plant_months": [10, 11, 12]},
    "sorghum": {"rain_min": 300, "rain_max": 800, "temp_min": 20, "temp_max": 35, "plant_months": [11, 12, 1]},
    "millet": {"rain_min": 250, "rain_max": 700, "temp_min": 20, "temp_max": 35, "plant_months": [11, 12, 1]},
    "groundnuts": {"rain_min": 400, "rain_max": 900, "temp_min": 18, "temp_max": 30, "plant_months": [11, 12]},
    "beans": {"rain_min": 350, "rain_max": 800, "temp_min": 16, "temp_max": 28, "plant_months": [11, 12]},
    "soybeans": {"rain_min": 450, "rain_max": 1000, "temp_min": 18, "temp_max": 32, "plant_months": [11, 12]},
    "sunflower": {"rain_min": 350, "rain_max": 900, "temp_min": 18, "temp_max": 32, "plant_months": [11, 12, 1]},
    "tomatoes": {"rain_min": 400, "rain_max": 800, "temp_min": 18, "temp_max": 30, "plant_months": [8, 9, 10]},
    "onions": {"rain_min": 300, "rain_max": 700, "temp_min": 13, "temp_max": 28, "plant_months": [6, 7, 8]},
    "potatoes": {"rain_min": 400, "rain_max": 1000, "temp_min": 12, "temp_max": 25, "plant_months": [6, 7, 8]}
}


CROP_COST_USD_PER_HA = {
    "maize": 380,
    "sorghum": 260,
    "millet": 240,
    "groundnuts": 420,
    "beans": 300,
    "soybeans": 450,
    "sunflower": 310,
    "tomatoes": 1500,
    "onions": 1200,
    "potatoes": 900
}

SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL", "").rstrip("/")
SPRING_BOOT_REVIEWS_PATH = os.getenv(
    "SPRING_BOOT_REVIEWS_PATH", "/api/reviews/user/{user_id}")
USE_REVIEW_PLACEHOLDER = os.getenv(
    "USE_REVIEW_PLACEHOLDER", "true").lower() in ("1", "true", "yes")


def _season_from_month(month: int) -> str:
    for season, months in ZWE_SEASONS.items():
        if month in months:
            return season
    return "unknown"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_mean(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    return float(series.mean())


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes", "y")
    if isinstance(value, (int, float)):
        return value != 0
    return None


_VADER_ANALYZER = None


def _get_vader_analyzer():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is False:
        return None
    if _VADER_ANALYZER is not None:
        return _VADER_ANALYZER

    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
    except Exception:
        _VADER_ANALYZER = False
        return None

    try:
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
        return _VADER_ANALYZER
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
        except Exception:
            _VADER_ANALYZER = False

    return _VADER_ANALYZER


def _comment_sentiment_score(comment: Optional[str]) -> Optional[float]:
    if comment is None or not str(comment).strip():
        return None
    analyzer = _get_vader_analyzer()
    if analyzer is None:
        return None
    scores = analyzer.polarity_scores(str(comment))
    return float(scores.get("compound", 0.0))


def _coerce_review_payload(payload: Dict[str, Any]) -> Optional[ReviewEntry]:
    rating = payload.get("rating")
    if rating is None:
        rating = payload.get("score")
    verified = payload.get("verified_purchase")
    if verified is None:
        verified = payload.get("verifiedPurchase")

    helpful = payload.get("helpful_votes")
    if helpful is None:
        helpful = payload.get("helpfulVotes")

    reported = payload.get("reported")
    if reported is None:
        reported = payload.get("flagged")

    review_date = payload.get("review_date")
    if review_date is None:
        review_date = payload.get("createdAt")

    comment = payload.get("comment")
    if comment is None:
        comment = payload.get("review")
    if comment is None:
        comment = payload.get("reviewText")
    if comment is None:
        comment = payload.get("text")

    rating_value = None
    if rating is not None:
        try:
            rating_value = float(rating)
        except (TypeError, ValueError):
            rating_value = None

    sentiment = _comment_sentiment_score(comment)
    text_rating = None
    if sentiment is not None:
        text_rating = _clamp(3.0 + (2.0 * sentiment), 1.0, 5.0)

    if rating_value is None and text_rating is None:
        return None

    if rating_value is None:
        rating_value = text_rating
    elif text_rating is not None:
        rating_value = (0.7 * rating_value) + (0.3 * text_rating)

    helpful_value = None
    if helpful is not None:
        try:
            helpful_value = int(helpful)
        except (TypeError, ValueError):
            helpful_value = None

    return ReviewEntry(
        rating=rating_value,
        comment=comment,
        verified_purchase=_coerce_bool(verified),
        helpful_votes=helpful_value,
        reported=_coerce_bool(reported),
        review_date=review_date,
    )


def _placeholder_reviews(user_id: str) -> List[ReviewEntry]:
    return [
        ReviewEntry(
            rating=4.6,
            comment="Quick delivery and the produce quality was excellent.",
            verified_purchase=True,
            helpful_votes=12,
            reported=False,
            review_date="2025-06-12",
        ),
        ReviewEntry(
            rating=4.2,
            comment="Good experience overall, but packaging could improve.",
            verified_purchase=True,
            helpful_votes=5,
            reported=False,
            review_date="2025-07-03",
        ),
        ReviewEntry(
            rating=3.8,
            comment="Decent service, average quality, nothing special.",
            verified_purchase=False,
            helpful_votes=2,
            reported=False,
            review_date="2025-09-01",
        ),
        ReviewEntry(
            rating=4.9,
            comment="Fantastic support and very fresh produce!",
            verified_purchase=True,
            helpful_votes=18,
            reported=False,
            review_date="2025-11-15",
        ),
        ReviewEntry(
            rating=2.6,
            comment="Late delivery and items were damaged.",
            verified_purchase=False,
            helpful_votes=1,
            reported=True,
            review_date="2026-01-10",
        ),
    ]


def _compute_trust_score(reviews: List[ReviewEntry]) -> Dict[str, Any]:
    if not reviews:
        return {
            "trust_score": 2.5,
            "review_count": 0,
            "average_rating": 0.0,
            "weighted_average": 0.0,
            "reported_ratio": 0.0,
            "verified_ratio": 0.0,
            "note": "no reviews available; returning neutral trust score",
        }

    ratings = []
    weights = []
    reported_count = 0
    verified_count = 0

    for review in reviews:
        rating = _clamp(float(review.rating), 1.0, 5.0)
        helpful_votes = review.helpful_votes or 0
        helpful_votes = max(0, helpful_votes)
        weight = 1.0 + (min(helpful_votes, 20) / 20.0)

        if review.verified_purchase:
            weight += 0.25
            verified_count += 1

        if review.reported:
            weight *= 0.6
            reported_count += 1

        ratings.append(rating)
        weights.append(weight)

    weighted_avg = float(np.average(ratings, weights=weights)) if weights else 0.0
    reported_ratio = reported_count / len(reviews)
    trust = weighted_avg - (reported_ratio * 0.8)
    trust = _clamp(trust, 1.0, 5.0)
    average_rating = float(np.mean(ratings)) if ratings else 0.0
    verified_ratio = verified_count / len(reviews)

    return {
        "trust_score": round(trust, 2),
        "review_count": len(reviews),
        "average_rating": round(average_rating, 2),
        "weighted_average": round(weighted_avg, 2),
        "reported_ratio": round(reported_ratio, 2),
        "verified_ratio": round(verified_ratio, 2),
    }


async def _fetch_user_reviews(user_id: str) -> Tuple[List[ReviewEntry], str, Optional[str]]:
    if USE_REVIEW_PLACEHOLDER or not SPRING_BOOT_BASE_URL:
        return _placeholder_reviews(user_id), "placeholder", None

    path = SPRING_BOOT_REVIEWS_PATH or "/api/reviews/user/{user_id}"
    if not path.startswith("/"):
        path = f"/{path}"
    url = f"{SPRING_BOOT_BASE_URL}{path.format(user_id=user_id)}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()
    except httpx.HTTPError as exc:
        return _placeholder_reviews(user_id), "placeholder_fallback", f"spring_boot_error: {exc.__class__.__name__}"

    raw_reviews = data
    if isinstance(data, dict) and "reviews" in data:
        raw_reviews = data["reviews"]

    reviews: List[ReviewEntry] = []
    if isinstance(raw_reviews, list):
        for item in raw_reviews:
            if not isinstance(item, dict):
                continue
            review = _coerce_review_payload(item)
            if review is not None:
                reviews.append(review)

    return reviews, "spring_boot", None


def _apply_region_filter(df: pd.DataFrame, region: str) -> pd.DataFrame:
    if df is None or df.empty or "region" not in df.columns:
        return df
    filtered = df[df["region"].fillna("").str.lower() == region.lower()].copy()
    return filtered if not filtered.empty else df.copy()


def _compute_weather_impact(weather_df: Optional[pd.DataFrame], season: Optional[str]) -> float:
    if weather_df is None or weather_df.empty:
        return 0.0
    if "date" in weather_df.columns:
        weather_df["date"] = pd.to_datetime(weather_df["date"])
        weather_df["season"] = weather_df["date"].dt.month.apply(
            _season_from_month)

    seasonal = weather_df
    if season:
        seasonal = weather_df[weather_df["season"] == season]
        if seasonal.empty:
            seasonal = weather_df

    rain_avg = _safe_mean(seasonal.get("rainfall_mm"))
    rain_all = _safe_mean(weather_df.get("rainfall_mm"))
    temp_avg = _safe_mean(seasonal.get("temperature_c"))
    temp_all = _safe_mean(weather_df.get("temperature_c"))

    rain_factor = 0.0 if rain_all == 0 else (rain_avg - rain_all) / rain_all
    temp_factor = 0.0 if temp_all == 0 else (temp_avg - temp_all) / temp_all

    return _clamp(0.4 * rain_factor + 0.2 * temp_factor, -0.3, 0.3)


def _compute_market_impact(market_df: Optional[pd.DataFrame], commodity: str) -> float:
    if market_df is None or market_df.empty:
        return 0.0
    commodity_df = market_df[market_df["commodity"].str.lower(
    ) == commodity.lower()].copy()
    if commodity_df.empty:
        return 0.0
    commodity_df["date"] = pd.to_datetime(commodity_df["date"])
    overall = _safe_mean(commodity_df["price"])
    recent_cutoff = commodity_df["date"].max() - timedelta(days=90)
    recent = commodity_df[commodity_df["date"] >= recent_cutoff]
    recent_avg = _safe_mean(recent["price"])
    if overall == 0:
        return 0.0
    price_trend = (recent_avg - overall) / overall
    return _clamp(-0.2 * price_trend, -0.2, 0.2)


def _forecast_monthly_series(
    monthly_df: pd.DataFrame,
    periods: int,
    weather_impact: float,
    market_impact: float
) -> List[Dict[str, Any]]:
    if monthly_df.empty:
        return []

    monthly_df = monthly_df.sort_values("year_month")
    overall_avg = _safe_mean(monthly_df["quantity"])
    monthly_df["month"] = monthly_df["year_month"].dt.month
    seasonal_means = monthly_df.groupby("month")["quantity"].mean()

    if len(monthly_df) >= 2:
        x = np.arange(len(monthly_df))
        y = monthly_df["quantity"].values
        slope = float(np.polyfit(x, y, 1)[0])
    else:
        slope = 0.0

    last_month = monthly_df["year_month"].max()
    results = []
    for i in range(1, periods + 1):
        future_month = last_month + pd.DateOffset(months=i)
        seasonal_base = seasonal_means.get(future_month.month, overall_avg)
        baseline = seasonal_base + slope * i
        adjusted = baseline * (1 + weather_impact + market_impact)
        results.append({
            "date": future_month.strftime("%Y-%m-%d"),
            "value": max(0.0, float(adjusted)),
            "season": _season_from_month(future_month.month)
        })
    return results


@app.post("/forecast/demand-supply")
async def demand_supply_forecast(request: DemandSupplyForecastRequest):
    if request.periods <= 0:
        raise HTTPException(
            status_code=400, detail="periods must be greater than 0")

    sales_df = pd.DataFrame([item.model_dump()
                            for item in request.historical_sales])
    if sales_df.empty:
        raise HTTPException(
            status_code=400, detail="historical_sales cannot be empty")
    sales_df["date"] = pd.to_datetime(sales_df["date"])
    sales_df = _apply_region_filter(sales_df, request.region)

    if request.historical_supply:
        supply_df = pd.DataFrame([item.model_dump()
                                 for item in request.historical_supply])
        supply_df["date"] = pd.to_datetime(supply_df["date"])
        supply_df = _apply_region_filter(supply_df, request.region)
    else:
        supply_df = pd.DataFrame()

    weather_df = None
    if request.weather:
        weather_df = pd.DataFrame([item.model_dump()
                                  for item in request.weather])
        weather_df = _apply_region_filter(weather_df, request.region)

    market_df = None
    if request.market_data:
        market_df = pd.DataFrame([item.model_dump()
                                 for item in request.market_data])
        market_df = _apply_region_filter(market_df, request.region)

    commodities = request.commodities or sorted(
        sales_df["commodity"].dropna().unique().tolist())
    if not commodities:
        raise HTTPException(
            status_code=400, detail="no commodities found in historical_sales")

    weather_impact = _compute_weather_impact(weather_df, request.season)
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
        demand_market_impact = _compute_market_impact(market_df, commodity)
        demand_forecast = _forecast_monthly_series(
            monthly_sales,
            request.periods,
            weather_impact,
            demand_market_impact
        )

        if not supply_df.empty:
            commodity_supply = supply_df[supply_df["commodity"].str.lower(
            ) == commodity.lower()].copy()
            commodity_supply["year_month"] = commodity_supply["date"].dt.to_period(
                "M").dt.to_timestamp()
            monthly_supply = commodity_supply.groupby(
                "year_month")["quantity"].sum().reset_index()
            supply_forecast = _forecast_monthly_series(
                monthly_supply,
                request.periods,
                weather_impact,
                0.0
            )
        else:
            supply_forecast = [
                {
                    "date": point["date"],
                    "value": max(0.0, point["value"] * request.supply_multiplier),
                    "season": point["season"]
                }
                for point in demand_forecast
            ]

        forecasts.append({
            "commodity": commodity,
            "demand": demand_forecast,
            "supply": supply_forecast
        })

        visual_series.append({
            "name": f"{commodity} demand",
            "data": [point["value"] for point in demand_forecast],
            "x": [point["date"] for point in demand_forecast]
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
            "notes": "visual data only; frontend renders charts"
        },
        "assumptions": [
            "seasonal averages and recent trend are used for forecasting",
            "weather and market impacts are applied as bounded adjustments",
            "supply forecast uses provided supply data or demand * supply_multiplier"
        ]
    }


@app.post("/recommendations/prescriptive")
async def prescriptive_recommendations(request: PrescriptiveRecommendationRequest):
    if request.top_n <= 0:
        raise HTTPException(
            status_code=400, detail="top_n must be greater than 0")

    target_month = request.month or datetime.now(UTC).month
    season = request.season or _season_from_month(target_month)

    demand_map = {item.commodity.lower(): item.expected_demand for item in (
        request.demand_forecast or [])}

    climate_rain = request.climate.rainfall_mm
    climate_temp = request.climate.temperature_c

    market_df = None
    if request.market_data:
        market_df = pd.DataFrame([item.model_dump()
                                 for item in request.market_data])
        market_df = _apply_region_filter(market_df, request.region)

    scored = []
    for crop, profile in CROP_PROFILES.items():
        plant_ok = target_month in profile["plant_months"]

        rain_ok = True if climate_rain is None else (
            profile["rain_min"] <= climate_rain <= profile["rain_max"])
        temp_ok = True if climate_temp is None else (
            profile["temp_min"] <= climate_temp <= profile["temp_max"])
        climate_score = 1.0 if (rain_ok and temp_ok) else 0.0

        demand_score = 0.5
        if crop in demand_map:
            demand_score = min(
                1.0, max(0.0, demand_map[crop] / max(demand_map.values() or [1])))

        cost = CROP_COST_USD_PER_HA.get(crop, 400)
        budget_score = 1.0 if request.budget_usd >= cost else 0.4

        total = (0.45 * climate_score) + \
            (0.35 * demand_score) + (0.20 * budget_score)
        if plant_ok:
            total += 0.1

        scored.append({
            "commodity": crop,
            "score": round(total, 3),
            "estimated_cost_usd_per_ha": cost,
            "planting_months": profile["plant_months"],
            "climate_fit": climate_score == 1.0
        })

    scored = sorted(scored, key=lambda x: x["score"], reverse=True)[
        : request.top_n]

    recommendations = []
    for item in scored:
        markets = []
        if market_df is not None and not market_df.empty:
            crop_prices = market_df[market_df["commodity"].str.lower(
            ) == item["commodity"]]
            if not crop_prices.empty and "market" in crop_prices.columns:
                best = crop_prices.groupby("market")["price"].mean(
                ).sort_values(ascending=False).head(3)
                markets = [{"market": idx, "avg_price": round(
                    float(val), 2)} for idx, val in best.items()]

        recommendations.append({
            "commodity": item["commodity"],
            "score": item["score"],
            "why": {
                "climate_fit": item["climate_fit"],
                "demand_signal": demand_map.get(item["commodity"], None),
                "estimated_cost_usd_per_ha": item["estimated_cost_usd_per_ha"],
                "planting_months": item["planting_months"]
            },
            "market_targets": markets
        })

    return {
        "region": request.region,
        "season": season,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recommendations": recommendations,
        "assumptions": [
            "crop profiles reflect typical Zimbabwe growing ranges",
            "budget score uses a rough cost-per-hectare estimate",
            "market targets use average prices in provided market_data"
        ]
    }


@app.get("/trust-score/{user_id}")
async def trust_score(user_id: str):
    reviews, source, warning = await _fetch_user_reviews(user_id)
    details = _compute_trust_score(reviews)

    response = {
        "user_id": user_id,
        "trust_score": details["trust_score"],
        "scale": 5,
        "review_count": details["review_count"],
        "source": source,
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "details": {
            "average_rating": details["average_rating"],
            "weighted_average": details["weighted_average"],
            "reported_ratio": details["reported_ratio"],
            "verified_ratio": details["verified_ratio"],
        },
    }

    if "note" in details:
        response["details"]["note"] = details["note"]

    if warning:
        response["warnings"] = [warning]

    return response


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def read_root():
    return {"message": "API is running. Use /predict-price for price predictions and /forecast/commodity for demand forecasts."}


'''
Datasets Links:

-https://data.humdata.org/dataset/global-wfp-food-prices
-https://bulks-faostat.fao.org/production/production_crops_E_All_Data_(Normalized).zip
-https://fenixservices.fao.org/faostat/static/bulkdownloads/FoodBalanceSheets_E_All_Data_(Normalized).zip
-https://data.chc.ucsb.edu/products/CHIRPS/v3.0/
-https://power.larc.nasa.gov/docs/services/api/temporal/hourly/
-https://zimstat.co.zw/agriculture-statistics/
'''
