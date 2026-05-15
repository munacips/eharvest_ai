import os
from typing import List

from dotenv import load_dotenv


load_dotenv()


def parse_cors_origins(raw_value: str) -> List[str]:
    if not raw_value:
        return ["*"]
    value = raw_value.strip()
    if value == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


CORS_ALLOW_ORIGINS = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
CORS_ALLOW_CREDENTIALS = os.getenv(
    "CORS_ALLOW_CREDENTIALS", "false").strip().lower()

cors_origins = parse_cors_origins(CORS_ALLOW_ORIGINS)
allow_credentials = CORS_ALLOW_CREDENTIALS in ("1", "true", "yes")
if cors_origins == ["*"]:
    allow_credentials = False


ZWE_SEASONS = {
    "rainy": [11, 12, 1, 2, 3],
    "post_harvest": [4, 5],
    "cool_dry": [6, 7, 8],
    "hot_dry": [9, 10],
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
    "potatoes": {"rain_min": 400, "rain_max": 1000, "temp_min": 12, "temp_max": 25, "plant_months": [6, 7, 8]},
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
    "potatoes": 900,
}


SPRING_BOOT_BASE_URL = os.getenv("SPRING_BOOT_BASE_URL", "").rstrip("/")
SPRING_BOOT_REVIEWS_PATH = os.getenv(
    "SPRING_BOOT_REVIEWS_PATH", "/api/reviews/user/{user_id}")
PLATFORM_API_BASE_URL = os.getenv(
    "PLATFORM_API_BASE_URL", "http://localhost:8080").rstrip("/")
PLATFORM_API_KEY = os.getenv("PLATFORM_API_KEY", "").strip()
PLATFORM_API_TIMEOUT = float(os.getenv("PLATFORM_API_TIMEOUT", "6"))
WEATHER_API_URL = os.getenv(
    "WEATHER_API_URL", "https://api.open-meteo.com/v1/forecast").strip()
MARKET_DATA_API_URL = os.getenv("MARKET_DATA_API_URL", "").strip()
