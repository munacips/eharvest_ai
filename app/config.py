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


REGION_ALIASES = {
    "byo": "bulawayo",
    "bulawayo province": "bulawayo",
    "harare province": "harare",
    "mash central": "mashonaland central",
    "mashonaland central province": "mashonaland central",
    "mashonaland central prov": "mashonaland central",
    "mash east": "mashonaland east",
    "mashonaland east province": "mashonaland east",
    "mashonaland east prov": "mashonaland east",
    "mash west": "mashonaland west",
    "mashonaland west province": "mashonaland west",
    "mashonaland west prov": "mashonaland west",
    "masvingo province": "masvingo",
    "mat north": "matabeleland north",
    "matabeleland n": "matabeleland north",
    "matabeleland north province": "matabeleland north",
    "matabeleland north prov": "matabeleland north",
    "mat south": "matabeleland south",
    "matabeleland s": "matabeleland south",
    "matabeleland south province": "matabeleland south",
    "matabeleland south prov": "matabeleland south",
    "manicaland province": "manicaland",
    "midlands province": "midlands",
}


REGION_COORDS = {
    "harare": (-17.8292, 31.0522),
    "bulawayo": (-20.1564, 28.5880),
    "manicaland": (-18.9216, 32.1746),
    "mashonaland central": (-16.7644, 31.3801),
    "mashonaland east": (-18.5872, 31.2626),
    "mashonaland west": (-17.4851, 29.8926),
    "masvingo": (-20.0640, 30.8320),
    "matabeleland north": (-18.5333, 27.8500),
    "matabeleland south": (-21.0523, 29.5280),
    "midlands": (-19.0552, 29.6035),
}


REGION_SEASON_CLIMATE = {
    "harare": {
        "rainy": {"rainfall_mm": 180.0, "temperature_c": 23.0},
        "post_harvest": {"rainfall_mm": 45.0, "temperature_c": 20.0},
        "cool_dry": {"rainfall_mm": 10.0, "temperature_c": 16.0},
        "hot_dry": {"rainfall_mm": 5.0, "temperature_c": 26.0},
    },
    "bulawayo": {
        "rainy": {"rainfall_mm": 130.0, "temperature_c": 24.0},
        "post_harvest": {"rainfall_mm": 35.0, "temperature_c": 21.0},
        "cool_dry": {"rainfall_mm": 8.0, "temperature_c": 15.0},
        "hot_dry": {"rainfall_mm": 4.0, "temperature_c": 27.0},
    },
    "manicaland": {
        "rainy": {"rainfall_mm": 220.0, "temperature_c": 22.0},
        "post_harvest": {"rainfall_mm": 60.0, "temperature_c": 19.0},
        "cool_dry": {"rainfall_mm": 12.0, "temperature_c": 15.0},
        "hot_dry": {"rainfall_mm": 6.0, "temperature_c": 25.0},
    },
    "mashonaland central": {
        "rainy": {"rainfall_mm": 190.0, "temperature_c": 23.0},
        "post_harvest": {"rainfall_mm": 50.0, "temperature_c": 20.0},
        "cool_dry": {"rainfall_mm": 10.0, "temperature_c": 16.0},
        "hot_dry": {"rainfall_mm": 5.0, "temperature_c": 26.0},
    },
    "mashonaland east": {
        "rainy": {"rainfall_mm": 200.0, "temperature_c": 23.0},
        "post_harvest": {"rainfall_mm": 55.0, "temperature_c": 20.0},
        "cool_dry": {"rainfall_mm": 10.0, "temperature_c": 16.0},
        "hot_dry": {"rainfall_mm": 5.0, "temperature_c": 26.0},
    },
    "mashonaland west": {
        "rainy": {"rainfall_mm": 175.0, "temperature_c": 23.0},
        "post_harvest": {"rainfall_mm": 45.0, "temperature_c": 20.0},
        "cool_dry": {"rainfall_mm": 9.0, "temperature_c": 16.0},
        "hot_dry": {"rainfall_mm": 4.0, "temperature_c": 26.0},
    },
    "masvingo": {
        "rainy": {"rainfall_mm": 150.0, "temperature_c": 24.0},
        "post_harvest": {"rainfall_mm": 40.0, "temperature_c": 21.0},
        "cool_dry": {"rainfall_mm": 8.0, "temperature_c": 15.0},
        "hot_dry": {"rainfall_mm": 4.0, "temperature_c": 27.0},
    },
    "matabeleland north": {
        "rainy": {"rainfall_mm": 140.0, "temperature_c": 24.0},
        "post_harvest": {"rainfall_mm": 35.0, "temperature_c": 21.0},
        "cool_dry": {"rainfall_mm": 7.0, "temperature_c": 15.0},
        "hot_dry": {"rainfall_mm": 4.0, "temperature_c": 27.0},
    },
    "matabeleland south": {
        "rainy": {"rainfall_mm": 125.0, "temperature_c": 24.0},
        "post_harvest": {"rainfall_mm": 30.0, "temperature_c": 21.0},
        "cool_dry": {"rainfall_mm": 7.0, "temperature_c": 15.0},
        "hot_dry": {"rainfall_mm": 3.0, "temperature_c": 27.0},
    },
    "midlands": {
        "rainy": {"rainfall_mm": 160.0, "temperature_c": 23.0},
        "post_harvest": {"rainfall_mm": 40.0, "temperature_c": 20.0},
        "cool_dry": {"rainfall_mm": 10.0, "temperature_c": 16.0},
        "hot_dry": {"rainfall_mm": 5.0, "temperature_c": 26.0},
    },
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
