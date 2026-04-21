from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    return {"status": "ok"}


@router.get("/")
def read_root():
    return {
        "message": "API is running. Use /predict-price for price predictions and /forecast/commodity for demand forecasts."
    }
