# How to run: python scripts/train_pipeline.py --wfp ai_training/wfp_food_prices_zwe.csv --output-dir ai_training


import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_wfp_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["admin1"] = df.get("admin1", pd.Series([None] * len(df))).fillna("Unknown")
    df["admin2"] = df.get("admin2", pd.Series([None] * len(df))).fillna("Unknown")
    df["pricetype"] = df.get("pricetype", pd.Series([None] * len(df))).fillna("Unknown")
    df["priceflag"] = df.get("priceflag", pd.Series([None] * len(df))).fillna("Unknown")

    for col in ["latitude", "longitude", "market_id", "commodity_id"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    for col in ["latitude", "longitude", "market_id", "commodity_id"]:
        median_val = df[col].median() if df[col].notna().any() else 0.0
        df[col] = df[col].fillna(median_val)

    return df


def train_dynamic_pricing(df: pd.DataFrame, output_dir: Path, seed: int) -> dict:
    features = [
        "commodity",
        "market",
        "category",
        "unit",
        "pricetype",
        "priceflag",
        "admin1",
        "admin2",
        "month",
        "year",
        "latitude",
        "longitude",
        "market_id",
        "commodity_id",
        "currency",
    ]

    missing = [col for col in features + ["usdprice"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for pricing model: {missing}")

    X = pd.get_dummies(df[features])
    y = df["usdprice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = RandomForestRegressor(n_estimators=200, random_state=seed)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "rows": int(len(df)),
    }

    _ensure_parent_dir(output_dir / "dynamic_pricing_model.pkl")
    joblib.dump(model, output_dir / "dynamic_pricing_model.pkl")
    joblib.dump(list(X.columns), output_dir / "model_columns.pkl")

    return metrics


def train_demand_forecast(df: pd.DataFrame, output_dir: Path) -> dict:
    try:
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError("prophet is required to train the demand forecast model") from exc

    df = df.copy()
    df["ds"] = pd.to_datetime(df["date"])
    df["y"] = df["usdprice"]
    df["admin1"] = df["admin1"].fillna("Unknown")

    df_agg = df.groupby(["ds", "commodity", "admin1"])["y"].mean().reset_index()

    df_prophet = pd.get_dummies(df_agg, columns=["commodity", "admin1"])
    regressor_cols = [
        col for col in df_prophet.columns if col.startswith("commodity_") or col.startswith("admin1_")
    ]

    model = Prophet(yearly_seasonality=True, daily_seasonality=False)
    for col in regressor_cols:
        model.add_regressor(col)

    model.fit(df_prophet)

    _ensure_parent_dir(output_dir / "demand_forecast_model.pkl")
    joblib.dump(model, output_dir / "demand_forecast_model.pkl")
    joblib.dump(regressor_cols, output_dir / "forecast_features.pkl")

    return {"rows": int(len(df_agg)), "regressors": len(regressor_cols)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train eHarvest AI models (pricing + demand forecast).")
    parser.add_argument(
        "--wfp",
        type=Path,
        default=Path("ai_training/wfp_food_prices_zwe.csv"),
        help="Path to WFP Zimbabwe food prices CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ai_training"),
        help="Directory to write trained models",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for training")
    args = parser.parse_args()

    if not args.wfp.exists():
        print(f"WFP dataset not found: {args.wfp}", file=sys.stderr)
        return 1

    df = load_wfp_prices(args.wfp)

    pricing_metrics = train_dynamic_pricing(df, args.output_dir, args.seed)
    forecast_metrics = train_demand_forecast(df, args.output_dir)

    print("Training complete")
    print(f"Dynamic pricing metrics: {pricing_metrics}")
    print(f"Demand forecast metrics: {forecast_metrics}")
    print(f"Models saved to: {args.output_dir.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
