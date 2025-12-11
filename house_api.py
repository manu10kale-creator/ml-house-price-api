# house_api.py (fixed for Pydantic v2)
import os
import logging
from typing import List, Optional, Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -------------------------
# Config & Logging
# -------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "linear_regression_model.pkl")
COLUMNS_PATH = os.getenv("COLUMNS_PATH", "model_columns.pkl")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------------
# App Initialization
# -------------------------
app = FastAPI(
    title="🏠 Advanced House Price Prediction API",
    description="Upgraded ML API with preprocessing pipeline, batch prediction, logging, and explanations.",
    version="1.0.0"
)

# Allow all origins for demo — restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# These will be populated at startup
model = None
model_columns: List[str] = []
feature_importance: Optional[pd.Series] = None
model_name: str = "unknown"

# -------------------------
# Schemas
# -------------------------
class HouseFeatures(BaseModel):
    area: int = Field(..., ge=200, description="House area in sqft")
    bedrooms: int = Field(..., ge=0, le=10)
    bathrooms: int = Field(..., ge=0, le=10)
    stories: int = Field(..., ge=1, le=4)
    # NOTE: pydantic v2 uses 'pattern' instead of 'regex'
    mainroad: str = Field(..., pattern="^(yes|no)$")
    guestroom: str = Field(..., pattern="^(yes|no)$")
    basement: str = Field(..., pattern="^(yes|no)$")
    hotwaterheating: str = Field(..., pattern="^(yes|no)$")
    airconditioning: str = Field(..., pattern="^(yes|no)$")
    parking: int = Field(..., ge=0, le=3)
    prefarea: str = Field(..., pattern="^(yes|no)$")
    furnishingstatus: str = Field(..., pattern="^(furnished|semi-furnished|unfurnished)$")


class BatchRequest(BaseModel):
    items: List[HouseFeatures]


class PredictResponse(BaseModel):
    predicted_price: float
    top_contributing_features: List[Dict[str, Any]]
    status: str


# -------------------------
# Preprocessing / Helpers
# -------------------------
BINARY_COLS = [
    'mainroad', 'guestroom', 'basement',
    'hotwaterheating', 'airconditioning', 'prefarea'
]


def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Convert single record (dict) to model-ready DataFrame using model_columns.
    """
    df = pd.DataFrame([data])

    for col in BINARY_COLS:
        df[col] = df[col].map({"yes": 1, "no": 0})

    df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)

    # ensure all expected columns are present
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    # preserve ordering
    df = df[model_columns]
    return df


def get_top_contributors(preprocessed_row: pd.DataFrame, top_k: int = 3):
    """
    Return top contributing features using linear coefficients or feature_importance.
    Works for linear models (coef_) and tree models (feature_importances_).
    """
    if preprocessed_row is None or preprocessed_row.shape[1] != len(model_columns):
        return []

    # if model has coef_ attribute (linear models)
    if hasattr(model, "coef_"):
        coef = getattr(model, "coef_")
        importance = preprocessed_row.values[0] * coef
        idx = importance.argsort()[::-1]
        return [
            {"feature": model_columns[i], "impact": round(float(importance[i]), 3)}
            for i in idx[:top_k]
        ]

    # if model has feature_importances_ (tree models)
    if hasattr(model, "feature_importances_"):
        fi = getattr(model, "feature_importances_")
        # multiply input value by feature importance to show contribution-like metric
        contrib = preprocessed_row.values[0] * fi
        idx = contrib.argsort()[::-1]
        return [
            {"feature": model_columns[i], "impact": round(float(contrib[i]), 3)}
            for i in idx[:top_k]
        ]

    return []


# -------------------------
# Startup: load artifacts
# -------------------------
@app.on_event("startup")
def load_artifacts():
    global model, model_columns, feature_importance, model_name
    try:
        logger.info("Loading model from %s", MODEL_PATH)
        model = joblib.load(MODEL_PATH)
        model_columns = joblib.load(COLUMNS_PATH)
        model_name = type(model).__name__
        logger.info("Loaded model '%s' and %d columns", model_name, len(model_columns))
    except Exception as e:
        logger.exception("Failed to load model or columns: %s", e)
        # keep model as None; endpoints will surface meaningful error


# -------------------------
# Simple health & metadata endpoints
# -------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/metadata")
async def metadata():
    return {
        "model_name": model_name,
        "n_columns": len(model_columns),
        "columns_sample": model_columns[:10]
    }


# -------------------------
# Prediction endpoints
# -------------------------
@app.post("/predict", response_model=PredictResponse)
async def predict(features: HouseFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        original = features.model_dump()
        preprocessed = preprocess_input(original)
        pred = float(model.predict(preprocessed)[0])
        explanation = get_top_contributors(preprocessed)

        logger.info("Prediction: input=%s -> pred=%s", original, pred)

        return {
            "predicted_price": round(pred, 2),
            "top_contributing_features": explanation,
            "status": "success"
        }
    except Exception as e:
        logger.exception("Error in /predict: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        inputs = [item.model_dump() for item in request.items]
        preds = []
        for inp in inputs:
            pre = preprocess_input(inp)
            preds.append(round(float(model.predict(pre)[0]), 2))

        logger.info("Batch predict for %d items", len(preds))
        return {"predictions": preds, "count": len(preds), "status": "success"}

    except Exception as e:
        logger.exception("Batch predict error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
