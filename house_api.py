from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pydantic import Field

model = joblib.load("linear_regression_model.pkl")
model_columns = joblib.load("model_columns.pkl")

app = FastAPI(title="🏠 House Price Prediction API")

class HouseFeatures(BaseModel):
    area: int = Field(..., description="House area in sqft")
    bedrooms: int = Field(..., description="No. of bedrooms")
    bathrooms: int = Field(..., description="No. of bathroms")
    stories: int = Field(..., description="Floors (1–4)")
    mainroad: str = Field(..., description="yes/no")
    guestroom: str = Field(..., description="yes/no")
    basement: str = Field(..., description="yes/no")
    hotwaterheating: str = Field(..., description="yes/no")
    airconditioning: str = Field(..., description="yes/no")
    parking: int = Field(..., description="Parking spots (0–3)")
    prefarea: str = Field(..., description="Preferred area (yes/no)")
    furnishingstatus: str = Field(..., description="furnished / semi-furnished / unfurnished")

def preprocess_input(data: dict):
    df = pd.DataFrame([data])

    binary_cols = ['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea']
    for col in binary_cols:
        df[col] = df[col].map({'yes': 1, 'no': 0})
    df = pd.get_dummies(df, columns=['furnishingstatus'], drop_first=True)
    for col in model_columns:
        if col not in df.columns:
            df[col] = 0
    df = df[model_columns]
    return df

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: HouseFeatures):
    data = preprocess_input(features.model_dump())
    prediction = model.predict(data)[0]
    return {"predicted_price": round(prediction, 2)}