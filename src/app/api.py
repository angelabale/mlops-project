from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Car Price Prediction API")

# Load trained pipeline once at startup
project_root = Path(__file__).parent.parent.parent
model_path = project_root / "model.joblib"
model = joblib.load(model_path)


# Define request schema
class CarFeatures(BaseModel):
    Brand: str
    Engine_Size: float = Field(alias="Engine Size")
    Fuel_Type: str = Field(alias="Fuel Type")
    Transmission: str
    Mileage: float
    Condition: str
    Model: str
    Year: int


@app.get("/")
def root():
    return {"message": "Car Price Prediction API is running"}


@app.post("/predict")
def predict(features: CarFeatures):
    # Convert input to DataFrame
    input_dict = features.dict(by_alias=True)
    df = pd.DataFrame([input_dict])

    # Recreate feature engineering done in training
    current_year = df["Year"].max()
    df["Car_Age"] = current_year - df["Year"]
    df["Mileage_log"] = np.log1p(df["Mileage"])

    # Drop unused columns like in training
    df = df.drop(columns=["Year"])

    # Predict (output is log-price)
    prediction_log = model.predict(df)[0]

    # Inverse log-transform
    prediction = np.expm1(prediction_log)

    return {"predicted_price": float(prediction)}
