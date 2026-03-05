from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Price Prediction API")

# Load trained pipeline once at startup
project_root = Path(__file__).parent.parent.parent
model_path = project_root / "model.joblib"
model = joblib.load(model_path)

# Monitoring metrics
request_count = 0
predict_count = 0
error_count = 0
total_latency = 0.0

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

    global request_count, predict_count, error_count, total_latency

    request_count += 1
    start_time = time.time()

    try:
        logger.info("Prediction request received")

        # Convert input to DataFrame
        input_dict = features.dict(by_alias=True)
        df = pd.DataFrame([input_dict])

        # Recreate feature engineering done in training
        current_year = df["Year"].max()
        df["Car_Age"] = current_year - df["Year"]
        df["Mileage_log"] = np.log1p(df["Mileage"])

        df = df.drop(columns=["Year"])

        # Predict (output is log-price)
        prediction_log = model.predict(df)[0]

        # Inverse log-transform
        prediction = np.expm1(prediction_log)

        latency = time.time() - start_time
        total_latency += latency
        predict_count += 1

        logger.info(f"Prediction completed in {latency:.4f} seconds")

        return {"predicted_price": float(prediction)}

    except Exception as e:
        error_count += 1
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():

    avg_latency = 0
    if predict_count > 0:
        avg_latency = total_latency / predict_count

    return {
        "request_count": request_count,
        "predict_count": predict_count,
        "error_count": error_count,
        "avg_latency_seconds": avg_latency
    }