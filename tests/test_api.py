from fastapi.testclient import TestClient
from src.app.api import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_predict():
    response = client.post("/predict", json={
        "Brand": "Toyota",
        "Engine Size": 2.0,
        "Fuel Type": "Petrol",
        "Transmission": "Automatic",
        "Mileage": 50000,
        "Condition": "Good",
        "Model": "Corolla",
        "Year": 2020
    })
    assert response.status_code == 200
    assert "predicted_price" in response.json()

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_metrics_initial():
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "request_count" in body
    assert "predict_count" in body
    assert "error_count" in body
    assert "avg_latency_seconds" in body

def test_metrics_increase_after_predict():
    # call predict once
    client.post("/predict", json={
        "Brand": "Toyota",
        "Engine Size": 2.0,
        "Fuel Type": "Petrol",
        "Transmission": "Automatic",
        "Mileage": 50000,
        "Condition": "Good",
        "Model": "Corolla",
        "Year": 2020
    })

    r = client.get("/metrics")
    body = r.json()
    assert body["predict_count"] >= 1