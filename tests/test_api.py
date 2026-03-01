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