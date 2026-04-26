from fastapi.testclient import TestClient
from src.deployment.api import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "running"

def test_predict_invalid_input():
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422
