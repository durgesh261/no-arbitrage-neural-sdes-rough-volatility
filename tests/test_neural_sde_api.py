import os
import sys
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Neural SDE" in data["service"]
    assert "Rough Volatility" in data["stochastic_regime"]


def test_martingale_check_endpoint(client):
    response = client.get("/api/v1/martingale-check?n_paths=1500")
    assert response.status_code == 200
    data = response.json()
    assert data["martingale_property_verified"] is True
    assert data["discrepancy_pct"] < 3.0


def test_option_pricing_endpoint(client):
    payload = {
        "spot_price_s0": 100.0,
        "strike_k": 100.0,
        "maturity_years": 0.5,
        "n_paths": 1500
    }
    response = client.post("/api/v1/price-option", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "neural_sde_call_price_eur" in data
    assert data["neural_sde_call_price_eur"] > 0.0
    assert "inverted_implied_volatility_pct" in data
