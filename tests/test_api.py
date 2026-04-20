"""tests/test_api.py — FastAPI endpoint integration tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from src.api.main import app
    with TestClient(app) as c:
        yield c


class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "models_loaded" in data


class TestDataEndpoints:
    def test_prices_returns_list(self, client):
        resp = client.get("/api/data/prices")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)
            if data:
                assert "date" in data[0]
                assert "spx"  in data[0]
                assert "btc"  in data[0]

    def test_regimes_returns_list(self, client):
        resp = client.get("/api/data/regimes")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)
            if data:
                assert "date"         in data[0]
                assert "regime_label" in data[0]
                assert "crisis_flag"  in data[0]


class TestModelEndpoints:
    def test_metrics_returns_dict(self, client):
        resp = client.get("/api/models/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)

    def test_benchmark_structure(self, client):
        resp = client.get("/api/models/benchmark")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert "best_model_by_crisis_pr_auc" in data
            assert "comparison" in data

    def test_unknown_model_404(self, client):
        resp = client.get("/api/models/nonexistent/predictions")
        assert resp.status_code == 404

    @pytest.mark.parametrize("name", ["hmm", "xgboost", "lstm", "tft", "vqh"])
    def test_model_confusion_matrix_structure(self, client, name):
        resp = client.get(f"/api/models/{name}/confusion-matrix")
        if resp.status_code == 200:
            data = resp.json()
            assert "matrix" in data
            assert len(data["matrix"]) == 3
            assert len(data["matrix"][0]) == 3
            assert "class_names" in data


class TestQuantumEndpoints:
    def test_entanglement_returns_list_or_404(self, client):
        resp = client.get("/api/quantum/entanglement")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)

    def test_lead_lag_returns_list_or_404(self, client):
        resp = client.get("/api/quantum/lead-lag")
        assert resp.status_code in (200, 404)

    def test_latest_predictions_returns_list_or_404(self, client):
        resp = client.get("/api/predictions/latest")
        assert resp.status_code in (200, 404)
        if resp.status_code == 200:
            data = resp.json()
            assert isinstance(data, list)
            if data:
                row = data[0]
                assert "date"        in row
                assert "prob_crisis" in row
                assert "match"       in row
