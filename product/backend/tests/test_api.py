"""Basic API tests."""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    """Test root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health():
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_list_arquetipos():
    """Test list archetypes endpoint."""
    response = client.get("/api/arquetipos/")
    assert response.status_code == 200
    assert "arquetipos" in response.json()


def test_list_templates():
    """Test list templates endpoint."""
    response = client.get("/api/arquetipos/templates/")
    assert response.status_code == 200
    assert "templates" in response.json()
    assert len(response.json()["templates"]) > 0


def test_list_analisis():
    """Test list analyses endpoint."""
    response = client.get("/api/analisis/")
    assert response.status_code == 200
    assert "analisis" in response.json()
