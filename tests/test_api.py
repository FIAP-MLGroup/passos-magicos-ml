from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

VALID_PAYLOAD = {
    "INDE_22": 7.0,
    "IAA": 6.5,
    "IEG": 5.0,
    "IPS": 6.0,
    "IDA": 5.5,
    "IPV": 7.0,
    "Matem": 6.0,
    "Portug": 5.5,
    "Fase": 3,
    "Genero": "Menino",
    "Pedra_22": "Ametista",
    "Atingiu_PV": "Não",
    "Indicado": "Não",
    "Idade_22": 14,
}


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([1])
    model.predict_proba.return_value = np.array([[0.2, 0.5, 0.3]])
    return model


@pytest.fixture
def client(mock_model):
    # Patch joblib.load para que o lifespan use o mock em vez do modelo real
    with patch("app.main.joblib.load", return_value=mock_model):
        with TestClient(app) as c:
            yield c


def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_predict_valid_input(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "risk_class" in data
    assert "risk_label" in data
    assert "probabilities" in data
    assert "recommendation" in data


def test_predict_risk_class_values(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["risk_class"] in [0, 1, 2]


def test_predict_risk_label_mapping(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["risk_label"] == "Risco Médio"


def test_predict_probabilities_sum_to_one(client):
    response = client.post("/predict", json=VALID_PAYLOAD)
    probas = list(response.json()["probabilities"].values())
    assert abs(sum(probas) - 1.0) < 1e-4


def test_predict_missing_required_field(client):
    payload = VALID_PAYLOAD.copy()
    del payload["INDE_22"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_fase(client):
    payload = {**VALID_PAYLOAD, "Fase": 99}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_inde_above_max(client):
    payload = {**VALID_PAYLOAD, "INDE_22": 15.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_sem_risco(client, mock_model):
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["risk_class"] == 0
    assert data["risk_label"] == "Sem Risco"


def test_predict_alto_risco(client, mock_model):
    mock_model.predict.return_value = np.array([2])
    mock_model.predict_proba.return_value = np.array([[0.05, 0.15, 0.80]])
    response = client.post("/predict", json=VALID_PAYLOAD)
    data = response.json()
    assert data["risk_class"] == 2
    assert data["risk_label"] == "Alto Risco"


def test_predict_optional_notas_null(client):
    payload = {**VALID_PAYLOAD, "Matem": None, "Portug": None}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
