"""Testes dos endpoints da API."""

from typing import Any

from fastapi.testclient import TestClient


def test_root_returns_application_information(
    api_client: TestClient,
) -> None:
    """Deve retornar as informações da aplicação."""
    response = api_client.get("/")

    assert response.status_code == 200

    response_data = response.json()

    assert (
        response_data["application"]
        == "Customer Churn Prediction API"
    )

    assert response_data["version"] == "0.1.0"
    assert response_data["health"] == "/health"
    assert response_data["prediction"] == "/predict"


def test_health_returns_loaded_model(
    api_client: TestClient,
) -> None:
    """Deve informar que o modelo está carregado."""
    response = api_client.get(
        "/health"
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data == {
        "status": "healthy",
        "model_loaded": True,
        "model_name": "ChurnMLP",
        "api_version": "0.1.0",
    }


def test_predict_returns_churn_prediction(
    api_client: TestClient,
    customer_payload: dict[str, Any],
) -> None:
    """Deve retornar uma previsão válida."""
    response = api_client.post(
        "/predict",
        json=customer_payload,
    )

    assert response.status_code == 200

    response_data = response.json()

    assert response_data[
        "churn_probability"
    ] > 0.50

    assert (
        response_data["churn_probability"]
        == 0.5922411680221558
    )

    assert response_data[
        "churn_prediction"
    ] == 1

    assert response_data[
        "churn_label"
    ] == "Churn"

    assert response_data["threshold"] == 0.50
    assert response_data["model_name"] == "ChurnMLP"
    assert response_data["model_version"] == "1.0.0"
    assert response_data["processing_time_ms"] >= 0


def test_predict_rejects_invalid_tenure(
    api_client: TestClient,
    customer_payload: dict[str, Any],
) -> None:
    """Deve rejeitar tenure acima do limite."""
    invalid_payload = {
        **customer_payload,
        "tenure": 100,
    }

    response = api_client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_predict_rejects_missing_feature(
    api_client: TestClient,
    customer_payload: dict[str, Any],
) -> None:
    """Deve rejeitar uma variável obrigatória ausente."""
    invalid_payload = (
        customer_payload.copy()
    )

    invalid_payload.pop(
        "Contract"
    )

    response = api_client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422


def test_predict_rejects_extra_feature(
    api_client: TestClient,
    customer_payload: dict[str, Any],
) -> None:
    """Deve rejeitar uma variável não prevista."""
    invalid_payload = {
        **customer_payload,
        "customerID": "TEST-0001",
    }

    response = api_client.post(
        "/predict",
        json=invalid_payload,
    )

    assert response.status_code == 422