"""Fixtures compartilhadas pelos testes."""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from churn_prediction.api.main import app
from churn_prediction.modeling.predict import (
    ChurnPredictor,
)


@pytest.fixture(scope="session")
def customer_payload() -> dict[str, Any]:
    """Retorna um cliente válido para os testes."""
    return {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    }


@pytest.fixture(scope="session")
def predictor() -> ChurnPredictor:
    """Carrega o preditor uma única vez por sessão."""
    return ChurnPredictor()


@pytest.fixture(scope="session")
def api_client() -> Generator[TestClient, None, None]:
    """Inicializa um cliente HTTP para os testes."""
    with TestClient(app) as client:
        yield client