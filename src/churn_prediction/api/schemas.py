"""Schemas de entrada e saída da API de previsão de churn."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CustomerFeatures(BaseModel):
    """Características de um cliente para previsão de churn."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
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
            ]
        },
    )

    gender: Literal[
        "Female",
        "Male",
    ]

    SeniorCitizen: Literal[
        0,
        1,
    ]

    Partner: Literal[
        "No",
        "Yes",
    ]

    Dependents: Literal[
        "No",
        "Yes",
    ]

    tenure: int = Field(
        ge=0,
        le=72,
    )

    PhoneService: Literal[
        "No",
        "Yes",
    ]

    MultipleLines: Literal[
        "No",
        "No phone service",
        "Yes",
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No",
    ]

    OnlineSecurity: Literal[
        "No",
        "No internet service",
        "Yes",
    ]

    OnlineBackup: Literal[
        "No",
        "No internet service",
        "Yes",
    ]

    DeviceProtection: Literal[
        "No",
        "No internet service",
        "Yes",
    ]

    TechSupport: Literal[
        "No",
        "No internet service",
        "Yes",
    ]

    StreamingTV: Literal[
        "No",
        "No internet service",
        "Yes",
    ]

    StreamingMovies: Literal[
        "No",
        "No internet service",
        "Yes",
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year",
    ]

    PaperlessBilling: Literal[
        "No",
        "Yes",
    ]

    PaymentMethod: Literal[
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    ]

    MonthlyCharges: float = Field(
        ge=0,
    )

    TotalCharges: float | None = Field(
        default=None,
        ge=0,
    )


class PredictionResponse(BaseModel):
    """Resposta da previsão de churn."""

    model_config = ConfigDict(
        extra="forbid"
    )

    churn_probability: float = Field(
        ge=0,
        le=1,
    )

    churn_prediction: Literal[
        0,
        1,
    ]

    churn_label: Literal[
        "Não Churn",
        "Churn",
    ]

    threshold: float = Field(
        ge=0,
        le=1,
    )

    model_name: str
    model_version: str

    processing_time_ms: float = Field(
        ge=0,
    )


class HealthResponse(BaseModel):
    """Resposta do endpoint de saúde da aplicação."""

    model_config = ConfigDict(
        extra="forbid"
    )

    status: Literal[
        "healthy",
        "unhealthy",
    ]

    model_loaded: bool
    model_name: str
    api_version: str