"""Construção e validação do pré-processamento dos dados."""

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RAW_FEATURES = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


def prepare_feature_frame(
    records: (
        dict[str, Any]
        | list[dict[str, Any]]
        | pd.DataFrame
    ),
) -> pd.DataFrame:
    """Converte e valida registros brutos."""
    if isinstance(records, pd.DataFrame):
        dataframe = records.copy()

    elif isinstance(records, dict):
        dataframe = pd.DataFrame(
            [records]
        )

    elif isinstance(records, list):
        if not records:
            raise ValueError(
                "A lista de registros não pode estar vazia."
            )

        dataframe = pd.DataFrame(
            records
        )

    else:
        raise TypeError(
            "A entrada deve ser um dicionário, "
            "uma lista de dicionários ou um DataFrame."
        )

    missing_columns = sorted(
        set(RAW_FEATURES)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Variáveis ausentes na entrada: "
            f"{missing_columns}"
        )

    dataframe = dataframe[
        RAW_FEATURES
    ].copy()

    for column in [
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe["TotalCharges"] = pd.to_numeric(
        dataframe["TotalCharges"],
        errors="coerce",
    )

    return dataframe


def build_preprocessor() -> ColumnTransformer:
    """Constrói o pipeline de pré-processamento."""
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )