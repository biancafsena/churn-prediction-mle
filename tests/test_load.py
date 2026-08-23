"""Testes do carregamento e da preparação dos dados."""

from pathlib import Path

import pandas as pd
import pytest

from churn_prediction.data.load import (
    load_raw_data,
    prepare_supervised_data,
)


def build_test_dataframe() -> pd.DataFrame:
    """Cria um dataset mínimo válido para os testes."""
    return pd.DataFrame(
        [
            {
                "customerID": "0001-A",
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
                "TotalCharges": "29.85",
                "Churn": "No",
            },
            {
                "customerID": "0002-B",
                "gender": "Male",
                "SeniorCitizen": 1,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 2,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 70.70,
                "TotalCharges": " ",
                "Churn": "Yes",
            },
        ]
    )


def save_test_dataset(
    dataframe: pd.DataFrame,
    file_path: Path,
) -> Path:
    """Salva um dataset temporário para os testes."""
    dataframe.to_csv(
        file_path,
        index=False,
    )

    return file_path


def test_load_raw_data_success(
    tmp_path: Path,
) -> None:
    """Valida o carregamento de um dataset válido."""
    file_path = save_test_dataset(
        build_test_dataframe(),
        tmp_path / "churn.csv",
    )

    dataframe = load_raw_data(file_path)

    assert dataframe.shape == (2, 21)
    assert dataframe["customerID"].is_unique
    assert set(dataframe["Churn"]) == {"No", "Yes"}


def test_load_raw_data_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Valida a rejeição de um arquivo inexistente."""
    with pytest.raises(
        FileNotFoundError,
        match="Dataset não encontrado",
    ):
        load_raw_data(
            tmp_path / "missing.csv"
        )


def test_load_raw_data_rejects_missing_column(
    tmp_path: Path,
) -> None:
    """Valida a rejeição de colunas obrigatórias ausentes."""
    dataframe = build_test_dataframe().drop(
        columns=["MonthlyCharges"]
    )

    file_path = save_test_dataset(
        dataframe,
        tmp_path / "missing_column.csv",
    )

    with pytest.raises(
        ValueError,
        match="Colunas obrigatórias ausentes",
    ):
        load_raw_data(file_path)


def test_load_raw_data_rejects_duplicate_customer(
    tmp_path: Path,
) -> None:
    """Valida a rejeição de identificadores duplicados."""
    dataframe = build_test_dataframe()
    dataframe.loc[1, "customerID"] = "0001-A"

    file_path = save_test_dataset(
        dataframe,
        tmp_path / "duplicated.csv",
    )

    with pytest.raises(
        ValueError,
        match="identificadores de clientes duplicados",
    ):
        load_raw_data(file_path)


def test_load_raw_data_rejects_invalid_target(
    tmp_path: Path,
) -> None:
    """Valida a rejeição de uma classe desconhecida."""
    dataframe = build_test_dataframe()
    dataframe.loc[1, "Churn"] = "Unknown"

    file_path = save_test_dataset(
        dataframe,
        tmp_path / "invalid_target.csv",
    )

    with pytest.raises(
        ValueError,
        match="Valores inválidos",
    ):
        load_raw_data(file_path)


def test_prepare_supervised_data() -> None:
    """Valida a separação das features e do target."""
    dataframe = build_test_dataframe()

    features, target = prepare_supervised_data(
        dataframe
    )

    assert features.shape == (2, 19)
    assert target.tolist() == [0, 1]
    assert target.dtype == "int64"
    assert pd.isna(
        features.loc[1, "TotalCharges"]
    )
    assert "customerID" not in features.columns
    assert "Churn" not in features.columns