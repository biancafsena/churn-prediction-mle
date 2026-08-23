"""Testes das funções de pré-processamento."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer

from churn_prediction.features.preprocessing import (
    RAW_FEATURES,
    build_preprocessor,
    prepare_feature_frame,
)


def test_prepare_feature_frame_returns_expected_columns(
    customer_payload: dict[str, Any],
) -> None:
    """Deve retornar as 19 variáveis na ordem correta."""
    dataframe = prepare_feature_frame(
        customer_payload
    )

    assert isinstance(
        dataframe,
        pd.DataFrame,
    )

    assert dataframe.shape == (1, 19)
    assert dataframe.columns.tolist() == RAW_FEATURES


def test_prepare_feature_frame_converts_total_charges(
    customer_payload: dict[str, Any],
) -> None:
    """Deve converter TotalCharges vazio em ausente."""
    payload = {
        **customer_payload,
        "TotalCharges": " ",
    }

    dataframe = prepare_feature_frame(
        payload
    )

    assert pd.isna(
        dataframe.loc[
            0,
            "TotalCharges",
        ]
    )


def test_prepare_feature_frame_rejects_missing_column(
    customer_payload: dict[str, Any],
) -> None:
    """Deve rejeitar uma variável ausente."""
    payload = customer_payload.copy()

    payload.pop(
        "PaymentMethod"
    )

    with pytest.raises(
        ValueError,
        match="Variáveis ausentes",
    ):
        prepare_feature_frame(
            payload
        )


def test_prepare_feature_frame_rejects_empty_list() -> None:
    """Deve rejeitar uma lista vazia."""
    with pytest.raises(
        ValueError,
        match="não pode estar vazia",
    ):
        prepare_feature_frame([])


def test_build_preprocessor_transforms_records(
    customer_payload: dict[str, Any],
) -> None:
    """Deve ajustar e transformar registros válidos."""
    second_customer = {
        **customer_payload,
        "gender": "Male",
        "Partner": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "MonthlyCharges": 70.00,
        "TotalCharges": " ",
    }

    dataframe = prepare_feature_frame([
        customer_payload,
        second_customer,
    ])

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(
        dataframe
    )

    assert isinstance(
        preprocessor,
        ColumnTransformer,
    )

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 4
    assert np.isfinite(transformed).all()