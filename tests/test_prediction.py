"""Testes do serviço de inferência."""

from typing import Any

import numpy as np
import pytest

from churn_prediction.modeling.predict import (
    ChurnPredictor,
)


def test_transform_generates_expected_shape(
    predictor: ChurnPredictor,
    customer_payload: dict[str, Any],
) -> None:
    """Deve gerar 45 features para um cliente."""
    transformed = predictor.transform(
        customer_payload
    )

    assert transformed.shape == (1, 45)
    assert transformed.dtype == np.float32
    assert np.isfinite(transformed).all()


def test_predict_proba_returns_valid_probability(
    predictor: ChurnPredictor,
    customer_payload: dict[str, Any],
) -> None:
    """Deve retornar uma probabilidade válida."""
    probabilities = predictor.predict_proba(
        customer_payload
    )

    assert probabilities.shape == (1,)
    assert 0 <= probabilities[0] <= 1

    assert probabilities[0] == pytest.approx(
        0.592241,
        abs=1e-5,
    )


def test_predict_returns_expected_class(
    predictor: ChurnPredictor,
    customer_payload: dict[str, Any],
) -> None:
    """Deve classificar o cliente como churn."""
    probabilities, predictions = predictor.predict(
        customer_payload
    )

    assert probabilities.shape == (1,)
    assert predictions.shape == (1,)
    assert predictions[0] == 1


def test_custom_threshold_changes_prediction(
    predictor: ChurnPredictor,
    customer_payload: dict[str, Any],
) -> None:
    """Deve respeitar um threshold customizado."""
    _, predictions = predictor.predict(
        customer_payload,
        threshold=0.70,
    )

    assert predictions[0] == 0


def test_missing_feature_raises_error(
    predictor: ChurnPredictor,
    customer_payload: dict[str, Any],
) -> None:
    """Deve rejeitar entrada com variável ausente."""
    invalid_payload = customer_payload.copy()

    invalid_payload.pop(
        "Contract"
    )

    with pytest.raises(
        ValueError,
        match="Variáveis ausentes",
    ):
        predictor.predict(
            invalid_payload
        )


@pytest.mark.parametrize(
    "invalid_threshold",
    [
        -0.01,
        1.01,
    ],
)
def test_invalid_threshold_raises_error(
    invalid_threshold: float,
) -> None:
    """Deve rejeitar threshold fora do intervalo."""
    with pytest.raises(
        ValueError,
        match="entre 0 e 1",
    ):
        ChurnPredictor(
            threshold=invalid_threshold
        )