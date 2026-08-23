"""Testes das funções de avaliação dos modelos."""

import pandas as pd
import pytest

from churn_prediction.modeling.evaluate import (
    calculate_relative_cost,
    evaluate_binary_classification,
    evaluate_thresholds,
    validate_evaluation_inputs,
)

Y_TRUE = [0, 0, 0, 1, 1, 1]
PROBABILITIES = [
    0.10,
    0.30,
    0.70,
    0.40,
    0.80,
    0.90,
]


def test_evaluate_binary_classification() -> None:
    """Valida as métricas da classificação binária."""
    metrics = evaluate_binary_classification(
        y_true=Y_TRUE,
        probabilities=PROBABILITIES,
        threshold=0.50,
    )

    assert metrics["threshold"] == 0.50
    assert metrics["accuracy"] == pytest.approx(
        2 / 3
    )
    assert metrics["precision"] == pytest.approx(
        2 / 3
    )
    assert metrics["recall"] == pytest.approx(
        2 / 3
    )
    assert metrics["f1_score"] == pytest.approx(
        2 / 3
    )
    assert metrics["roc_auc"] == pytest.approx(
        8 / 9
    )
    assert metrics["average_precision"] == pytest.approx(
        11 / 12
    )
    assert metrics["true_negatives"] == 2
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 2


def test_calculate_relative_cost() -> None:
    """Valida o custo relativo de FP e FN."""
    cost = calculate_relative_cost(
        false_positives=117,
        false_negatives=171,
        false_positive_cost=1,
        false_negative_cost=5,
    )

    assert cost == 972.0


def test_evaluate_thresholds() -> None:
    """Valida a comparação entre thresholds."""
    result = evaluate_thresholds(
        y_true=Y_TRUE,
        probabilities=PROBABILITIES,
        thresholds=[0.70, 0.30, 0.50],
    )

    assert isinstance(result, pd.DataFrame)
    assert result["threshold"].tolist() == [
        0.30,
        0.50,
        0.70,
    ]
    assert result.loc[
        0,
        "relative_total_cost",
    ] == 2.0

    assert result.loc[
        1,
        "relative_total_cost",
    ] == 6.0


def test_rejects_invalid_threshold() -> None:
    """Valida a rejeição de threshold inválido."""
    with pytest.raises(
        ValueError,
        match="threshold deve estar entre 0 e 1",
    ):
        evaluate_binary_classification(
            y_true=Y_TRUE,
            probabilities=PROBABILITIES,
            threshold=1.0,
        )


def test_rejects_different_input_sizes() -> None:
    """Valida entradas com quantidades diferentes."""
    with pytest.raises(
        ValueError,
        match="mesma quantidade de observações",
    ):
        validate_evaluation_inputs(
            y_true=[0, 1],
            probabilities=[0.20],
            threshold=0.50,
        )


def test_rejects_invalid_probability() -> None:
    """Valida a rejeição de probabilidade inválida."""
    with pytest.raises(
        ValueError,
        match="probabilidades devem estar entre 0 e 1",
    ):
        evaluate_binary_classification(
            y_true=[0, 1],
            probabilities=[0.20, 1.10],
            threshold=0.50,
        )


def test_rejects_single_class() -> None:
    """Valida a exigência das duas classes."""
    with pytest.raises(
        ValueError,
        match="deve possuir as duas classes",
    ):
        evaluate_binary_classification(
            y_true=[0, 0, 0],
            probabilities=[0.10, 0.20, 0.30],
            threshold=0.50,
        )


def test_rejects_negative_error_cost() -> None:
    """Valida a rejeição de custos negativos."""
    with pytest.raises(
        ValueError,
        match="custos dos erros não podem ser negativos",
    ):
        calculate_relative_cost(
            false_positives=1,
            false_negatives=2,
            false_positive_cost=-1,
            false_negative_cost=5,
        )