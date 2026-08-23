"""Avaliação de modelos de classificação binária."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def validate_evaluation_inputs(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: (
        Sequence[float]
        | np.ndarray
        | pd.Series
    ),
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida e padroniza as entradas da avaliação."""
    if not 0 < threshold < 1:
        raise ValueError(
            "O threshold deve estar entre 0 e 1."
        )

    y_true_array = np.asarray(
        y_true,
        dtype=np.int64,
    ).reshape(-1)

    probability_array = np.asarray(
        probabilities,
        dtype=np.float64,
    ).reshape(-1)

    if y_true_array.size == 0:
        raise ValueError(
            "A variável y_true não pode estar vazia."
        )

    if probability_array.size == 0:
        raise ValueError(
            "As probabilidades não podem estar vazias."
        )

    if y_true_array.shape[0] != probability_array.shape[0]:
        raise ValueError(
            "y_true e probabilities devem possuir "
            "a mesma quantidade de observações."
        )

    unique_classes = set(
        np.unique(y_true_array).tolist()
    )

    if not unique_classes.issubset({0, 1}):
        raise ValueError(
            "y_true deve conter somente as classes 0 e 1."
        )

    if len(unique_classes) < 2:
        raise ValueError(
            "y_true deve possuir as duas classes para "
            "o cálculo completo das métricas."
        )

    if not np.isfinite(probability_array).all():
        raise ValueError(
            "As probabilidades devem ser valores finitos."
        )

    if (
        (probability_array < 0).any()
        or (probability_array > 1).any()
    ):
        raise ValueError(
            "As probabilidades devem estar entre 0 e 1."
        )

    return y_true_array, probability_array


def evaluate_binary_classification(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: (
        Sequence[float]
        | np.ndarray
        | pd.Series
    ),
    threshold: float = 0.50,
) -> dict[str, float | int]:
    """Calcula métricas para uma classificação binária."""
    y_true_array, probability_array = (
        validate_evaluation_inputs(
            y_true=y_true,
            probabilities=probabilities,
            threshold=threshold,
        )
    )

    predictions = (
        probability_array >= threshold
    ).astype(np.int64)

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            y_true_array,
            predictions,
            labels=[0, 1],
        ).ravel()
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(
                y_true_array,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true_array,
                predictions,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true_array,
                probability_array,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true_array,
                probability_array,
            )
        ),
        "true_negatives": int(true_negative),
        "false_positives": int(false_positive),
        "false_negatives": int(false_negative),
        "true_positives": int(true_positive),
    }


def calculate_relative_cost(
    false_positives: int,
    false_negatives: int,
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
) -> float:
    """Calcula o custo relativo dos erros."""
    if false_positives < 0 or false_negatives < 0:
        raise ValueError(
            "As quantidades de erros não podem ser negativas."
        )

    if false_positive_cost < 0 or false_negative_cost < 0:
        raise ValueError(
            "Os custos dos erros não podem ser negativos."
        )

    return float(
        false_positives * false_positive_cost
        + false_negatives * false_negative_cost
    )


def evaluate_thresholds(
    y_true: Sequence[int] | np.ndarray | pd.Series,
    probabilities: (
        Sequence[float]
        | np.ndarray
        | pd.Series
    ),
    thresholds: Sequence[float],
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 5.0,
) -> pd.DataFrame:
    """Compara métricas e custos em diferentes thresholds."""
    if not thresholds:
        raise ValueError(
            "Ao menos um threshold deve ser informado."
        )

    results: list[dict[str, float | int]] = []

    for threshold in thresholds:
        metrics = evaluate_binary_classification(
            y_true=y_true,
            probabilities=probabilities,
            threshold=float(threshold),
        )

        metrics["relative_total_cost"] = (
            calculate_relative_cost(
                false_positives=int(
                    metrics["false_positives"]
                ),
                false_negatives=int(
                    metrics["false_negatives"]
                ),
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            )
        )

        results.append(metrics)

    return (
        pd.DataFrame(results)
        .sort_values("threshold")
        .reset_index(drop=True)
    )