"""Testes do treinamento da MLP PyTorch."""

import numpy as np
import pytest
import torch

from churn_prediction.modeling.model import ChurnMLP
from churn_prediction.modeling.train import (
    TrainingConfig,
    configure_reproducibility,
    create_data_loader,
    predict_probabilities,
    train_model,
    validate_training_arrays,
)


def build_training_data() -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """Cria dados sintéticos reproduzíveis."""
    generator = np.random.default_rng(42)

    features = generator.normal(
        size=(48, 6)
    ).astype(np.float32)

    target = (
        features[:, 0]
        + features[:, 1]
        > 0
    ).astype(np.float32)

    return features, target


def build_model() -> ChurnMLP:
    """Cria uma MLP pequena para os testes."""
    return ChurnMLP(
        input_size=6,
        hidden_size_1=8,
        hidden_size_2=4,
        dropout_rate=0.10,
    )


def test_training_config_rejects_invalid_batch_size() -> None:
    """Valida a rejeição de batch size inválido."""
    with pytest.raises(
        ValueError,
        match="batch_size deve ser maior que zero",
    ):
        TrainingConfig(
            batch_size=0
        )


def test_validate_training_arrays() -> None:
    """Valida a padronização dos arrays."""
    features, target = build_training_data()

    validated_features, validated_target = (
        validate_training_arrays(
            features=features,
            target=target,
        )
    )

    assert validated_features.dtype == np.float32
    assert validated_target.dtype == np.float32
    assert validated_features.shape == (48, 6)
    assert validated_target.shape == (48, 1)


def test_validate_training_arrays_rejects_invalid_target() -> None:
    """Valida a rejeição de classes inválidas."""
    features, target = build_training_data()
    target[0] = 2

    with pytest.raises(
        ValueError,
        match="classes 0 e 1",
    ):
        validate_training_arrays(
            features=features,
            target=target,
        )


def test_create_data_loader() -> None:
    """Valida os formatos produzidos pelo DataLoader."""
    features, target = build_training_data()

    data_loader = create_data_loader(
        features=features,
        target=target,
        batch_size=16,
        shuffle=False,
    )

    feature_batch, target_batch = next(
        iter(data_loader)
    )

    assert len(data_loader) == 3
    assert feature_batch.shape == (16, 6)
    assert target_batch.shape == (16, 1)
    assert feature_batch.dtype == torch.float32
    assert target_batch.dtype == torch.float32


def test_train_model() -> None:
    """Valida a execução do treinamento."""
    configure_reproducibility(42)

    features, target = build_training_data()

    training_loader = create_data_loader(
        features=features[:36],
        target=target[:36],
        batch_size=12,
        shuffle=True,
    )

    validation_loader = create_data_loader(
        features=features[36:],
        target=target[36:],
        batch_size=12,
        shuffle=False,
    )

    result = train_model(
        model=build_model(),
        training_loader=training_loader,
        validation_loader=validation_loader,
        config=TrainingConfig(
            batch_size=12,
            max_epochs=5,
            patience=3,
            min_delta=0.0001,
        ),
    )

    assert result.epochs_completed <= 5
    assert len(result.training_losses) == (
        result.epochs_completed
    )
    assert len(result.validation_losses) == (
        result.epochs_completed
    )
    assert result.best_epoch >= 1
    assert np.isfinite(
        result.best_validation_loss
    )
    assert result.model.training is False


def test_early_stopping() -> None:
    """Valida o acionamento do Early Stopping."""
    configure_reproducibility(42)

    features, target = build_training_data()

    training_loader = create_data_loader(
        features=features[:36],
        target=target[:36],
        batch_size=12,
        shuffle=False,
    )

    validation_loader = create_data_loader(
        features=features[36:],
        target=target[36:],
        batch_size=12,
        shuffle=False,
    )

    result = train_model(
        model=build_model(),
        training_loader=training_loader,
        validation_loader=validation_loader,
        config=TrainingConfig(
            batch_size=12,
            learning_rate=0.000001,
            max_epochs=10,
            patience=2,
            min_delta=10.0,
        ),
    )

    assert result.early_stopping_triggered is True
    assert result.epochs_completed == 3
    assert result.best_epoch == 1


def test_predict_probabilities() -> None:
    """Valida as probabilidades produzidas pela MLP."""
    features, target = build_training_data()

    data_loader = create_data_loader(
        features=features,
        target=target,
        batch_size=16,
        shuffle=False,
    )

    probabilities = predict_probabilities(
        model=build_model(),
        data_loader=data_loader,
    )

    assert probabilities.shape == (48,)
    assert np.isfinite(probabilities).all()
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()