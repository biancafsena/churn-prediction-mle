"""Treinamento reproduzível da MLP PyTorch."""

import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from churn_prediction.modeling.model import ChurnMLP


@dataclass(frozen=True)
class TrainingConfig:
    """Configurações utilizadas no treinamento."""

    batch_size: int = 32
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    max_epochs: int = 300
    patience: int = 20
    min_delta: float = 0.0001
    random_state: int = 42

    def __post_init__(self) -> None:
        """Valida as configurações de treinamento."""
        if self.batch_size <= 0:
            raise ValueError(
                "batch_size deve ser maior que zero."
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate deve ser maior que zero."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay não pode ser negativo."
            )

        if self.max_epochs <= 0:
            raise ValueError(
                "max_epochs deve ser maior que zero."
            )

        if self.patience <= 0:
            raise ValueError(
                "patience deve ser maior que zero."
            )

        if self.min_delta < 0:
            raise ValueError(
                "min_delta não pode ser negativo."
            )


@dataclass
class TrainingResult:
    """Resultados produzidos pelo treinamento."""

    model: ChurnMLP
    training_losses: list[float]
    validation_losses: list[float]
    best_epoch: int
    best_validation_loss: float
    epochs_completed: int
    early_stopping_triggered: bool


def configure_reproducibility(
    random_state: int = 42,
) -> None:
    """Configura as sementes e os algoritmos determinísticos."""
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            random_state
        )

    torch.use_deterministic_algorithms(
        True,
        warn_only=True,
    )


def validate_training_arrays(
    features: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Valida e padroniza arrays utilizados no treinamento."""
    features_array = np.asarray(
        features,
        dtype=np.float32,
    )

    target_array = np.asarray(
        target,
        dtype=np.float32,
    ).reshape(-1, 1)

    if features_array.ndim != 2:
        raise ValueError(
            "As features devem possuir duas dimensões."
        )

    if features_array.shape[0] == 0:
        raise ValueError(
            "As features não podem estar vazias."
        )

    if features_array.shape[0] != target_array.shape[0]:
        raise ValueError(
            "Features e target devem possuir a mesma "
            "quantidade de observações."
        )

    if not np.isfinite(features_array).all():
        raise ValueError(
            "As features devem conter somente valores finitos."
        )

    if not np.isfinite(target_array).all():
        raise ValueError(
            "O target deve conter somente valores finitos."
        )

    target_values = set(
        np.unique(target_array).tolist()
    )

    if not target_values.issubset(
        {0.0, 1.0}
    ):
        raise ValueError(
            "O target deve conter somente as classes 0 e 1."
        )

    return features_array, target_array


def create_data_loader(
    features: np.ndarray,
    target: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    random_state: int = 42,
) -> DataLoader:
    """Cria um DataLoader a partir de arrays NumPy."""
    if batch_size <= 0:
        raise ValueError(
            "batch_size deve ser maior que zero."
        )

    features_array, target_array = (
        validate_training_arrays(
            features=features,
            target=target,
        )
    )

    dataset = TensorDataset(
        torch.from_numpy(features_array),
        torch.from_numpy(target_array),
    )

    generator = torch.Generator()
    generator.manual_seed(random_state)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


def calculate_training_epoch_loss(
    model: ChurnMLP,
    data_loader: DataLoader,
    loss_function: nn.Module,
    optimizer: Adam,
    device: torch.device,
) -> float:
    """Executa uma época de treinamento."""
    model.train()

    accumulated_loss = 0.0
    observations = 0

    for features, target in data_loader:
        features = features.to(device)
        target = target.to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        logits = model(features)
        loss = loss_function(
            logits,
            target,
        )

        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]

        accumulated_loss += (
            loss.item()
            * batch_size
        )

        observations += batch_size

    return accumulated_loss / observations


def calculate_validation_loss(
    model: ChurnMLP,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    """Calcula a loss no conjunto de validação."""
    model.eval()

    accumulated_loss = 0.0
    observations = 0

    with torch.inference_mode():
        for features, target in data_loader:
            features = features.to(device)
            target = target.to(device)

            logits = model(features)
            loss = loss_function(
                logits,
                target,
            )

            batch_size = features.shape[0]

            accumulated_loss += (
                loss.item()
                * batch_size
            )

            observations += batch_size

    return accumulated_loss / observations


def train_model(
    model: ChurnMLP,
    training_loader: DataLoader,
    validation_loader: DataLoader,
    config: TrainingConfig | None = None,
    device: torch.device | None = None,
) -> TrainingResult:
    """Treina a MLP utilizando Early Stopping."""
    selected_config = (
        config
        if config is not None
        else TrainingConfig()
    )

    selected_device = (
        device
        if device is not None
        else torch.device("cpu")
    )

    configure_reproducibility(
        selected_config.random_state
    )

    model.to(selected_device)

    loss_function = nn.BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        lr=selected_config.learning_rate,
        weight_decay=selected_config.weight_decay,
    )

    training_losses: list[float] = []
    validation_losses: list[float] = []

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = deepcopy(
        model.state_dict()
    )

    early_stopping_triggered = False

    for epoch in range(
        1,
        selected_config.max_epochs + 1,
    ):
        training_loss = (
            calculate_training_epoch_loss(
                model=model,
                data_loader=training_loader,
                loss_function=loss_function,
                optimizer=optimizer,
                device=selected_device,
            )
        )

        validation_loss = (
            calculate_validation_loss(
                model=model,
                data_loader=validation_loader,
                loss_function=loss_function,
                device=selected_device,
            )
        )

        training_losses.append(
            training_loss
        )

        validation_losses.append(
            validation_loss
        )

        improvement = (
            best_validation_loss
            - validation_loss
        )

        if improvement > selected_config.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(
                model.state_dict()
            )
            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= selected_config.patience
        ):
            early_stopping_triggered = True
            break

    model.load_state_dict(
        best_state
    )

    model.to(selected_device)
    model.eval()

    return TrainingResult(
        model=model,
        training_losses=training_losses,
        validation_losses=validation_losses,
        best_epoch=best_epoch,
        best_validation_loss=float(
            best_validation_loss
        ),
        epochs_completed=len(
            training_losses
        ),
        early_stopping_triggered=(
            early_stopping_triggered
        ),
    )


def predict_probabilities(
    model: ChurnMLP,
    data_loader: DataLoader,
    device: torch.device | None = None,
) -> np.ndarray:
    """Gera probabilidades para um DataLoader."""
    selected_device = (
        device
        if device is not None
        else torch.device("cpu")
    )

    model.to(selected_device)
    model.eval()

    probability_batches: list[np.ndarray] = []

    with torch.inference_mode():
        for features, _ in data_loader:
            features = features.to(
                selected_device
            )

            probabilities = torch.sigmoid(
                model(features)
            )

            probability_batches.append(
                probabilities
                .cpu()
                .numpy()
                .reshape(-1)
            )

    return np.concatenate(
        probability_batches
    )