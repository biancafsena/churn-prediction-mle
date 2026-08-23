"""Arquitetura da rede neural utilizada na previsão de churn."""

import torch
from torch import nn


class ChurnMLP(nn.Module):
    """MLP para classificação binária de churn."""

    def __init__(
        self,
        input_size: int,
        hidden_size_1: int = 64,
        hidden_size_2: int = 32,
        dropout_rate: float = 0.20,
    ) -> None:
        """Inicializa a arquitetura da rede neural."""
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "input_size deve ser maior que zero."
            )

        if hidden_size_1 <= 0 or hidden_size_2 <= 0:
            raise ValueError(
                "As camadas ocultas devem possuir "
                "pelo menos um neurônio."
            )

        if not 0 <= dropout_rate < 1:
            raise ValueError(
                "dropout_rate deve estar no intervalo [0, 1)."
            )

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_size_1,
            ),
            nn.ReLU(),
            nn.Dropout(
                p=dropout_rate,
            ),
            nn.Linear(
                hidden_size_1,
                hidden_size_2,
            ),
            nn.ReLU(),
            nn.Dropout(
                p=dropout_rate,
            ),
            nn.Linear(
                hidden_size_2,
                1,
            ),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula os logits de churn."""
        return self.network(features)