"""Carregamento de artefatos e inferência da MLP PyTorch."""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from churn_prediction.config import (
    DEFAULT_THRESHOLD,
    EXPECTED_PROCESSED_FEATURES,
    MODEL_METADATA_PATH,
    MODEL_STATE_PATH,
    PREPROCESSOR_PATH,
)
from churn_prediction.modeling.model import ChurnMLP


class ChurnPredictor:
    """Serviço de inferência para previsão de churn."""

    def __init__(
        self,
        model_state_path: Path = MODEL_STATE_PATH,
        preprocessor_path: Path = PREPROCESSOR_PATH,
        metadata_path: Path = MODEL_METADATA_PATH,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        """Carrega o pré-processador, os metadados e a MLP."""
        self._validate_threshold(threshold)

        self.threshold = threshold
        self.device = torch.device("cpu")

        self.metadata = self._load_metadata(
            metadata_path
        )

        self.preprocessor = self._load_preprocessor(
            preprocessor_path
        )

        self.model = self._load_model(
            model_state_path
        )

        self.raw_feature_names = list(
            self.preprocessor.feature_names_in_
        )

    @staticmethod
    def _validate_threshold(
        threshold: float,
    ) -> None:
        """Valida o threshold de classificação."""
        if not 0 <= threshold <= 1:
            raise ValueError(
                "O threshold deve estar entre 0 e 1."
            )

    @staticmethod
    def _load_metadata(
        metadata_path: Path,
    ) -> dict[str, Any]:
        """Carrega os metadados do modelo."""
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadados não encontrados: {metadata_path}"
            )

        with metadata_path.open(
            mode="r",
            encoding="utf-8",
        ) as metadata_file:
            return json.load(metadata_file)

    @staticmethod
    def _load_preprocessor(
        preprocessor_path: Path,
    ) -> Any:
        """Carrega o pré-processador treinado."""
        if not preprocessor_path.exists():
            raise FileNotFoundError(
                "Pré-processador não encontrado: "
                f"{preprocessor_path}"
            )

        return joblib.load(
            preprocessor_path
        )

    def _load_model(
        self,
        model_state_path: Path,
    ) -> ChurnMLP:
        """Reconstrói a arquitetura e carrega os pesos."""
        if not model_state_path.exists():
            raise FileNotFoundError(
                f"Pesos não encontrados: {model_state_path}"
            )

        artifact = torch.load(
            model_state_path,
            map_location=self.device,
            weights_only=True,
        )

        model = ChurnMLP(
            input_size=artifact["input_size"],
            hidden_size_1=artifact["hidden_size_1"],
            hidden_size_2=artifact["hidden_size_2"],
            dropout_rate=artifact["dropout_rate"],
        )

        model.load_state_dict(
            artifact["model_state_dict"]
        )

        model.to(self.device)
        model.eval()

        return model

    def _prepare_dataframe(
        self,
        records: (
            dict[str, Any]
            | list[dict[str, Any]]
            | pd.DataFrame
        ),
    ) -> pd.DataFrame:
        """Converte e valida os registros de entrada."""
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
            set(self.raw_feature_names)
            - set(dataframe.columns)
        )

        if missing_columns:
            raise ValueError(
                "Variáveis ausentes na entrada: "
                f"{missing_columns}"
            )

        dataframe = dataframe[
            self.raw_feature_names
        ].copy()

        dataframe["TotalCharges"] = pd.to_numeric(
            dataframe["TotalCharges"],
            errors="coerce",
        )

        return dataframe

    def transform(
        self,
        records: (
            dict[str, Any]
            | list[dict[str, Any]]
            | pd.DataFrame
        ),
    ) -> np.ndarray:
        """Aplica o pré-processamento aos registros."""
        dataframe = self._prepare_dataframe(
            records
        )

        transformed = self.preprocessor.transform(
            dataframe
        )

        transformed_array = np.asarray(
            transformed,
            dtype=np.float32,
        )

        if (
            transformed_array.shape[1]
            != EXPECTED_PROCESSED_FEATURES
        ):
            raise ValueError(
                "Quantidade inesperada de features após "
                "o pré-processamento: "
                f"{transformed_array.shape[1]}"
            )

        return transformed_array

    def predict_proba(
        self,
        records: (
            dict[str, Any]
            | list[dict[str, Any]]
            | pd.DataFrame
        ),
    ) -> np.ndarray:
        """Retorna as probabilidades previstas de churn."""
        transformed = self.transform(
            records
        )

        features_tensor = torch.from_numpy(
            transformed
        ).to(self.device)

        with torch.inference_mode():
            logits = self.model(
                features_tensor
            )

            probabilities = torch.sigmoid(
                logits
            )

        return (
            probabilities
            .cpu()
            .numpy()
            .reshape(-1)
        )

    def predict(
        self,
        records: (
            dict[str, Any]
            | list[dict[str, Any]]
            | pd.DataFrame
        ),
        threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna probabilidades e classes previstas."""
        selected_threshold = (
            self.threshold
            if threshold is None
            else threshold
        )

        self._validate_threshold(
            selected_threshold
        )

        probabilities = self.predict_proba(
            records
        )

        predictions = (
            probabilities
            >= selected_threshold
        ).astype(np.int64)

        return probabilities, predictions