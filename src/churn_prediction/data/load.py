"""Carregamento e preparação dos dados de churn."""

import logging
from pathlib import Path

import pandas as pd

from churn_prediction.config import (
    IDENTIFIER_COLUMN,
    RAW_DATA_PATH,
    TARGET_COLUMN,
)
from churn_prediction.data.schema import (
    validate_raw_dataset,
)
from churn_prediction.features.preprocessing import (
    RAW_FEATURES,
    prepare_feature_frame,
)

logger = logging.getLogger(__name__)

EXPECTED_DATASET_COLUMNS = [
    IDENTIFIER_COLUMN,
    *RAW_FEATURES,
    TARGET_COLUMN,
]

VALID_TARGET_VALUES = {
    "No",
    "Yes",
}


def load_raw_data(
    file_path: str | Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Carrega e valida o dataset bruto de churn."""
    path = Path(file_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"O caminho informado não é um arquivo: {path}"
        )

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(
            "O dataset carregado está vazio."
        )

    missing_columns = sorted(
        set(EXPECTED_DATASET_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes no dataset: "
            f"{missing_columns}"
        )

    duplicated_identifiers = int(
        dataframe[IDENTIFIER_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicated_identifiers:
        raise ValueError(
            "Foram encontrados "
            f"{duplicated_identifiers} identificadores "
            "de clientes duplicados."
        )

    target_values = set(
        dataframe[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    invalid_target_values = sorted(
        target_values
        - VALID_TARGET_VALUES
    )

    if invalid_target_values:
        raise ValueError(
            "Valores inválidos encontrados na variável "
            f"{TARGET_COLUMN}: {invalid_target_values}"
        )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"A variável {TARGET_COLUMN} possui valores ausentes."
        )

    validated_dataframe = validate_raw_dataset(
        dataframe
    )
    
    logger.info(       
        "Dataset carregado e validado com sucesso.",   
        extra={   
            "dataset_path": str(path),
            "rows": len(validated_dataframe),
            "columns": len(
                validated_dataframe.columns
            ),
            "schema": "telco_customer_churn",
        },
    )

    return validated_dataframe


def prepare_supervised_data(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa e prepara as variáveis explicativas e o target."""
    missing_columns = sorted(
        set(EXPECTED_DATASET_COLUMNS)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes para preparação: "
            f"{missing_columns}"
        )

    features = prepare_feature_frame(
        dataframe[RAW_FEATURES]
    )

    target = dataframe[TARGET_COLUMN].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    if target.isna().any():
        invalid_values = sorted(
            dataframe.loc[
                target.isna(),
                TARGET_COLUMN,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Não foi possível converter os valores do target: "
            f"{invalid_values}"
        )

    target = target.astype("int64")
    target.name = TARGET_COLUMN

    logger.info(
        "Dados supervisionados preparados com sucesso.",
        extra={
            "rows": len(features),
            "features": len(features.columns),
            "positive_class": int(target.sum()),
            "negative_class": int((target == 0).sum()),
        },
    )

    return features, target