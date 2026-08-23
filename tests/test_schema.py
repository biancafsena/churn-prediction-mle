"""Testes do schema Pandera do dataset de churn."""

import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from churn_prediction.data.schema import (
    validate_raw_dataset,
)
from tests.test_load import (
    build_test_dataframe,
)


def test_validate_raw_dataset() -> None:
    """Valida um dataset compatível com o schema."""
    dataframe = build_test_dataframe()

    validated = validate_raw_dataset(
        dataframe
    )

    assert isinstance(
        validated,
        pd.DataFrame,
    )
    assert validated.shape == (2, 21)


def test_schema_rejects_invalid_tenure() -> None:
    """Valida a rejeição de tenure fora do domínio."""
    dataframe = build_test_dataframe()
    dataframe.loc[0, "tenure"] = 100

    with pytest.raises(
        SchemaErrors
    ):
        validate_raw_dataset(
            dataframe
        )


def test_schema_rejects_invalid_category() -> None:
    """Valida a rejeição de categoria desconhecida."""
    dataframe = build_test_dataframe()
    dataframe.loc[
        0,
        "InternetService",
    ] = "Satellite"

    with pytest.raises(
        SchemaErrors
    ):
        validate_raw_dataset(
            dataframe
        )


def test_schema_rejects_extra_column() -> None:
    """Valida a rejeição de coluna não declarada."""
    dataframe = build_test_dataframe()
    dataframe["unexpected_column"] = "value"

    with pytest.raises(
        SchemaErrors
    ):
        validate_raw_dataset(
            dataframe
        )