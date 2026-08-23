"""Schema Pandera para validação do dataset de churn."""

import pandas as pd
from pandera.pandas import (
    Check,
    Column,
    DataFrameSchema,
)

YES_NO_VALUES = [
    "No",
    "Yes",
]

INTERNET_SERVICE_VALUES = [
    "DSL",
    "Fiber optic",
    "No",
]

MULTIPLE_LINES_VALUES = [
    "No",
    "No phone service",
    "Yes",
]

INTERNET_OPTION_VALUES = [
    "No",
    "No internet service",
    "Yes",
]

CONTRACT_VALUES = [
    "Month-to-month",
    "One year",
    "Two year",
]

PAYMENT_METHOD_VALUES = [
    "Bank transfer (automatic)",
    "Credit card (automatic)",
    "Electronic check",
    "Mailed check",
]

TELCO_CHURN_SCHEMA = DataFrameSchema(
    columns={
        "customerID": Column(
            str,
            nullable=False,
            unique=True,
        ),
        "gender": Column(
            str,
            Check.isin(
                ["Female", "Male"]
            ),
            nullable=False,
        ),
        "SeniorCitizen": Column(
            int,
            Check.isin([0, 1]),
            nullable=False,
        ),
        "Partner": Column(
            str,
            Check.isin(YES_NO_VALUES),
            nullable=False,
        ),
        "Dependents": Column(
            str,
            Check.isin(YES_NO_VALUES),
            nullable=False,
        ),
        "tenure": Column(
            int,
            Check.in_range(
                min_value=0,
                max_value=72,
            ),
            nullable=False,
        ),
        "PhoneService": Column(
            str,
            Check.isin(YES_NO_VALUES),
            nullable=False,
        ),
        "MultipleLines": Column(
            str,
            Check.isin(
                MULTIPLE_LINES_VALUES
            ),
            nullable=False,
        ),
        "InternetService": Column(
            str,
            Check.isin(
                INTERNET_SERVICE_VALUES
            ),
            nullable=False,
        ),
        "OnlineSecurity": Column(
            str,
            Check.isin(
                INTERNET_OPTION_VALUES
            ),
            nullable=False,
        ),
        "OnlineBackup": Column(
            str,
            Check.isin(
                INTERNET_OPTION_VALUES
            ),
            nullable=False,
        ),
        "DeviceProtection": Column(
            str,
            Check.isin(
                INTERNET_OPTION_VALUES
            ),
            nullable=False,
        ),
        "TechSupport": Column(
            str,
            Check.isin(
                INTERNET_OPTION_VALUES
            ),
            nullable=False,
        ),
        "StreamingTV": Column(
            str,
            Check.isin(
                INTERNET_OPTION_VALUES
            ),
            nullable=False,
        ),
        "StreamingMovies": Column(
            str,
            Check.isin(
                INTERNET_OPTION_VALUES
            ),
            nullable=False,
        ),
        "Contract": Column(
            str,
            Check.isin(CONTRACT_VALUES),
            nullable=False,
        ),
        "PaperlessBilling": Column(
            str,
            Check.isin(YES_NO_VALUES),
            nullable=False,
        ),
        "PaymentMethod": Column(
            str,
            Check.isin(
                PAYMENT_METHOD_VALUES
            ),
            nullable=False,
        ),
        "MonthlyCharges": Column(
            float,
            Check.greater_than_or_equal_to(0),
            nullable=False,
        ),
        "TotalCharges": Column(
            object,
            nullable=True,
        ),
        "Churn": Column(
            str,
            Check.isin(YES_NO_VALUES),
            nullable=False,
        ),
    },
    strict=True,
    ordered=True,
    coerce=False,
    name="telco_customer_churn",
)


def validate_raw_dataset(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Valida o dataset bruto com Pandera."""
    return TELCO_CHURN_SCHEMA.validate(
        dataframe,
        lazy=True,
    )