"""Configurações centrais do projeto de previsão de churn."""

from pathlib import Path

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODELS_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_STATE_PATH = (
    MODELS_DIR
    / "mlp_pytorch_state_dict.pt"
)

PREPROCESSOR_PATH = (
    MODELS_DIR
    / "mlp_pytorch_preprocessor.joblib"
)

MODEL_METADATA_PATH = (
    MODELS_DIR
    / "mlp_pytorch_metadata.json"
)

APP_NAME = "Customer Churn Prediction API"
APP_VERSION = "0.1.0"
MODEL_VERSION = "1.0.0"

DEFAULT_THRESHOLD = 0.50
BEST_F1_THRESHOLD = 0.30
MINIMUM_COST_THRESHOLD = 0.20

EXPECTED_RAW_FEATURES = 19
EXPECTED_PROCESSED_FEATURES = 45