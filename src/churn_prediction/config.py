"""Configurações centrais do projeto de previsão de churn."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

RAW_DATA_PATH = (
    RAW_DATA_DIR
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

MODELS_DIR = PROJECT_ROOT / "models"

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

IDENTIFIER_COLUMN = "customerID"
TARGET_COLUMN = "Churn"

APP_NAME = "Customer Churn Prediction API"
APP_VERSION = "0.1.0"
MODEL_VERSION = "1.0.0"

DEFAULT_THRESHOLD = 0.50
BEST_F1_THRESHOLD = 0.30
MINIMUM_COST_THRESHOLD = 0.20

EXPECTED_RAW_FEATURES = 19
EXPECTED_PROCESSED_FEATURES = 45