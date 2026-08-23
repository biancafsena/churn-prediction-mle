"""API REST para previsão de churn."""

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request

from churn_prediction.api.schemas import (
    CustomerFeatures,
    HealthResponse,
    PredictionResponse,
)
from churn_prediction.config import (
    APP_NAME,
    APP_VERSION,
    MODEL_VERSION,
)
from churn_prediction.modeling.predict import (
    ChurnPredictor,
)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "churn_prediction_api"
)


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """Carrega os artefatos durante a inicialização."""
    logger.info(
        "Iniciando o carregamento dos artefatos."
    )

    try:
        application.state.predictor = (
            ChurnPredictor()
        )

    except Exception:
        logger.exception(
            "Falha ao carregar os artefatos."
        )
        raise

    logger.info(
        "Artefatos carregados com sucesso."
    )

    yield

    logger.info(
        "Encerrando a aplicação."
    )


app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "API REST para previsão da probabilidade "
        "de churn utilizando uma MLP em PyTorch."
    ),
    lifespan=lifespan,
)


@app.get(
    "/",
    tags=["Application"],
)
def root() -> dict[str, str]:
    """Apresenta informações básicas da API."""
    return {
        "application": APP_NAME,
        "version": APP_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "prediction": "/predict",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Monitoring"],
)
def health(
    request: Request,
) -> HealthResponse:
    """Verifica a disponibilidade da aplicação."""
    predictor = getattr(
        request.app.state,
        "predictor",
        None,
    )

    model_loaded = predictor is not None

    return HealthResponse(
        status=(
            "healthy"
            if model_loaded
            else "unhealthy"
        ),
        model_loaded=model_loaded,
        model_name=(
            predictor.metadata.get(
                "model_name",
                "unknown",
            )
            if model_loaded
            else "unavailable"
        ),
        api_version=APP_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
)
def predict(
    customer: CustomerFeatures,
    request: Request,
) -> PredictionResponse:
    """Retorna a probabilidade e a classe de churn."""
    predictor = getattr(
        request.app.state,
        "predictor",
        None,
    )

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="O modelo não está disponível.",
        )

    started_at = perf_counter()

    try:
        probabilities, predictions = (
            predictor.predict(
                customer.model_dump()
            )
        )

    except (TypeError, ValueError) as error:
        logger.warning(
            "Entrada rejeitada durante a inferência: %s",
            error,
        )

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Falha inesperada durante a inferência."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Não foi possível realizar "
                "a previsão."
            ),
        ) from error

    probability = float(
        probabilities[0]
    )

    prediction = int(
        predictions[0]
    )

    processing_time_ms = (
        perf_counter()
        - started_at
    ) * 1000

    log_data = {
        "event": "prediction_completed",
        "prediction": prediction,
        "probability": round(
            probability,
            6,
        ),
        "threshold": predictor.threshold,
        "processing_time_ms": round(
            processing_time_ms,
            3,
        ),
    }

    logger.info(
        json.dumps(
            log_data,
            ensure_ascii=False,
        )
    )

    return PredictionResponse(
        churn_probability=probability,
        churn_prediction=prediction,
        churn_label=(
            "Churn"
            if prediction == 1
            else "Não Churn"
        ),
        threshold=predictor.threshold,
        model_name=predictor.metadata.get(
            "model_name",
            "ChurnMLP",
        ),
        model_version=MODEL_VERSION,
        processing_time_ms=processing_time_ms,
    )