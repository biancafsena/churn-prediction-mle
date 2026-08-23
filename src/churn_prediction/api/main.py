"""API REST para previsão de churn."""

import json
import logging
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
)
from contextlib import asynccontextmanager
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
)

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


def log_event(
    event: str,
    level: int = logging.INFO,
    **details: Any,
) -> None:
    """Registra um evento estruturado em JSON."""
    log_data = {
        "event": event,
        **details,
    }

    logger.log(
        level,
        json.dumps(
            log_data,
            ensure_ascii=False,
            default=str,
        ),
    )


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:
    """Carrega os artefatos durante a inicialização."""
    log_event(
        "application_starting"
    )

    try:
        application.state.predictor = (
            ChurnPredictor()
        )

    except Exception:
        logger.exception(
            json.dumps(
                {
                    "event": "artifact_loading_failed",
                }
            )
        )
        raise

    log_event(
        "artifacts_loaded",
        model_name=(
            application
            .state
            .predictor
            .metadata
            .get(
                "model_name",
                "ChurnMLP",
            )
        ),
    )

    yield

    log_event(
        "application_stopping"
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


@app.middleware("http")
async def request_observability_middleware(
    request: Request,
    call_next: Callable[
        [Request],
        Awaitable[Response],
    ],
) -> Response:
    """Registra latência e metadados das requisições."""
    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or uuid4().hex
    )

    request.state.request_id = request_id

    started_at = perf_counter()

    try:
        response = await call_next(
            request
        )

    except Exception:
        processing_time_ms = (
            perf_counter()
            - started_at
        ) * 1000

        logger.exception(
            json.dumps(
                {
                    "event": "request_failed",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "processing_time_ms": round(
                        processing_time_ms,
                        3,
                    ),
                },
                ensure_ascii=False,
            )
        )

        raise

    processing_time_ms = (
        perf_counter()
        - started_at
    ) * 1000

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Process-Time-Ms"
    ] = f"{processing_time_ms:.3f}"

    log_event(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        processing_time_ms=round(
            processing_time_ms,
            3,
        ),
    )

    return response


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
        log_event(
            "prediction_input_rejected",
            level=logging.WARNING,
            request_id=getattr(
                request.state,
                "request_id",
                "unavailable",
            ),
            error=str(error),
        )

        raise HTTPException(
            status_code=422,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            json.dumps(
                {
                    "event": "prediction_failed",
                    "request_id": getattr(
                        request.state,
                        "request_id",
                        "unavailable",
                    ),
                },
                ensure_ascii=False,
            )
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

    log_event(
        "prediction_completed",
        request_id=getattr(
            request.state,
            "request_id",
            "unavailable",
        ),
        prediction=prediction,
        probability=round(
            probability,
            6,
        ),
        threshold=predictor.threshold,
        processing_time_ms=round(
            processing_time_ms,
            3,
        ),
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