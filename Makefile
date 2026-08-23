PYTHON ?= python
HOST ?= 127.0.0.1
API_PORT ?= 8000
MLFLOW_PORT ?= 5000

.PHONY: help install lint format test coverage check run-api run-mlflow

help:
	@echo "Comandos disponíveis:"
	@echo "  make install     Instala o projeto e as dependências de desenvolvimento"
	@echo "  make lint        Executa a validação estática com Ruff"
	@echo "  make format      Formata o código com Ruff"
	@echo "  make test        Executa os testes automatizados"
	@echo "  make coverage    Executa os testes com relatório de cobertura"
	@echo "  make check       Executa lint e testes"
	@echo "  make run-api     Inicia a API FastAPI"
	@echo "  make run-mlflow  Inicia a interface do MLflow"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check src tests

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check src tests --fix

test:
	$(PYTHON) -m pytest -q

coverage:
	$(PYTHON) -m pytest \
		--cov=src/churn_prediction \
		--cov-report=term-missing \
		--cov-report=html

check: lint test

run-api:
	$(PYTHON) -m uvicorn \
		churn_prediction.api.main:app \
		--host $(HOST) \
		--port $(API_PORT) \
		--reload

run-mlflow:
	$(PYTHON) -m mlflow ui \
		--backend-store-uri sqlite:///mlflow.db \
		--host $(HOST) \
		--port $(MLFLOW_PORT)