# Customer Churn Prediction — Machine Learning Engineering

Pipeline de Machine Learning de ponta a ponta para previsão de churn de clientes, desenvolvido com foco em experimentação reprodutível, comparação de modelos, engenharia de software, testes automatizados e disponibilização de inferência por API REST.

## Visão Geral

A retenção de clientes é um desafio relevante para empresas que operam com serviços recorrentes. Identificar antecipadamente clientes com maior probabilidade de cancelamento permite direcionar estratégias de retenção de forma mais eficiente.

Este projeto desenvolve uma solução completa de Machine Learning para previsão de churn, contemplando desde a análise e preparação dos dados até a disponibilização do modelo selecionado por meio de uma API REST.

O desenvolvimento segue uma separação clara entre:

- experimentação e análise em notebooks;
- código produtivo modularizado em `src/`;
- modelos e artefatos gerados;
- testes automatizados;
- documentação técnica.

## Objetivos

O projeto tem como principais objetivos:

- compreender os fatores associados ao churn de clientes;
- avaliar a qualidade e a prontidão dos dados para modelagem;
- definir métricas técnicas e de negócio adequadas ao problema;
- estabelecer uma Regressão Logística como modelo baseline;
- treinar e avaliar modelos baseados em árvores/ensembles;
- treinar uma rede neural simples utilizando `MLPClassifier`;
- aplicar validação cruzada para avaliar a robustez dos modelos;
- comparar os modelos utilizando um protocolo consistente de avaliação;
- selecionar e persistir o modelo campeão;
- disponibilizar previsões por meio de uma API REST com FastAPI;
- validar componentes críticos utilizando testes automatizados com Pytest.

## Problema de Negócio

O churn representa a saída ou o cancelamento de clientes de um serviço.

O objetivo preditivo deste projeto é identificar clientes com maior risco de churn, permitindo que ações de retenção possam ser direcionadas de maneira mais eficiente.

A avaliação considera também as consequências dos erros:

- **Falso Positivo (FP):** possível intervenção de retenção desnecessária.
- **Falso Negativo (FN):** cliente com churn não identificado, representando uma oportunidade de retenção perdida.

## Dataset

O projeto utiliza o dataset **Telco Customer Churn**, contendo informações demográficas, contratuais e relacionadas aos serviços utilizados pelos clientes.

A variável alvo é:

- `Churn = No` → cliente permaneceu;
- `Churn = Yes` → cliente cancelou.

O dataset bruto deve ser armazenado localmente em:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Os dados brutos não são versionados neste repositório.

---

## Metodologia

O desenvolvimento está organizado em quatro etapas principais.

### Etapa 1 — Entendimento e Preparação

- definição e compreensão do problema de negócio;
- construção do ML Canvas;
- análise exploratória dos dados (EDA);
- avaliação da qualidade e prontidão dos dados;
- definição das métricas técnicas e de negócio;
- construção do baseline com Regressão Logística.

### Etapa 2 — Modelagem e Avaliação

Serão avaliadas três famílias de modelos:

1. **Regressão Logística** — modelo baseline;
2. **Random Forest / Ensemble** — modelo baseado em árvores;
3. **MLPClassifier** — rede neural simples utilizando Scikit-Learn.

Os modelos serão avaliados utilizando o mesmo protocolo experimental e validação cruzada para permitir uma comparação consistente.

### Etapa 3 — Engenharia e API

Após a seleção do modelo campeão:

- o pré-processamento será refatorado para código modular;
- o fluxo de inferência será implementado em `src/`;
- o modelo será persistido para utilização em produção;
- será desenvolvida uma API REST com FastAPI;
- serão disponibilizados os endpoints `GET /health` e `POST /predict`;
- serão implementados testes automatizados com Pytest.

### Etapa 4 — Documentação e Apresentação

A etapa final contempla:

- documentação do projeto;
- Model Card do modelo selecionado;
- consolidação dos resultados;
- limitações e possíveis vieses;
- preparação da apresentação final utilizando a estrutura STAR.

---

## Estratégia de Avaliação

A principal referência técnica definida para comparação dos modelos é a **PR-AUC / Average Precision**, considerando a relevância da classe positiva e o desbalanceamento observado no problema.

Também serão reportadas as seguintes métricas:

- Recall;
- F1-Score;
- ROC-AUC;
- Precision;
- Accuracy.

A seleção do modelo campeão não será baseada exclusivamente em uma única métrica.

Também serão considerados:

- estabilidade durante a validação cruzada;
- capacidade de generalização;
- comportamento de Falsos Positivos e Falsos Negativos;
- impacto potencial dos erros para o negócio;
- equilíbrio entre desempenho preditivo e aplicabilidade da solução.


---

## Estrutura do Projeto

```text
churn-prediction-mle/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   └── churn_prediction/
│       ├── data/
│       ├── features/
│       ├── modeling/
│       └── api/
├── models/
├── artifacts/
│   ├── metrics/
│   └── figures/
├── tests/
├── docs/
├── pyproject.toml
├── .gitignore
└── README.md
```

### Responsabilidades dos Diretórios

- `data/raw/` — dados originais utilizados no projeto;
- `data/processed/` — dados processados gerados durante o pipeline;
- `notebooks/` — análise exploratória, experimentação e comparação dos modelos;
- `src/churn_prediction/data/` — carregamento e manipulação dos dados;
- `src/churn_prediction/features/` — pré-processamento e engenharia de atributos;
- `src/churn_prediction/modeling/` — treinamento, avaliação e inferência;
- `src/churn_prediction/api/` — aplicação FastAPI e contratos de entrada e saída;
- `models/` — modelos treinados e persistidos;
- `artifacts/metrics/` — resultados e métricas dos experimentos;
- `artifacts/figures/` — visualizações e gráficos gerados;
- `tests/` — testes automatizados;
- `docs/` — documentação complementar do projeto.


---

## Notebooks

O fluxo experimental está organizado sequencialmente:

1. `01_eda.ipynb` — análise exploratória e avaliação da qualidade dos dados;
2. `02_metricas_negocio_tecnicas.ipynb` — definição das métricas técnicas e de negócio;
3. `03_baseline_logistic_regression.ipynb` — construção e avaliação do modelo baseline;
4. `04_random_forest.ipynb` — treinamento e avaliação do modelo baseado em árvores;
5. `05_mlp_classifier.ipynb` — treinamento e avaliação da rede neural simples;
6. `06_comparacao_modelos.ipynb` — comparação final e seleção do modelo campeão.

O fluxo segue a sequência:

EDA → Métricas → Baseline → Random Forest → MLPClassifier → Comparação → Modelo Campeão

---

## Resultados

Os resultados serão atualizados progressivamente conforme os experimentos forem concluídos.

| Modelo | PR-AUC | ROC-AUC | F1-Score | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| Regressão Logística | A definir | A definir | A definir | A definir | A definir |
| Random Forest / Ensemble | A definir | A definir | A definir | A definir | A definir |
| MLPClassifier | A definir | A definir | A definir | A definir | A definir |

### Modelo Campeão

A definir após a conclusão da comparação experimental.

A escolha será fundamentada nos resultados obtidos no conjunto de avaliação, na validação cruzada e na análise dos impactos dos erros de classificação.


---

## Tecnologias

Principais tecnologias utilizadas no projeto:

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- MLflow
- Ruff
- Jupyter

---

## Reprodutibilidade

O projeto adota práticas para garantir experimentos consistentes e reproduzíveis:

- ambiente Python isolado;
- dependências declaradas em `pyproject.toml`;
- controle de sementes aleatórias;
- divisão estratificada dos dados quando aplicável;
- validação cruzada;
- protocolo consistente de comparação entre modelos;
- registro dos resultados experimentais;
- persistência do pipeline e do modelo selecionado;
- testes automatizados.

As instruções completas de instalação e execução serão adicionadas após a implementação do pipeline produtivo e da API.

---

## API

A solução disponibilizará uma API REST construída com FastAPI.

Endpoints previstos:

### `GET /health`

Verifica a disponibilidade e o estado da aplicação.

### `POST /predict`

Recebe os atributos de um cliente e retorna a previsão de churn gerada pelo modelo selecionado.

A documentação interativa será disponibilizada automaticamente pelo FastAPI após a execução da aplicação.

---

## Testes

Os testes automatizados serão implementados com Pytest e contemplarão componentes críticos da solução, incluindo:

- pré-processamento dos dados;
- fluxo de predição;
- funcionamento dos endpoints da API.

---

## Documentação

A documentação complementar está organizada em `docs/`:

- `ml_canvas.md` — estruturação do problema de Machine Learning;
- `model_card.md` — documentação do modelo selecionado, incluindo performance, limitações e possíveis vieses;
- `star_video_script.md` — roteiro da apresentação final utilizando o método STAR.

---

## Status do Projeto

🚧 **Em desenvolvimento**

Etapa atual: entendimento, preparação dos dados e construção do baseline.

---

## Autora

**Bianca Firmino Ferreira de Sena**

Projeto desenvolvido como parte da Pós-Tech em Machine Learning Engineering — FIAP.
