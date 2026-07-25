## Customer Churn Prediction — Machine Learning Engineering

**Tech Challenge — Fase 1**

**Pós-Tech em Machine Learning Engineering — FIAP**

**Grupo:** 56

**Autora:** Bianca Firmino Ferreira de Sena

---

### Visão Geral

A retenção de clientes é um dos principais desafios enfrentados por empresas que operam com serviços recorrentes. Identificar antecipadamente clientes com maior probabilidade de cancelamento permite direcionar estratégias de retenção mais eficientes, reduzindo perdas financeiras e aumentando o valor gerado ao negócio.

Este projeto desenvolve uma solução completa de Machine Learning para previsão de churn de clientes, contemplando todas as etapas do ciclo de vida de um modelo, desde a análise exploratória dos dados até a disponibilização do modelo selecionado por meio de uma API REST.

O desenvolvimento segue uma separação clara entre:

- experimentação e análise em notebooks;
- código produtivo modularizado em `src/`;
- modelos e artefatos gerados;
- testes automatizados;
- documentação técnica.

---

### Sumário

- Visão Geral
- Objetivos
- Problema de Negócio
- Dataset
- Metodologia
- Estratégia de Avaliação
- Estrutura do Projeto
- Notebooks
- Resultados
- Tecnologias
- Reprodutibilidade
- API
- Testes
- Documentação
- Status do Projeto
- Autora

---

### Objetivos

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

---

### Problema de Negócio

O churn representa a saída ou o cancelamento de clientes de um serviço.

O objetivo preditivo deste projeto é identificar clientes com maior risco de churn, permitindo que ações de retenção possam ser direcionadas de maneira mais eficiente.

A avaliação considera também as consequências dos erros:

- **Falso Positivo (FP):** possível intervenção de retenção desnecessária.
- **Falso Negativo (FN):** cliente com churn não identificado, representando uma oportunidade de retenção perdida.

---

### Dataset

O projeto utiliza o dataset **Telco Customer Churn**, contendo informações demográficas, contratuais e relacionadas aos serviços utilizados pelos clientes.

A variável alvo é:

- `Churn = No` → cliente permaneceu;
- `Churn = Yes` → cliente cancelou.

O dataset bruto deve ser armazenado localmente em:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Os dados brutos não são versionados neste repositório.

---

### Metodologia

O desenvolvimento está organizado em quatro etapas principais.

#### Etapa 1 — Entendimento e Preparação

- definição e compreensão do problema de negócio;
- construção do ML Canvas;
- análise exploratória dos dados (EDA);
- avaliação da qualidade e prontidão dos dados;
- definição das métricas técnicas e de negócio;
- construção do baseline com Regressão Logística.

####  Etapa 2 — Modelagem e Avaliação

Serão avaliadas três famílias de modelos:

1. Regressão Logística (Baseline);
2. Random Forest;
3. MLPClassifier.

Todos os modelos serão avaliados utilizando exatamente o mesmo protocolo experimental, permitindo uma comparação consistente entre desempenho, robustez e capacidade de generalização.

####  Etapa 3 — Engenharia e API

Após a seleção do modelo campeão:

- refatoração do pipeline para código modular;
- implementação do fluxo produtivo em `src/`;
- persistência do modelo;
- desenvolvimento de API REST utilizando FastAPI;
- implementação dos endpoints `GET /health` e `POST /predict`;
- testes automatizados com Pytest.

####  Etapa 4 — Documentação e Apresentação

- documentação técnica;
- Model Card;
- consolidação dos resultados;
- limitações e vieses;
- preparação da apresentação final utilizando a metodologia STAR.

---

### Estratégia de Avaliação

A principal referência técnica definida para comparação dos modelos será a **PR-AUC (Average Precision)**, considerando o desbalanceamento existente entre as classes.

Também serão avaliadas:

- Recall;
- F1-Score;
- ROC-AUC;
- Precision;
- Accuracy.

A seleção do modelo campeão não será baseada exclusivamente em uma única métrica.

Também serão considerados:

- estabilidade na validação cruzada;
- capacidade de generalização;
- comportamento dos Falsos Positivos;
- comportamento dos Falsos Negativos;
- impacto dos erros para o negócio;
- equilíbrio entre desempenho preditivo e interpretabilidade.

---

### Estrutura do Projeto

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

- `data/raw/` — dados originais.
- `data/processed/` — dados processados.
- `notebooks/` — análises e experimentação.
- `src/churn_prediction/data/` — carregamento dos dados.
- `src/churn_prediction/features/` — engenharia de atributos.
- `src/churn_prediction/modeling/` — treinamento e inferência.
- `src/churn_prediction/api/` — FastAPI.
- `models/` — modelos persistidos.
- `artifacts/metrics/` — métricas experimentais.
- `artifacts/figures/` — gráficos.
- `tests/` — testes automatizados.
- `docs/` — documentação complementar.

---

### Notebooks

O fluxo experimental está organizado sequencialmente:

1. `01_eda.ipynb`
2. `02_metricas_negocio_tecnicas.ipynb`
3. `03_baseline_logistic_regression.ipynb`
4. `04_random_forest.ipynb`
5. `05_mlp_classifier.ipynb`
6. `06_comparacao_modelos.ipynb`

Fluxo do projeto:

```text
EDA
   ↓
Métricas
   ↓
Baseline
   ↓
Random Forest
   ↓
MLPClassifier
   ↓
Comparação
   ↓
Modelo Campeão
   ↓
API REST
```

---

### Resultados

Os resultados serão atualizados conforme cada experimento for concluído.

| Modelo | PR-AUC | ROC-AUC | F1-Score | Recall | Precision |
|---------|:------:|:-------:|:--------:|:------:|:---------:|
| Regressão Logística | *(Atualizar)* | *(Atualizar)* | *(Atualizar)* | *(Atualizar)* | *(Atualizar)* |
| Random Forest | A definir | A definir | A definir | A definir | A definir |
| MLPClassifier | A definir | A definir | A definir | A definir | A definir |

### Modelo Campeão

Será definido após a comparação experimental entre todos os modelos.

---

### Tecnologias

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Joblib
- FastAPI
- Pydantic
- Uvicorn
- Pytest
- MLflow
- Ruff
- Jupyter Notebook

---

### Reprodutibilidade

O projeto adota práticas para garantir experimentos reproduzíveis:

- ambiente Python isolado;
- dependências declaradas;
- controle de sementes aleatórias;
- divisão estratificada dos dados;
- validação cruzada;
- pipeline reproduzível;
- persistência dos modelos;
- testes automatizados.

---

### API

Será disponibilizada uma API REST utilizando FastAPI.

Endpoints previstos:

#### `GET /health`

Verifica o estado da aplicação.

#### `POST /predict`

Recebe os dados de um cliente e retorna a previsão de churn.

---

### Testes

Serão implementados testes automatizados utilizando Pytest para validar:

- pré-processamento;
- treinamento;
- inferência;
- endpoints da API.

---

### Documentação

A documentação complementar estará organizada em `docs/`.

- `ml_canvas.md`
- `model_card.md`
- `star_video_script.md`

---

### Status do Projeto

🚧 **Em desenvolvimento**

### Etapas concluídas

- ✅ Análise Exploratória dos Dados (EDA)
- ✅ Definição das Métricas Técnicas e de Negócio
- ✅ Modelo Baseline — Regressão Logística

### Próximas etapas

- 🔄 Random Forest
- ⏳ MLPClassifier
- ⏳ Comparação dos Modelos
- ⏳ API REST
- ⏳ Testes Automatizados
- ⏳ Documentação Final

---

## Autora

**Bianca Firmino Ferreira de Sena**

Projeto desenvolvido como parte da **Pós-Tech em Machine Learning Engineering — FIAP**.