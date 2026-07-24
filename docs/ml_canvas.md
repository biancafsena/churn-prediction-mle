# ML Canvas — Customer Churn Prediction

Este documento apresenta o planejamento do projeto de Machine Learning para previsão de cancelamento de clientes (churn), conectando o problema de negócio à estratégia de modelagem, aos critérios de avaliação, à utilização das predições e às práticas de reprodutibilidade.

---

## 1. Proposta de Valor

O projeto tem como objetivo identificar clientes com maior probabilidade de cancelamento antes que o churn ocorra, permitindo que a empresa direcione ações preventivas de retenção.

A solução busca apoiar decisões de negócio relacionadas à retenção de clientes, contribuindo potencialmente para:

- redução da perda de clientes;
- priorização de clientes com maior risco de churn;
- maior eficiência das campanhas de retenção;
- redução de intervenções desnecessárias;
- preservação potencial de receita.

O modelo não substitui a decisão de negócio. Sua função é fornecer uma estimativa de risco que possa apoiar a priorização das ações de retenção.

---

## 2. Problema de Machine Learning

O problema é formulado como uma tarefa de **classificação binária supervisionada**.

A variável alvo é `Churn`:

- `0` — cliente permaneceu;
- `1` — cliente cancelou.

A solução deverá produzir:

1. a probabilidade estimada de churn;
2. a classificação final do cliente, determinada a partir de um limiar de decisão (threshold).

A classe positiva do problema é:

**Churn = 1 — cliente que cancelou.**

---

## 3. Stakeholders

Os principais stakeholders potenciais da solução são:

- equipes de Customer Success;
- áreas de CRM e retenção;
- Marketing;
- áreas comerciais;
- gestores responsáveis por indicadores de churn;
- equipes de Dados e Machine Learning responsáveis pela manutenção da solução.

Essas áreas poderão utilizar as previsões para priorizar clientes e apoiar estratégias de retenção.

---

## 4. Fonte dos Dados

O projeto utiliza o dataset **Telco Customer Churn**, disponibilizado originalmente pela IBM para estudos relacionados à previsão de churn.

Características principais:

- granularidade: uma linha por cliente;
- problema: classificação binária;
- target: `Churn` (`Yes` / `No`);
- informações disponíveis:
  - características demográficas;
  - características contratuais;
  - serviços contratados;
  - tempo de relacionamento;
  - forma de pagamento;
  - cobranças mensais e totais.

O dataset bruto é armazenado localmente em:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Os dados brutos não são versionados no repositório.

---

## 5. Features

As variáveis explicativas disponíveis representam características dos clientes e de sua relação com os serviços contratados.

Entre os principais grupos de atributos estão:

### Perfil do cliente

- gênero;
- senioridade;
- existência de parceiro;
- dependentes.

### Relacionamento

- tempo de permanência (`tenure`);
- tipo de contrato.

### Serviços

- telefonia;
- internet;
- segurança online;
- backup;
- suporte técnico;
- streaming;
- proteção de dispositivos.

### Financeiro e cobrança

- cobrança mensal;
- cobrança total;
- método de pagamento;
- faturamento eletrônico.

O identificador `customerID` não deverá ser utilizado como variável preditora.

As transformações necessárias serão incorporadas ao pipeline de pré-processamento para garantir consistência entre treinamento e inferência.

---

## 6. Estratégia de Modelagem

A modelagem será realizada utilizando o ecossistema **Scikit-Learn**.

### Referência ingênua opcional

Um `DummyClassifier` poderá ser utilizado como referência mínima para verificar se os modelos treinados apresentam ganho real em relação a uma estratégia sem capacidade preditiva efetiva.

### Baseline oficial

**Regressão Logística**

A Regressão Logística será utilizada como baseline por oferecer:

- simplicidade;
- interpretabilidade;
- baixo custo computacional;
- referência consistente para comparação com modelos mais complexos.

### Modelos candidatos

Serão avaliados pelo menos:

1. **Random Forest / modelo ensemble baseado em árvores**;
2. **MLPClassifier**, representando uma rede neural simples implementada no Scikit-Learn.

Todos os modelos serão avaliados utilizando um protocolo consistente de comparação.

---

## 7. Estratégia de Avaliação

Os modelos serão avaliados utilizando o mesmo conjunto de teste e procedimentos consistentes de validação.

Quando aplicável, será utilizada divisão estratificada para preservar a proporção da variável alvo.

Também será utilizada validação cruzada para avaliar a estabilidade e capacidade de generalização dos modelos.

Será adotado:

`random_state = 42`

sempre que suportado pelos algoritmos utilizados.

---

## 8. Métricas Técnicas

A métrica técnica principal definida para comparação dos modelos é:

### PR-AUC / Average Precision

A PR-AUC será priorizada por avaliar a relação entre Precision e Recall sobre a classe positiva e por ser especialmente informativa em problemas com distribuição desigual entre as classes.

Durante a análise exploratória, a classe positiva de churn representou aproximadamente **26,5% dos registros**.

Também serão reportadas:

- Recall;
- F1-Score;
- ROC-AUC;
- Precision;
- Accuracy.

### Recall

Avalia a capacidade de identificar clientes que realmente apresentam churn.

Um Recall baixo implica maior quantidade de Falsos Negativos.

### Precision

Avalia, entre os clientes classificados como churn, quantos realmente pertencem à classe positiva.

Uma Precision baixa pode resultar em maior número de intervenções desnecessárias.

### F1-Score

Representa o equilíbrio entre Precision e Recall.

### ROC-AUC

Avalia a capacidade global do modelo de discriminar clientes com e sem churn em diferentes thresholds.

### Accuracy

Será reportada como métrica complementar, mas não será utilizada isoladamente para selecionar o modelo campeão.

---

## 9. Impacto dos Erros de Classificação

Os erros possuem consequências diferentes para o negócio.

### Falso Positivo — FP

O modelo classifica um cliente como risco de churn, mas ele permaneceria no serviço.

Possíveis consequências:

- campanha de retenção desnecessária;
- concessão desnecessária de desconto ou benefício;
- aumento do custo operacional.

### Falso Negativo — FN

O modelo classifica um cliente como sem risco, mas ele efetivamente cancela.

Possíveis consequências:

- oportunidade de retenção perdida;
- perda potencial de receita;
- perda do relacionamento com o cliente.

No contexto deste projeto, os Falsos Negativos possuem impacto relevante, pois representam clientes em risco que não foram identificados.

Entretanto, a seleção do modelo não será baseada apenas na redução de Falsos Negativos. Será considerado o equilíbrio entre Recall, Precision, PR-AUC e os custos potenciais das intervenções.

---

## 10. Métricas de Negócio

A avaliação de negócio poderá considerar os seguintes indicadores:

- quantidade de clientes abordados;
- churns potencialmente evitados;
- quantidade estimada de clientes retidos;
- custo total das intervenções;
- valor potencial do churn evitado;
- valor líquido estimado das ações;
- valor potencial perdido associado aos Falsos Negativos.

Como o dataset utilizado não fornece informações reais sobre receita por cliente, custo de campanhas ou taxa real de sucesso das ações de retenção, indicadores financeiros deverão ser tratados como **cenários ou estimativas**, com premissas explicitamente documentadas.

Em um ambiente real, métricas como receita preservada e ROI das campanhas deverão utilizar dados financeiros observados pela organização.

---

## 11. Critérios de Sucesso

### Critério técnico

O modelo candidato deverá apresentar desempenho superior ou competitivo em relação ao baseline de Regressão Logística, considerando principalmente:

- PR-AUC;
- Recall;
- F1-Score;
- ROC-AUC;
- estabilidade na validação cruzada.

### Critério de negócio

A solução deverá demonstrar capacidade de apoiar a identificação e priorização de clientes com maior risco de churn.

O sucesso de negócio deverá considerar o equilíbrio entre:

- clientes em risco corretamente identificados;
- churns potencialmente evitáveis;
- custo das intervenções;
- oportunidades perdidas por Falsos Negativos.

### Critério de engenharia

A solução deverá ser:

- reproduzível;
- modular;
- testável;
- documentada;
- utilizável por meio de uma API de inferência.

---

## 12. Seleção do Modelo Campeão

A escolha do modelo campeão não será baseada exclusivamente em uma única métrica.

A decisão considerará conjuntamente:

1. PR-AUC;
2. Recall;
3. F1-Score;
4. ROC-AUC;
5. Precision;
6. estabilidade na validação cruzada;
7. comportamento de Falsos Positivos e Falsos Negativos;
8. impacto potencial para o negócio;
9. capacidade de generalização.

Essa abordagem busca evitar a seleção de um modelo baseada apenas em um indicador isolado.

---

## 13. Como a Predição Será Utilizada

O modelo produzirá uma probabilidade estimada de churn para cada cliente.

Essa probabilidade poderá ser comparada a um threshold para classificar clientes em risco.

Fluxo conceitual:

`Dados do cliente → Pré-processamento → Modelo → Probabilidade de churn → Threshold → Decisão`

Clientes classificados como maior risco poderão ser priorizados para ações como:

- campanhas de retenção;
- contato preventivo;
- ofertas personalizadas;
- análise por equipes de Customer Success.

O threshold poderá ser ajustado conforme os objetivos e custos do negócio.

---

## 14. Inferência

Após a seleção do modelo campeão, o fluxo de pré-processamento e predição será disponibilizado por meio de uma API REST construída com **FastAPI**.

Endpoints previstos:

### `GET /health`

Permite verificar se a aplicação está disponível.

### `POST /predict`

Recebe os atributos necessários de um cliente e retorna a previsão produzida pelo modelo.

A validação dos dados de entrada será realizada utilizando modelos definidos com **Pydantic**.

---

## 15. Reprodutibilidade

O projeto será estruturado para garantir reprodutibilidade por meio de:

- gerenciamento de dependências com `pyproject.toml`;
- ambiente Python isolado;
- controle de aleatoriedade com `random_state = 42`;
- divisão estratificada dos dados quando aplicável;
- validação cruzada;
- pipeline consistente de pré-processamento;
- versionamento do código com Git;
- separação entre experimentação (`notebooks/`) e código produtivo (`src/`);
- registro estruturado dos resultados experimentais;
- testes automatizados com Pytest.

O MLflow poderá ser utilizado como ferramenta complementar para rastreamento dos experimentos, registrando parâmetros, métricas e informações relevantes dos modelos.

---

## 16. Riscos e Limitações

Algumas limitações devem ser consideradas:

- o dataset representa um cenário específico de telecomunicações;
- os resultados não devem ser generalizados automaticamente para outras empresas ou setores;
- o dataset não contém custos reais de retenção ou receita perdida por churn;
- relações observadas nos dados não implicam necessariamente causalidade;
- mudanças no comportamento dos clientes ao longo do tempo podem reduzir o desempenho do modelo;
- atributos demográficos exigem atenção quanto a possíveis vieses;
- o desempenho offline não garante o mesmo resultado em ambiente real.

Antes de uma utilização em produção, seria necessária validação com dados reais e atuais da organização.

---

## 17. Entregáveis

O projeto será composto por:

- análise exploratória dos dados;
- definição das métricas técnicas e de negócio;
- baseline com Regressão Logística;
- modelos candidatos Random Forest/ensemble e MLPClassifier;
- validação cruzada;
- comparação dos modelos;
- seleção e persistência do modelo campeão;
- código modular em `src/`;
- API REST com FastAPI;
- testes automatizados com Pytest;
- README com instruções de execução;
- Model Card;
- documentação complementar;
- apresentação final estruturada pelo método STAR.