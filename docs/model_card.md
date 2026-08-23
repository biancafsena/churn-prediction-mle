## Model Card — Customer Churn Prediction

### 1. Visão Geral

Este documento apresenta as características, resultados, limitações e recomendações de uso dos modelos desenvolvidos para previsão de churn de clientes de uma operadora de telecomunicações.

O projeto possui dois modelos com papéis complementares:

- **Random Forest:** modelo recomendado para o cenário de negócio;
- **MLP PyTorch:** modelo neural central desenvolvido para o Tech Challenge.

Essa separação permite selecionar o modelo mais adequado para as ações de retenção, mantendo a implementação e a avaliação da rede neural exigida pelo desafio.

---

### 2. Problema de Negócio

Churn representa o cancelamento ou a saída de um cliente de um serviço.

O objetivo da solução é identificar clientes com maior probabilidade de cancelamento, permitindo que equipes de CRM, Customer Success e retenção priorizem ações preventivas.

O problema foi formulado como uma classificação binária:

- `0` — Não Churn: cliente permaneceu;
- `1` — Churn: cliente cancelou.

Os erros possuem impactos distintos:

- **Falso Positivo:** cliente sem churn classificado como possível cancelamento, podendo receber uma ação de retenção desnecessária;
- **Falso Negativo:** cliente que realizou churn, mas não foi identificado pelo modelo, representando uma oportunidade de retenção perdida.

No cenário de custo utilizado no projeto, um Falso Negativo possui custo relativo cinco vezes maior que um Falso Positivo.

---

### 3. Dataset

Foi utilizado o dataset público **Telco Customer Churn**.

Principais características:

- 7.043 clientes;
- 21 colunas originais;
- 19 variáveis explicativas;
- 1.869 clientes com churn;
- 5.174 clientes sem churn;
- taxa de churn de 26,54%;
- 11 valores ausentes em `TotalCharges`.

As variáveis contêm informações:

- demográficas;
- contratuais;
- financeiras;
- relacionadas aos serviços utilizados;
- relacionadas ao método de pagamento.

A variável `customerID` foi utilizada apenas como identificador e removida da modelagem.

---

### 4. Pré-processamento

O pré-processamento foi implementado com Scikit-Learn e ajustado exclusivamente sobre os dados de treinamento.

#### 4.1 Variáveis numéricas

As variáveis numéricas utilizadas foram:

- `SeniorCitizen`;
- `tenure`;
- `MonthlyCharges`;
- `TotalCharges`.

Tratamentos aplicados:

- conversão para formato numérico;
- imputação de valores ausentes pela mediana;
- padronização com `StandardScaler`.

#### 4.2 Variáveis categóricas

Foram utilizadas 15 variáveis categóricas.

Tratamentos aplicados:

- imputação pelo valor mais frequente;
- One-Hot Encoding;
- tratamento de categorias desconhecidas durante a inferência.

Após o pré-processamento, as 19 variáveis explicativas originaram 45 features numéricas.

O pipeline treinado foi persistido em:

```text
models/mlp_pytorch_preprocessor.joblib
```

---

### 5. Modelos Avaliados

Os seguintes modelos foram comparados:

1. DummyClassifier;
2. Regressão Logística;
3. Random Forest;
4. MLPClassifier do Scikit-Learn;
5. MLP implementada com PyTorch.

Todos os modelos foram avaliados sobre o mesmo conjunto de teste, contendo 1.409 observações e preservando a proporção da classe positiva.

---

### 6. Métricas de Avaliação

A principal métrica técnica utilizada foi a **Average Precision**, pois o conjunto apresenta desbalanceamento entre as classes e o interesse principal está na identificação dos clientes com churn.

Também foram consideradas:

- Accuracy;
- Precision;
- Recall;
- F1-Score;
- ROC-AUC;
- matriz de confusão;
- validação cruzada estratificada;
- Falsos Positivos;
- Falsos Negativos;
- custo relativo dos erros;
- consistência entre teste e validação cruzada.

A seleção final não foi baseada exclusivamente em uma única métrica.

---

### 7. Comparação dos Modelos

Resultados obtidos no conjunto de teste com threshold igual a `0,50`:

| Modelo | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| DummyClassifier | 0,7346 | 0,0000 | 0,0000 | 0,0000 | 0,5000 | 0,2654 |
| Regressão Logística | **0,8055** | **0,6572** | 0,5588 | 0,6040 | **0,8419** | 0,6334 |
| Random Forest | 0,7729 | 0,5544 | **0,7353** | **0,6322** | 0,8401 | **0,6492** |
| MLPClassifier | 0,7850 | 0,6330 | 0,4519 | 0,5273 | 0,8341 | 0,6231 |
| MLP PyTorch | 0,7956 | 0,6344 | 0,5428 | 0,5850 | **0,8419** | 0,6349 |

A Regressão Logística apresentou a maior Accuracy e Precision.

O Random Forest apresentou o maior Recall, F1-Score e Average Precision.

A MLP PyTorch empatou com a Regressão Logística no maior ROC-AUC e permaneceu como o modelo neural central do projeto.

---

### 8. Modelo Recomendado para o Negócio

#### Random Forest

O Random Forest foi selecionado como modelo recomendado para o cenário de negócio.

Principais resultados no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,7729 |
| Precision | 0,5544 |
| Recall | 0,7353 |
| F1-Score | 0,6322 |
| ROC-AUC | 0,8401 |
| Average Precision | 0,6492 |
| Verdadeiros Negativos | 814 |
| Falsos Positivos | 221 |
| Falsos Negativos | 99 |
| Verdadeiros Positivos | 275 |
| Custo relativo total | 716 |

O modelo apresentou:

- maior Recall;
- maior F1-Score;
- maior Average Precision;
- menor quantidade de Falsos Negativos;
- menor custo relativo total;
- maior consistência entre teste e validação cruzada.

Esses resultados tornam o Random Forest mais adequado para o cenário em que deixar de identificar um cliente com risco de churn possui impacto superior ao custo de uma abordagem preventiva desnecessária.

---

### 9. Validação Cruzada do Random Forest

A validação cruzada estratificada foi executada com cinco folds.

| Métrica | Média | Desvio-padrão |
|---|---:|---:|
| Accuracy | 0,7748 | 0,0106 |
| Precision | 0,5584 | 0,0155 |
| Recall | 0,7224 | 0,0341 |
| F1-Score | 0,6297 | 0,0208 |
| ROC-AUC | 0,8449 | 0,0097 |
| Average Precision | 0,6550 | 0,0219 |

A diferença absoluta média entre os resultados do conjunto de teste e da validação cruzada foi de `0,0053`.

Essa proximidade fornece evidências de estabilidade e capacidade de generalização.

---

### 10. Modelo Neural Central

#### MLP PyTorch

A MLP PyTorch foi desenvolvida como modelo neural central do projeto.

Arquitetura:

```text
45 features
     ↓
Linear(45, 64) + ReLU + Dropout(20%)
     ↓
Linear(64, 32) + ReLU + Dropout(20%)
     ↓
Linear(32, 1)
     ↓
Logit de churn
```

Configuração do treinamento:

- 5.057 parâmetros treináveis;
- função de perda `BCEWithLogitsLoss`;
- otimizador Adam;
- learning rate de `0,001`;
- regularização L2 de `0,0001`;
- batch size de 32;
- máximo de 300 épocas;
- paciência de 20 épocas para Early Stopping;
- restauração dos melhores pesos;
- melhor época: 9;
- épocas executadas: 29;
- melhor loss de validação: 0,414244.

Resultados no conjunto de teste:

| Métrica | Resultado |
|---|---:|
| Accuracy | 0,7956 |
| Precision | 0,6344 |
| Recall | 0,5428 |
| F1-Score | 0,5850 |
| ROC-AUC | 0,8419 |
| Average Precision | 0,6349 |
| Test Loss | 0,4204 |

Matriz de confusão:

| Resultado | Quantidade |
|---|---:|
| Verdadeiros Negativos | 918 |
| Falsos Positivos | 117 |
| Falsos Negativos | 171 |
| Verdadeiros Positivos | 203 |

A MLP apresentou boa capacidade discriminativa, empatando com a Regressão Logística no maior ROC-AUC do conjunto de teste.

---

### 11. Validação Cruzada da MLP PyTorch

A validação cruzada estratificada foi executada com cinco folds.

| Métrica | Média | Desvio-padrão |
|---|---:|---:|
| Accuracy | 0,8012 | 0,0112 |
| Precision | 0,6537 | 0,0271 |
| Recall | 0,5340 | 0,0200 |
| F1-Score | 0,5878 | 0,0226 |
| ROC-AUC | 0,8419 | 0,0150 |
| Average Precision | 0,6488 | 0,0288 |

Os resultados do conjunto de teste permaneceram próximos das médias obtidas na validação cruzada.

A diferença absoluta média entre teste e validação cruzada foi de `0,0084`, indicando comportamento consistente nos diferentes subconjuntos avaliados.

---

### 12. Thresholds da MLP PyTorch

Foram analisados diferentes thresholds para compreender o equilíbrio entre Precision, Recall e custo dos erros.

| Critério | Threshold | Resultado principal |
|---|---:|---|
| Padrão | 0,50 | F1-Score de 0,5850 |
| Melhor F1 entre os avaliados | 0,30 | F1-Score de 0,6215 |
| Menor custo na simulação | 0,20 | Custo relativo de 621 |

No threshold `0,30`:

- Precision: 0,5313;
- Recall: 0,7487;
- F1-Score: 0,6215;
- Falsos Positivos: 247;
- Falsos Negativos: 94.

A alteração do threshold de `0,50` para `0,30` reduziu 77 Falsos Negativos, com aumento de 130 Falsos Positivos.

No threshold `0,20`, o custo relativo foi reduzido em 36,11% quando comparado ao threshold padrão.

Os custos utilizados são hipotéticos e devem ser substituídos por valores reais antes de uma aplicação em produção.

---

### 13. Persistência dos Artefatos

Os artefatos da MLP PyTorch estão armazenados em:

```text
models/
├── mlp_pytorch_state_dict.pt
├── mlp_pytorch_preprocessor.joblib
└── mlp_pytorch_metadata.json
```

Os arquivos armazenam:

- pesos treinados;
- arquitetura necessária para reconstrução da rede;
- pré-processador ajustado;
- nomes das features;
- hiperparâmetros;
- métricas;
- thresholds de referência;
- informações de treinamento.

O modelo reconstruído foi validado por testes automatizados de carregamento e inferência.

Durante as previsões, a rede permanece em modo de avaliação e utiliza `torch.inference_mode()`, evitando o cálculo desnecessário de gradientes.

---

### 14. Rastreamento com MLflow

Os cinco modelos foram registrados no experimento:

```text
churn-prediction-model-comparison
```

Para cada modelo foram armazenados:

- parâmetros;
- métricas do conjunto de teste;
- métricas de validação cruzada, quando disponíveis;
- componentes da matriz de confusão;
- custos relativos;
- tags;
- tabelas;
- gráficos;
- resumos em JSON.

Tags de seleção:

- Random Forest: `recommended_for_business=true`;
- MLP PyTorch: `central_neural_model=true`.

Os artefatos locais do MLflow, como `mlflow.db` e `mlartifacts/`, não são versionados no Git.

Os relatórios consolidados utilizados na documentação estão disponíveis em:

```text
reports/mlflow_experiments/
```

---

### 15. API de Inferência

A MLP PyTorch está disponível por meio de uma API FastAPI.

Endpoints:

- `GET /` — informações da aplicação;
- `GET /health` — disponibilidade da API e do modelo;
- `POST /predict` — previsão de churn;
- `GET /docs` — documentação Swagger;
- `GET /redoc` — documentação alternativa.

A API possui:

- validação das entradas com Pydantic;
- rejeição de campos desconhecidos;
- carregamento único dos artefatos durante a inicialização;
- execução do modelo em modo de avaliação;
- inferência sem cálculo de gradientes;
- logging estruturado;
- identificador único por requisição;
- medição de latência;
- tratamento de erros;
- headers de rastreamento.

Headers de observabilidade:

- `X-Request-ID`;
- `X-Process-Time-Ms`.

---

### 16. Uso Pretendido

A solução pode ser utilizada para:

- priorizar clientes com maior risco de churn;
- apoiar campanhas de retenção;
- segmentar ações preventivas;
- auxiliar equipes de CRM e Customer Success;
- comparar estratégias de threshold;
- simular diferentes custos operacionais;
- apoiar a tomada de decisão baseada em risco.

A previsão deve apoiar a tomada de decisão, e não substituir integralmente a análise humana ou as regras de negócio.

---

### 17. Usos Não Recomendados

O modelo não deve ser utilizado:

- como única justificativa para decisões que prejudiquem clientes;
- para negar atendimento, benefícios ou serviços;
- em populações muito diferentes da base de treinamento sem nova validação;
- sem acompanhamento de drift e degradação de desempenho;
- sem revisão dos custos reais de Falsos Positivos e Falsos Negativos;
- para inferir características sensíveis não relacionadas ao objetivo original;
- como sistema completamente autônomo de decisão;
- para finalidades diferentes da previsão de churn sem novo treinamento e validação.

---

### 18. Limitações

As principais limitações são:

- utilização de um dataset público e relativamente pequeno;
- ausência de informações comportamentais em tempo real;
- ausência de informações sobre campanhas de retenção anteriores;
- ausência do valor financeiro individual de cada cliente;
- custos dos erros definidos por simulação;
- possibilidade de mudanças no comportamento dos clientes;
- necessidade de monitorar drift;
- necessidade de reavaliar o threshold em produção;
- possibilidade de viés relacionado às variáveis demográficas e contratuais;
- ausência de garantia de desempenho em outra organização;
- necessidade de validação com dados reais antes da utilização em produção.

---

### 19. Riscos e Considerações Éticas

A utilização do modelo pode causar abordagens desnecessárias ou tratamento desigual de determinados grupos.

Recomendações:

- limitar o acesso aos dados;
- seguir os princípios da LGPD;
- evitar exposição de informações pessoais;
- registrar decisões e versões do modelo;
- revisar métricas por segmentos relevantes;
- manter supervisão humana;
- impedir o uso da previsão para práticas discriminatórias;
- avaliar periodicamente possíveis vieses;
- garantir transparência sobre a finalidade da previsão;
- coletar somente os dados necessários para a finalidade declarada.

A variável `gender` integra o dataset original e deve receber atenção especial nas análises de equidade antes de uma implantação real.

---

### 20. Monitoramento Recomendado

#### 20.1 Operação

Devem ser acompanhados:

- disponibilidade da API;
- volume de requisições;
- taxa de respostas por código HTTP;
- taxa de erros;
- latência média;
- percentis de latência;
- falhas no carregamento dos artefatos;
- utilização de recursos computacionais.

#### 20.2 Dados

Devem ser acompanhados:

- campos ausentes;
- categorias desconhecidas;
- valores fora dos limites esperados;
- alteração na distribuição das variáveis;
- alteração na proporção das categorias;
- alteração na taxa prevista de churn;
- falhas de validação do schema.

#### 20.3 Modelo

Quando os resultados reais estiverem disponíveis, devem ser acompanhados:

- Precision;
- Recall;
- F1-Score;
- ROC-AUC;
- Average Precision;
- Falsos Positivos;
- Falsos Negativos;
- custo relativo;
- calibração das probabilidades;
- drift de desempenho;
- desempenho por segmentos relevantes.

---

### 21. Testes e Qualidade

O projeto possui testes automatizados para:

- carregamento dos dados;
- validação do schema com Pandera;
- pré-processamento;
- arquitetura da MLP;
- treinamento;
- avaliação das métricas;
- avaliação de thresholds;
- cálculo do custo relativo;
- persistência e recarregamento;
- inferência;
- validação dos schemas Pydantic;
- endpoints da API;
- middleware de observabilidade;
- headers de rastreamento e latência.

Situação atual dos testes:

```text
44 passed
```

A validação estática com Ruff também foi executada:

```text
All checks passed!
```

O código produtivo não utiliza chamadas `print()`. As informações operacionais são registradas por logging estruturado.

---

### 22. Reprodutibilidade

O projeto utiliza:

- Python 3.11;
- dependências declaradas no `pyproject.toml`;
- random state igual a 42;
- divisão estratificada dos dados;
- validação cruzada estratificada;
- pré-processamento ajustado somente sobre o conjunto de treinamento;
- sementes aleatórias configuradas;
- algoritmos determinísticos no PyTorch;
- persistência do pré-processador, dos pesos e dos metadados;
- testes de recarregamento dos artefatos;
- rastreamento de experimentos com MLflow.

---

### 23. Responsabilidade e Manutenção

Responsável pelo projeto:

**Bianca Firmino Ferreira de Sena**

O modelo deverá ser revisado quando ocorrer:

- degradação relevante das métricas;
- mudança na distribuição dos dados;
- alteração dos produtos ou contratos;
- mudança na definição de churn;
- mudança no custo dos erros;
- inclusão de novas variáveis;
- atualização das regras de negócio;
- alteração relevante nas dependências;
- aumento da taxa de erros ou da latência da API.

---

### 24. Conclusão

O Random Forest apresentou o melhor resultado para o cenário de negócio, principalmente pela maior identificação de clientes com churn, menor quantidade de Falsos Negativos, maior Average Precision e menor custo relativo.

A MLP PyTorch apresentou boa capacidade discriminativa, persistência reproduzível e integração completa com a API de inferência, cumprindo o papel de modelo neural central do projeto.

A solução deve ser utilizada como apoio às estratégias de retenção, acompanhada por monitoramento contínuo, supervisão humana e revisão periódica dos dados, métricas, custos e thresholds.

---