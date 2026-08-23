## Roteiro do Vídeo STAR — Customer Churn Prediction

### 1. Informações da Apresentação

- **Projeto:** Customer Churn Prediction
- **Tech Challenge:** Fase 1
- **Curso:** Pós-Tech em Machine Learning Engineering — FIAP
- **Autora:** Bianca Firmino Ferreira de Sena
- **Duração planejada:** aproximadamente 5 minutos
- **Método:** STAR — Situação, Tarefa, Ação e Resultado

---

### 2. Objetivo do Vídeo

Apresentar de forma objetiva:

- o problema de negócio;
- a preparação dos dados;
- os modelos avaliados;
- a construção da MLP em PyTorch;
- os resultados técnicos;
- o impacto dos erros;
- a seleção do modelo recomendado;
- o rastreamento com MLflow;
- a API de inferência;
- os testes e os mecanismos de observabilidade.

---

### 3. Estrutura de Tempo

| Tempo | Etapa | Conteúdo |
|---|---|---|
| 00:00–00:25 | Introdução | Apresentação pessoal e objetivo |
| 00:25–01:05 | Situação | Problema de churn e impacto dos erros |
| 01:05–01:35 | Tarefa | Objetivos técnicos do desafio |
| 01:35–03:50 | Ação | Dados, modelos, PyTorch, MLflow e API |
| 03:50–04:40 | Resultado | Métricas e seleção dos modelos |
| 04:40–05:00 | Encerramento | Conclusão e evolução futura |

---

## 4. Roteiro Completo

### 00:00–00:25 — Introdução

#### Tela sugerida

Mostrar:

- título do projeto no README;
- nome da autora;
- visão geral do repositório.

#### Fala

Olá, meu nome é Bianca Firmino Ferreira de Sena e este é o projeto Customer Churn Prediction, desenvolvido para o Tech Challenge da Fase 1 da Pós-Tech em Machine Learning Engineering da FIAP.

O objetivo foi construir uma solução de Machine Learning de ponta a ponta para identificar clientes com risco de cancelamento e apoiar a priorização de ações de retenção.

---

### 00:25–01:05 — Situação

#### Tela sugerida

Mostrar:

- seção “Problema de Negócio” do README;
- quantidade de clientes;
- taxa de churn;
- explicação sobre Falsos Positivos e Falsos Negativos.

#### Fala

O churn representa o cancelamento de um serviço por parte do cliente.

A base utilizada possui 7.043 clientes, 19 variáveis explicativas e taxa de churn de aproximadamente 26,54%, caracterizando uma distribuição desigual entre as classes.

Nesse cenário, os erros possuem impactos diferentes. Um Falso Positivo pode gerar uma campanha de retenção desnecessária. Já um Falso Negativo representa um cliente que realmente cancelaria, mas que não foi identificado a tempo.

Por isso, além das métricas tradicionais, simulei um cenário no qual um Falso Negativo custa cinco vezes mais que um Falso Positivo.

---

### 01:05–01:35 — Tarefa

#### Tela sugerida

Mostrar:

- seção de objetivos;
- lista dos notebooks;
- estrutura do projeto.

#### Fala

Minha tarefa foi desenvolver uma MLP com PyTorch, comparar seu desempenho com modelos de referência e transformar o experimento em uma solução reproduzível.

Para isso, também era necessário preparar os dados, definir métricas técnicas e de negócio, aplicar validação cruzada, analisar thresholds, rastrear os experimentos com MLflow, persistir os artefatos e disponibilizar a inferência por meio de uma API.

---

### 01:35–02:15 — Ação: dados e pré-processamento

#### Tela sugerida

Mostrar:

- notebook de análise exploratória;
- informações do dataset;
- pipeline de pré-processamento;
- schema Pandera.

#### Fala

Primeiro, realizei a análise exploratória e a validação da qualidade dos dados.

A coluna TotalCharges possuía 11 valores ausentes, tratados por imputação no pipeline.

Para as variáveis numéricas, apliquei imputação pela mediana e padronização com StandardScaler. Para as variáveis categóricas, utilizei imputação pelo valor mais frequente e One-Hot Encoding.

As 19 variáveis explicativas resultaram em 45 features processadas.

O pré-processamento foi ajustado exclusivamente nos dados de treino para evitar vazamento de dados. Também implementei validação estrutural com Pandera e validação das entradas da API com Pydantic.

---

### 02:15–02:55 — Ação: modelos e MLP PyTorch

#### Tela sugerida

Mostrar:

- comparação dos cinco modelos;
- arquitetura da MLP;
- gráfico de loss;
- Early Stopping.

#### Fala

Foram avaliados cinco modelos: DummyClassifier, Regressão Logística, Random Forest, MLPClassifier do Scikit-Learn e uma MLP implementada com PyTorch.

A rede neural possui 45 entradas, duas camadas ocultas com 64 e 32 neurônios, ativações ReLU, Dropout de 20% e um neurônio de saída.

O treinamento utilizou BCEWithLogitsLoss, otimizador Adam, regularização L2, batches de 32 registros e Early Stopping.

O treinamento foi encerrado após 29 épocas, e os melhores pesos, obtidos na época 9, foram restaurados.

---

### 02:55–03:25 — Ação: avaliação e thresholds

#### Tela sugerida

Mostrar:

- curva ROC;
- curva Precision-Recall;
- tabela de thresholds;
- gráfico de custo relativo.

#### Fala

Como a classe positiva é minoritária, defini a Average Precision como principal referência técnica, complementada por Accuracy, Precision, Recall, F1-Score, ROC-AUC e matriz de confusão.

Também avaliei diferentes thresholds.

Na MLP PyTorch, o threshold padrão de 0,50 produziu F1-Score de 0,5850. Entre os valores avaliados, o threshold de 0,30 aumentou o F1-Score para 0,6215 e reduziu 77 Falsos Negativos.

Na simulação de custo, o threshold de 0,20 apresentou o menor custo relativo, com redução de 36,11% em relação ao threshold padrão.

---

### 03:25–03:50 — Ação: engenharia e MLOps

#### Tela sugerida

Mostrar:

- interface do MLflow;
- estrutura de `src`;
- documentação Swagger;
- resultado dos testes.

#### Fala

Os cinco modelos foram registrados no MLflow com parâmetros, métricas, tags, tabelas e gráficos.

Também refatorei a solução em módulos reutilizáveis dentro de src, persisti os pesos, o pré-processador e os metadados, e desenvolvi uma API FastAPI com os endpoints de saúde e previsão.

A API possui validação de entrada, logging estruturado, Request ID, monitoramento de latência e tratamento de erros.

A qualidade foi validada com Ruff e 44 testes automatizados.

---

### 03:50–04:40 — Resultado

#### Tela sugerida

Mostrar:

- tabela comparativa dos modelos;
- gráfico de métricas;
- gráfico de impacto para o negócio;
- indicação do modelo recomendado.

#### Fala

No conjunto de teste, a Regressão Logística apresentou a maior Accuracy, com 0,8055, e a maior Precision, com 0,6572.

O Random Forest apresentou o maior Recall, de 0,7353, o maior F1-Score, de 0,6322, e a maior Average Precision, de 0,6492.

Além disso, apresentou apenas 99 Falsos Negativos e o menor custo relativo total, igual a 716.

Por isso, o Random Forest foi selecionado como modelo recomendado para o cenário de negócio.

A MLP PyTorch alcançou Accuracy de 0,7956, ROC-AUC de 0,8419 e Average Precision de 0,6349. Ela permaneceu como o modelo neural central do projeto e foi disponibilizada por meio da API.

Os resultados também permaneceram consistentes na validação cruzada estratificada.

---

### 04:40–05:00 — Encerramento

#### Tela sugerida

Mostrar:

- README;
- Model Card;
- arquitetura de inferência;
- plano de monitoramento;
- repositório no GitHub.

#### Fala

Como resultado, o projeto entrega não apenas um modelo, mas uma solução completa e reproduzível, com análise de negócio, rede neural em PyTorch, comparação de modelos, MLflow, persistência, API, testes e observabilidade.

Como próximos passos, a solução poderá evoluir com containerização, implantação em nuvem, monitoramento automático de drift e integração com dados reais de retenção.

Obrigada.

---

## 5. Telas Recomendadas

Preparar previamente as seguintes telas:

1. README na seção de visão geral;
2. informações do dataset;
3. pipeline de pré-processamento;
4. arquitetura da MLP PyTorch;
5. gráfico de evolução da loss;
6. curva ROC;
7. curva Precision-Recall;
8. tabela de thresholds;
9. comparação dos cinco modelos;
10. gráfico de impacto dos erros;
11. interface do MLflow;
12. Swagger da API;
13. resultado com `44 passed`;
14. Model Card;
15. arquitetura de inferência;
16. plano de monitoramento;
17. página do repositório no GitHub.

---

## 6. Demonstração da API

Caso exista tempo, mostrar rapidamente a documentação Swagger:

```text
http://127.0.0.1:8000/docs
```

Executar:

```text
GET /health
```

Destacar:

- status `healthy`;
- modelo carregado;
- versão da API.

Depois executar:

```text
POST /predict
```

Destacar:

- probabilidade de churn;
- classe prevista;
- threshold;
- nome e versão do modelo;
- tempo de processamento.

A demonstração deve ser rápida para não ultrapassar o limite do vídeo.

---

## 7. Comandos para Preparação

Antes da gravação, validar o projeto:

```bash
conda activate meu_env
make check
```

Iniciar a API:

```bash
make run-api
```

Em outro terminal, iniciar o MLflow:

```bash
conda activate meu_env
make run-mlflow
```

Endereços:

```text
API:     http://127.0.0.1:8000
Swagger: http://127.0.0.1:8000/docs
MLflow:  http://127.0.0.1:5000
```

---

## 8. Checklist Antes da Gravação

- [ ] fechar abas e aplicativos desnecessários;
- [ ] ocultar notificações;
- [ ] confirmar que não existem dados pessoais na tela;
- [ ] aumentar o zoom do navegador e do editor;
- [ ] validar a API;
- [ ] validar o MLflow;
- [ ] executar `make check`;
- [ ] deixar os gráficos previamente abertos;
- [ ] deixar o README aberto;
- [ ] testar o microfone;
- [ ] testar a captura de tela;
- [ ] ensaiar o roteiro;
- [ ] medir o tempo;
- [ ] manter a apresentação próxima de cinco minutos.

---

## 9. Recomendações de Apresentação

Durante a gravação:

- falar com ritmo constante;
- evitar ler códigos extensos;
- destacar decisões e resultados;
- explicar por que o Random Forest foi recomendado;
- explicar por que a MLP PyTorch permanece como modelo neural central;
- não permanecer muito tempo em uma única tela;
- utilizar o cursor para destacar os números;
- evitar excesso de detalhes técnicos;
- encerrar dentro do tempo disponível.

---

## 10. Resumo dos Números Principais

| Item | Resultado |
|---|---:|
| Clientes | 7.043 |
| Variáveis explicativas | 19 |
| Features processadas | 45 |
| Taxa de churn | 26,54% |
| Modelos comparados | 5 |
| Modelo recomendado | Random Forest |
| Modelo neural central | MLP PyTorch |
| Recall do Random Forest | 0,7353 |
| Average Precision do Random Forest | 0,6492 |
| Custo relativo do Random Forest | 716 |
| ROC-AUC da MLP PyTorch | 0,8419 |
| Average Precision da MLP PyTorch | 0,6349 |
| Melhor época da MLP | 9 |
| Testes automatizados | 44 |
| Resultado do Ruff | All checks passed |

---

## 11. Plano Alternativo para Redução do Tempo

Se o vídeo ultrapassar cinco minutos:

1. resumir a explicação do pré-processamento;
2. não detalhar todas as métricas;
3. apresentar apenas os resultados do Random Forest e da MLP PyTorch;
4. mostrar a interface do MLflow sem navegar por cada execução;
5. demonstrar apenas o endpoint `/predict`;
6. encurtar a apresentação das evoluções futuras.

Não remover:

- problema de negócio;
- MLP PyTorch;
- comparação dos modelos;
- modelo recomendado;
- impacto dos erros;
- MLflow;
- API;
- resultados principais.

---