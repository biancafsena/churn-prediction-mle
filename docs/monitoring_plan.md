## Plano de Monitoramento — Customer Churn Prediction

### 1. Visão Geral

Este documento define o plano de monitoramento da solução de previsão de churn.

O objetivo é acompanhar continuamente:

- disponibilidade da API;
- latência das requisições;
- taxa de erros;
- qualidade dos dados recebidos;
- distribuição das previsões;
- drift dos dados;
- degradação do desempenho do modelo;
- impacto dos Falsos Positivos e Falsos Negativos;
- funcionamento e integridade dos artefatos.

O monitoramento permite identificar problemas antes que eles provoquem impactos relevantes nas ações de retenção.

---

### 2. Escopo

O plano contempla os seguintes componentes:

| Camada | Elementos monitorados |
|---|---|
| Aplicação | API FastAPI, endpoints e middleware |
| Infraestrutura | CPU, memória, disponibilidade e volume de requisições |
| Dados | Schema, valores ausentes, categorias e distribuições |
| Modelo | Probabilidades, classes, métricas e drift |
| Negócio | Falsos Negativos, Falsos Positivos e custo relativo |
| Artefatos | Pesos, pré-processador e metadados |
| Experimentos | Execuções, métricas, parâmetros e tags no MLflow |

---

### 3. Objetivos do Monitoramento

Os objetivos são:

1. garantir que a API esteja disponível;
2. identificar aumento de latência;
3. detectar falhas nas requisições;
4. identificar mudanças na distribuição dos dados;
5. detectar aumento de campos inválidos ou ausentes;
6. acompanhar a distribuição das probabilidades;
7. identificar degradação das métricas;
8. monitorar o impacto dos erros para o negócio;
9. rastrear a versão responsável por cada previsão;
10. orientar decisões de investigação, rollback e retreinamento.

---

### 4. Papéis dos Modelos

O projeto possui dois modelos principais:

- **Random Forest:** modelo recomendado para o cenário de negócio;
- **MLP PyTorch:** modelo neural central utilizado pela API.

O monitoramento deve considerar separadamente:

- desempenho técnico do modelo disponibilizado;
- desempenho do modelo recomendado para o negócio;
- diferenças entre os modelos;
- versão efetivamente utilizada em cada previsão.

Na arquitetura atual, a API utiliza a MLP PyTorch.

---

### 5. Monitoramento Operacional

As métricas operacionais devem ser coletadas continuamente.

| Métrica | Descrição | Frequência |
|---|---|---|
| Disponibilidade | Percentual de tempo em que a API responde | Contínua |
| Volume de requisições | Quantidade de requisições recebidas | Por minuto |
| Taxa de sucesso | Percentual de respostas HTTP 2xx | Por minuto |
| Taxa de erro do cliente | Percentual de respostas HTTP 4xx | Por minuto |
| Taxa de erro interno | Percentual de respostas HTTP 5xx | Por minuto |
| Latência média | Tempo médio das requisições | Por minuto |
| Latência P95 | Tempo abaixo do qual estão 95% das requisições | A cada 5 minutos |
| Latência P99 | Tempo abaixo do qual estão 99% das requisições | A cada 5 minutos |
| Requisições por segundo | Carga recebida pela API | Contínua |
| Reinicializações | Quantidade de reinicializações da aplicação | Contínua |

---

### 6. Indicadores Operacionais Iniciais

Os limites abaixo são referências iniciais e devem ser recalibrados após a coleta de dados reais.

| Indicador | Saudável | Atenção | Crítico |
|---|---:|---:|---:|
| Disponibilidade | ≥ 99,5% | 99,0% a 99,5% | < 99,0% |
| Respostas HTTP 5xx | < 1% | 1% a 3% | > 3% |
| Respostas HTTP 422 | < 5% | 5% a 10% | > 10% |
| Latência média | < 200 ms | 200 a 500 ms | > 500 ms |
| Latência P95 | < 500 ms | 500 ms a 1 s | > 1 s |
| Falha no `/health` | 0 | Uma ocorrência isolada | Duas ou mais consecutivas |
| Modelo indisponível | 0 | Não aplicável | Qualquer ocorrência |

Esses limites não representam um SLA contratual. São valores de referência para o projeto.

---

### 7. Endpoint de Saúde

O endpoint monitorado é:

```text
GET /health
```

Resposta saudável esperada:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "ChurnMLP",
  "api_version": "0.1.0"
}
```

Devem ser validados:

- código HTTP igual a 200;
- campo `status` igual a `healthy`;
- campo `model_loaded` igual a `true`;
- nome esperado do modelo;
- versão esperada da API;
- presença do header `X-Request-ID`;
- presença do header `X-Process-Time-Ms`.

---

### 8. Monitoramento de Latência

A solução mede dois tipos de latência.

#### Latência total da requisição

Disponível no header:

```text
X-Process-Time-Ms
```

Representa o tempo total medido pelo middleware.

#### Latência interna da previsão

Disponível no corpo da resposta:

```text
processing_time_ms
```

Representa o tempo utilizado dentro do endpoint para executar o fluxo de inferência.

A comparação entre os dois valores ajuda a identificar se o atraso está:

- no pré-processamento;
- no modelo;
- no middleware;
- no servidor;
- na serialização da resposta;
- em componentes externos.

---

### 9. Logging Estruturado

Os logs são emitidos em formato JSON.

Eventos recomendados:

| Evento | Finalidade |
|---|---|
| `application_starting` | Início da aplicação |
| `artifacts_loaded` | Artefatos carregados |
| `artifact_loading_failed` | Falha no carregamento |
| `request_completed` | Requisição concluída |
| `request_failed` | Falha durante a requisição |
| `prediction_completed` | Previsão concluída |
| `prediction_rejected` | Entrada rejeitada |
| `prediction_failed` | Erro inesperado na inferência |
| `application_stopping` | Encerramento da aplicação |

Campos recomendados:

- timestamp;
- nível do log;
- evento;
- Request ID;
- método HTTP;
- rota;
- código HTTP;
- latência;
- nome do modelo;
- versão do modelo;
- probabilidade;
- classe prevista;
- threshold;
- mensagem do erro.

Dados pessoais completos não devem ser incluídos nos logs.

---

### 10. Rastreamento das Requisições

Cada requisição deve possuir um identificador único.

Header:

```text
X-Request-ID
```

Esse identificador deve estar presente:

- no header da resposta;
- no log da requisição;
- no log da previsão;
- nos registros de erro;
- em eventuais traces distribuídos.

O Request ID possibilita reconstruir o caminho de uma requisição sem registrar os dados pessoais do cliente.

---

### 11. Monitoramento da Qualidade dos Dados

As entradas devem ser acompanhadas para identificar alterações no comportamento da fonte de dados.

Indicadores:

- quantidade de registros;
- campos ausentes;
- campos adicionais;
- valores inválidos;
- valores fora dos limites;
- categorias desconhecidas;
- falhas de conversão numérica;
- registros duplicados em processamento batch;
- alteração dos tipos;
- alteração da proporção das categorias.

A validação ocorre em duas camadas:

- Pydantic para as entradas da API;
- Pandera para o dataset utilizado nos fluxos de dados e treinamento.

---

### 12. Variáveis Numéricas Monitoradas

As principais variáveis numéricas são:

- `SeniorCitizen`;
- `tenure`;
- `MonthlyCharges`;
- `TotalCharges`.

Para cada variável devem ser acompanhados:

- média;
- mediana;
- desvio-padrão;
- mínimo;
- máximo;
- percentis;
- proporção de valores ausentes;
- proporção de valores fora do domínio;
- alteração da distribuição.

Uma mudança relevante pode indicar:

- alteração no comportamento dos clientes;
- mudança no sistema de origem;
- falha na coleta;
- erro de unidade;
- mudança no portfólio de produtos.

---

### 13. Variáveis Categóricas Monitoradas

Devem ser monitoradas as 15 variáveis categóricas utilizadas pelo modelo.

Para cada variável devem ser acompanhados:

- frequência absoluta;
- frequência relativa;
- categorias novas;
- categorias ausentes;
- alteração na categoria mais frequente;
- aumento de categorias desconhecidas;
- divergência da distribuição de referência.

O pré-processador utiliza `handle_unknown="ignore"`, evitando interrupção imediata da inferência. Entretanto, o aumento de categorias desconhecidas deve gerar investigação, pois pode reduzir a qualidade das previsões.

---

### 14. Monitoramento de Data Drift

Data drift representa uma alteração na distribuição dos dados de entrada em relação à referência de treinamento.

Métodos sugeridos:

- Population Stability Index para variáveis numéricas e categóricas;
- teste de Kolmogorov-Smirnov para variáveis numéricas;
- teste qui-quadrado para variáveis categóricas;
- comparação de médias e percentis;
- Jensen-Shannon Divergence;
- análise visual das distribuições.

Referência inicial para PSI:

| PSI | Interpretação |
|---:|---|
| < 0,10 | Sem mudança relevante |
| 0,10 a 0,25 | Mudança moderada |
| > 0,25 | Mudança relevante |

Os limites deverão ser ajustados após observação do comportamento real.

---

### 15. Monitoramento das Previsões

Mesmo antes da obtenção dos rótulos reais, é possível acompanhar:

- quantidade de previsões;
- probabilidade média de churn;
- mediana das probabilidades;
- distribuição das probabilidades;
- percentual de previsões positivas;
- percentual de previsões negativas;
- quantidade de previsões próximas ao threshold;
- alteração da taxa prevista de churn;
- previsões por período e segmento.

A taxa histórica de churn do dataset é de 26,54%.

Uma diferença persistente e relevante entre a taxa prevista e a referência deve ser investigada, sem assumir automaticamente que representa erro do modelo.

---

### 16. Monitoramento de Concept Drift

Concept drift ocorre quando a relação entre as características dos clientes e o churn muda.

A detecção depende da obtenção posterior dos resultados reais.

Sinais possíveis:

- queda de Recall;
- queda de Average Precision;
- aumento de Falsos Negativos;
- aumento do custo relativo;
- mudança na calibração das probabilidades;
- redução do desempenho em segmentos específicos;
- divergência persistente entre previsão e resultado real.

Possíveis causas:

- mudanças comerciais;
- alteração nos contratos;
- novos produtos;
- mudança de preços;
- ações de concorrentes;
- campanhas de retenção;
- mudança no comportamento dos consumidores.

---

### 17. Monitoramento do Desempenho

Quando os rótulos reais estiverem disponíveis, deverão ser calculadas:

- Accuracy;
- Precision;
- Recall;
- F1-Score;
- ROC-AUC;
- Average Precision;
- matriz de confusão;
- Falsos Positivos;
- Falsos Negativos;
- Verdadeiros Positivos;
- Verdadeiros Negativos;
- custo relativo dos erros;
- calibração das probabilidades.

A **Average Precision** permanece como principal métrica técnica.

O Recall e a quantidade de Falsos Negativos permanecem como indicadores relevantes para o negócio.

---

### 18. Valores de Referência da MLP PyTorch

Resultados de referência do conjunto de teste com threshold `0,50`:

| Métrica | Referência |
|---|---:|
| Accuracy | 0,7956 |
| Precision | 0,6344 |
| Recall | 0,5428 |
| F1-Score | 0,5850 |
| ROC-AUC | 0,8419 |
| Average Precision | 0,6349 |
| Falsos Positivos | 117 |
| Falsos Negativos | 171 |

Esses valores devem servir como baseline de comparação, considerando o intervalo de variação observado na validação cruzada.

---

### 19. Valores de Referência do Random Forest

Resultados de referência do conjunto de teste com threshold `0,50`:

| Métrica | Referência |
|---|---:|
| Accuracy | 0,7729 |
| Precision | 0,5544 |
| Recall | 0,7353 |
| F1-Score | 0,6322 |
| ROC-AUC | 0,8401 |
| Average Precision | 0,6492 |
| Falsos Positivos | 221 |
| Falsos Negativos | 99 |
| Custo relativo total | 716 |

O Random Forest é o modelo recomendado para o cenário de negócio.

---

### 20. Alertas de Desempenho

Limites iniciais sugeridos:

| Indicador | Atenção | Crítico |
|---|---:|---:|
| Queda de Average Precision | > 5% | > 10% |
| Queda de Recall | > 5% | > 10% |
| Queda de F1-Score | > 5% | > 10% |
| Aumento de Falsos Negativos | > 10% | > 20% |
| Aumento do custo relativo | > 10% | > 20% |
| PSI de uma variável | ≥ 0,10 | > 0,25 |
| Categorias desconhecidas | > 1% | > 5% |
| Entradas inválidas | > 5% | > 10% |

As variações devem ser avaliadas em janelas com volume suficiente para evitar alertas causados por amostras pequenas.

---

### 21. Custo Relativo dos Erros

A simulação utiliza:

```text
Custo de um Falso Positivo = 1
Custo de um Falso Negativo = 5
```

Fórmula:

```text
Custo relativo total =
    Falsos Positivos × 1
    +
    Falsos Negativos × 5
```

Os valores utilizados são hipotéticos.

Antes da utilização real, deverão ser estimados:

- custo médio de uma campanha;
- custo de incentivo ou desconto;
- valor médio do cliente;
- perda de receita provocada pelo churn;
- probabilidade de sucesso da retenção;
- custo operacional da abordagem.

---

### 22. Monitoramento do Threshold

O threshold deve ser tratado como uma decisão operacional.

Referências da MLP PyTorch:

| Finalidade | Threshold |
|---|---:|
| Padrão | 0,50 |
| Melhor F1 entre os avaliados | 0,30 |
| Menor custo na simulação | 0,20 |

Devem ser acompanhados para cada threshold:

- Precision;
- Recall;
- F1-Score;
- Falsos Positivos;
- Falsos Negativos;
- quantidade de clientes abordados;
- custo da campanha;
- custo das oportunidades perdidas.

Uma alteração do threshold deve ser aprovada pelo responsável de negócio e registrada como nova configuração.

---

### 23. Monitoramento por Segmentos

As métricas devem ser avaliadas também por segmentos, quando houver volume suficiente.

Exemplos:

- tipo de contrato;
- serviço de internet;
- forma de pagamento;
- tempo de relacionamento;
- faixa de cobrança mensal;
- indicador de idoso;
- utilização de suporte técnico;
- utilização de segurança online.

O objetivo é identificar:

- degradação concentrada;
- grupos com maior taxa de erro;
- possíveis vieses;
- segmentos com maior oportunidade de retenção.

A análise por grupos deve respeitar a LGPD e não deve ser utilizada para tratamento discriminatório.

---

### 24. Monitoramento de Equidade

A variável `gender` integra o dataset original e exige atenção especial.

Devem ser comparadas entre grupos:

- taxa prevista de churn;
- Precision;
- Recall;
- Falsos Positivos;
- Falsos Negativos;
- distribuição das probabilidades;
- taxa de abordagem.

Diferenças relevantes devem ser investigadas antes da utilização operacional.

A análise deve considerar:

- tamanho das amostras;
- possíveis variáveis correlacionadas;
- contexto do negócio;
- impacto das decisões;
- necessidade de revisão das features.

---

### 25. Monitoramento dos Artefatos

Devem ser verificados:

- existência dos arquivos;
- permissão de leitura;
- tamanho dos arquivos;
- hash;
- versão;
- compatibilidade das dependências;
- quantidade esperada de features;
- carregamento do `state_dict`;
- modo de avaliação do modelo;
- integridade dos metadados.

Artefatos atuais:

```text
models/
├── mlp_pytorch_state_dict.pt
├── mlp_pytorch_preprocessor.joblib
└── mlp_pytorch_metadata.json
```

Qualquer alteração deve gerar uma nova versão lógica do modelo.

---

### 26. Monitoramento com MLflow

O MLflow é utilizado para registrar:

- parâmetros;
- métricas;
- tags;
- tabelas;
- gráficos;
- resumos;
- resultados do hold-out;
- resultados da validação cruzada;
- custos relativos.

Experimento:

```text
churn-prediction-model-comparison
```

Tags importantes:

```text
recommended_for_business=true
central_neural_model=true
```

O Run ID e a versão do modelo devem ser relacionados às futuras implantações.

---

### 27. Frequência das Análises

| Análise | Frequência sugerida |
|---|---|
| Disponibilidade | Contínua |
| Taxa de erros | Contínua |
| Latência | Contínua |
| Volume de requisições | Contínua |
| Entradas inválidas | Diária |
| Distribuição das previsões | Diária |
| Qualidade dos dados | Diária |
| Data drift | Semanal |
| Desempenho com rótulos reais | Mensal |
| Custo relativo | Mensal |
| Equidade por segmentos | Mensal |
| Revisão do threshold | Trimestral |
| Revisão completa do modelo | Trimestral ou por evento |
| Auditoria dos artefatos | A cada implantação |

A frequência deverá ser ajustada ao volume real de dados e à velocidade de disponibilização dos rótulos.

---

### 28. Responsabilidades

| Papel | Responsabilidade |
|---|---|
| Engenharia de ML | Modelo, pipeline, métricas, drift e artefatos |
| Engenharia de Software | API, testes, disponibilidade e latência |
| Engenharia de Dados | Qualidade, disponibilidade e schema dos dados |
| Negócio | Custos, threshold e estratégia de retenção |
| Segurança | Acessos, logs, segredos e vulnerabilidades |
| Governança | LGPD, auditoria, documentação e aprovação |

No contexto acadêmico, essas responsabilidades estão centralizadas na autora do projeto.

---

### 29. Classificação dos Incidentes

| Severidade | Descrição |
|---|---|
| SEV-1 | API indisponível, modelo não carregado ou previsões impossíveis |
| SEV-2 | Degradação relevante, alta taxa de erros ou drift crítico |
| SEV-3 | Alerta moderado sem interrupção da operação |
| SEV-4 | Problema documental ou melhoria sem impacto imediato |

---

### 30. Playbook — API Indisponível

Ações:

1. consultar o endpoint `/health`;
2. verificar os logs de inicialização;
3. confirmar a existência dos artefatos;
4. verificar permissões dos arquivos;
5. validar as versões de Python, PyTorch e Scikit-Learn;
6. executar o smoke test de inferência;
7. reiniciar a aplicação;
8. realizar rollback se o problema começou após implantação;
9. registrar o incidente e a causa;
10. executar os testes antes de uma nova implantação.

---

### 31. Playbook — Aumento de Latência

Ações:

1. confirmar a latência no middleware;
2. comparar com a latência interna da inferência;
3. verificar volume de requisições;
4. verificar CPU e memória;
5. verificar reinicializações;
6. analisar o tempo do pré-processamento;
7. analisar o tempo do modelo;
8. revisar a quantidade de workers;
9. executar teste de carga;
10. avaliar escalabilidade horizontal.

---

### 32. Playbook — Aumento de Erros 422

Ações:

1. identificar os campos responsáveis;
2. comparar o payload com o contrato da API;
3. verificar alterações no sistema de origem;
4. verificar novas categorias;
5. confirmar tipos e unidades;
6. comunicar o produtor dos dados;
7. não relaxar o schema sem análise;
8. adicionar teste para o novo cenário;
9. documentar qualquer alteração aprovada.

---

### 33. Playbook — Drift de Dados

Ações:

1. identificar as variáveis afetadas;
2. confirmar se houve mudança legítima no negócio;
3. verificar falhas de coleta;
4. comparar distribuições;
5. analisar previsões por segmento;
6. verificar degradação das métricas;
7. avaliar necessidade de novo treinamento;
8. validar o modelo candidato;
9. registrar o experimento no MLflow;
10. implantar somente após aprovação.

---

### 34. Playbook — Degradação do Modelo

Ações:

1. confirmar a disponibilidade dos rótulos reais;
2. recalcular as métricas;
3. analisar a matriz de confusão;
4. verificar aumento dos Falsos Negativos;
5. calcular o custo relativo;
6. analisar métricas por segmento;
7. verificar data drift e concept drift;
8. revisar o threshold;
9. avaliar novo treinamento;
10. comparar o candidato com o modelo atual;
11. registrar resultados no MLflow;
12. realizar rollback ou promover nova versão.

---

### 35. Critérios para Retreinamento

O retreinamento deve ser considerado quando ocorrer:

- queda persistente superior a 10% na Average Precision;
- queda persistente superior a 10% no Recall;
- PSI superior a 0,25 em variáveis relevantes;
- aumento superior a 20% nos Falsos Negativos;
- aumento superior a 20% no custo relativo;
- inclusão ou remoção de produtos;
- mudança na definição de churn;
- alteração importante dos contratos;
- mudança relevante no comportamento dos clientes;
- acúmulo de volume suficiente de novos dados rotulados;
- mudança aprovada das features.

O retreinamento não deve resultar em implantação automática sem validação.

---

### 36. Critérios para Rollback

O rollback deve ser considerado quando:

- a aplicação não consegue carregar o modelo;
- a nova versão aumenta erros HTTP;
- a latência ultrapassa o limite crítico;
- o schema se torna incompatível;
- as probabilidades apresentam comportamento inesperado;
- ocorre degradação relevante das métricas;
- os artefatos estão corrompidos;
- o custo relativo aumenta de maneira relevante.

A versão anterior deve permanecer disponível e identificada.

---

### 37. Evidências e Auditoria

Devem ser preservados:

- versão do código;
- commit do Git;
- versão do modelo;
- Run ID do MLflow;
- parâmetros;
- métricas;
- hash dos artefatos;
- data do treinamento;
- período dos dados;
- threshold;
- responsáveis pela aprovação;
- registros de implantação;
- incidentes;
- decisões de rollback;
- decisões de retreinamento.

---

### 38. Privacidade e LGPD

O monitoramento deve seguir os princípios de:

- finalidade;
- adequação;
- necessidade;
- segurança;
- prevenção;
- transparência;
- responsabilização.

Não devem ser armazenados nos logs:

- dados completos do cliente;
- informações pessoais desnecessárias;
- payload integral;
- credenciais;
- tokens;
- segredos;
- informações financeiras sensíveis.

Os dados utilizados para monitoramento devem ser agregados ou anonimizados sempre que possível.

---

### 39. Ferramentas Sugeridas

Possíveis ferramentas para uma evolução produtiva:

| Finalidade | Ferramentas possíveis |
|---|---|
| Métricas da API | Prometheus |
| Dashboards | Grafana |
| Logs | ELK, OpenSearch ou serviço de nuvem |
| Tracing | OpenTelemetry |
| Experimentos | MLflow |
| Drift | Evidently ou implementação própria |
| Alertas | Alertmanager, e-mail ou ferramenta corporativa |
| Infraestrutura | Docker e Kubernetes |
| CI/CD | GitHub Actions |
| Segurança | Scanner de dependências e gestão de segredos |

Essas ferramentas são recomendações de evolução e não representam componentes já implantados.

---

### 40. Situação Atual

Já foram implementados:

- endpoint `/health`;
- endpoint `/predict`;
- validação das entradas;
- tratamento de erros;
- logging estruturado;
- Request ID;
- medição de latência;
- headers de observabilidade;
- testes automatizados;
- rastreamento de experimentos com MLflow;
- persistência dos artefatos;
- validação de schema com Pandera;
- validação estática com Ruff.

Situação dos testes:

```text
44 passed
```

Validação estática:

```text
All checks passed!
```

Ainda são evoluções futuras:

- armazenamento centralizado dos logs;
- dashboards operacionais;
- alertas automáticos;
- coleta de métricas em produção;
- monitoramento automático de drift;
- implantação em nuvem;
- ciclo automatizado de retreinamento.

---

### 41. Conclusão

O plano de monitoramento combina indicadores operacionais, qualidade dos dados, desempenho do modelo e impacto de negócio.

A separação entre alertas, critérios de retreinamento, critérios de rollback e playbooks permite responder de forma estruturada a falhas ou degradações.

Os limites apresentados são referências iniciais. Em uma implantação real, deverão ser recalibrados com base no volume de requisições, no comportamento dos dados, nos custos reais e nos objetivos das ações de retenção.

---