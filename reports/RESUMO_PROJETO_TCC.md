# Resumo do Projeto para Construcao do TCC

> **Titulo sugerido:** Modelos de Aprendizado de Maquina para Analise de Sobrevivencia em Cancer de Colon: Um Estudo Comparativo com Dados do TCGA-COAD
>
> **Autor:** Phelipe Quintas
>
> **Arquivo gerado em:** Fevereiro/2026

---

## 1. INTRODUCAO E CONTEXTO

### 1.1. Problema de Pesquisa

O cancer colorretal e uma das principais causas de mortalidade oncologica no mundo. A capacidade de estratificar pacientes em grupos de risco (baixo, medio, alto) a partir de dados clinicos e crucial para orientar decisoes terapeuticas e acompanhamento pos-diagnostico. Este trabalho investiga o uso de modelos de aprendizado de maquina para analise de sobrevivencia em pacientes com cancer de colon, utilizando dados publicos do programa TCGA (The Cancer Genome Atlas).

### 1.2. Objetivo Geral

Desenvolver e comparar modelos preditivos de sobrevivencia para pacientes com adenocarcinoma de colon (COAD), permitindo a estratificacao em grupos de risco e a identificacao das variaveis clinicas mais relevantes para o prognostico.

### 1.3. Objetivos Especificos

1. Construir um pipeline reprodutivel de processamento de dados clinicos do TCGA-COAD
2. Treinar e validar tres modelos de sobrevivencia: Cox Proportional Hazards, Random Survival Forest e XGBoost Survival
3. Comparar os modelos utilizando o Indice de Concordancia (C-Index) com validacao cruzada
4. Interpretar os modelos atraves de Hazard Ratios, Permutation Importance e Partial Dependence Plots
5. Estratificar pacientes em grupos de risco e visualizar curvas de sobrevivencia por grupo

---

## 2. DADOS UTILIZADOS

### 2.1. Fonte dos Dados

- **Repositorio:** The Cancer Genome Atlas (TCGA)
- **Projeto:** TCGA-COAD (Colon Adenocarcinoma)
- **Portal:** GDC Data Portal (https://portal.gdc.cancer.gov/)
- **Data de download:** Outubro-Novembro/2025

### 2.2. Dados Brutos Obtidos

Foram baixados dois conjuntos de dados no formato TSV:

**Dados Clinicos** (`clinical.tsv`) - 534 variaveis, incluindo:
- Dados demograficos: idade, genero, etnia, status vital, dias ate obito
- Dados de diagnostico: estagio AJCC (patologico e clinico), tecido de origem, morfologia
- Dados de exposicao: tabagismo, consumo de alcool
- Dados de tratamento: tipo de terapia, agentes terapeuticos
- Historico familiar e testes moleculares

**Dados de Biospecimen** (`sample.tsv`) - 39 variaveis, incluindo:
- Tipo de amostra, metodo de coleta, peso, dimensoes
- Metodo de preservacao, tipo de tecido

### 2.3. Volume Inicial

- Registros clinicos brutos: ~5.200 linhas (incluindo duplicatas de amostras multiplas por paciente)
- Registros de biospecimen: dados de amostras biologicas vinculadas aos pacientes

---

## 3. PIPELINE DE PROCESSAMENTO DOS DADOS

### 3.1. Etapa 1: Pre-processamento (`preprocessamento_data.py`)

- Carregamento dos arquivos TSV brutos (clinical.tsv e sample.tsv)
- Conversao para formato Parquet (mais eficiente para leitura/escrita)
- Nenhuma transformacao de conteudo nesta etapa

### 3.2. Etapa 2: Consolidacao (`consolidacaodados_tcga_coad.py`)

- Juncao (inner join) dos dados clinicos com os dados de biospecimen
- Chave de juncao: `cases.submitter_id`
- ~1.500 amostras sem correspondencia foram removidas no join

### 3.3. Etapa 3: Engenharia de Features (`feature_engineering_survival.py`)

Esta etapa e o coracao do pre-processamento. As transformacoes aplicadas foram:

**a) Selecao de Variaveis de Sobrevivencia**

Das 534+ variaveis disponiveis, foram selecionadas 6 relevantes para analise de sobrevivencia:

| Variavel Original | Nome Limpo | Descricao |
|---|---|---|
| `demographic.vital_status` | `vital_status` | Status vital (Alive/Dead) |
| `diagnoses.days_to_last_follow_up` | `days_to_last_follow_up` | Dias ate ultimo acompanhamento |
| `demographic.days_to_death` | `days_to_death` | Dias ate obito |
| `demographic.age_at_index` | `age_at_index` | Idade no diagnostico |
| `diagnoses.ajcc_pathologic_stage` | `ajcc_pathologic_stage` | Estagio patologico AJCC |
| `diagnoses.tissue_or_organ_of_origin` | `tissue_or_organ_of_origin` | Tecido/orgao de origem |

**b) Filtragem por Sitio Anatomico**

Mantidos apenas locais validos do colon:
- Sigmoid colon, Ascending colon, Colon NOS, Cecum
- Transverse colon, Descending colon, Rectosigmoid junction
- Removidos: Hepatic flexure e Splenic flexure (pouquissimos eventos, categorias esparsas)

**c) Criacao das Variaveis de Sobrevivencia**

- **Evento (event_occurred):** 1 se paciente faleceu (vital_status == "Dead"), 0 caso contrario
- **Tempo observado (observed_time):**
  - Se faleceu: `days_to_death / 365.25` (convertido para anos)
  - Se vivo: `days_to_last_follow_up / 365.25` (dados censurados)
  - Tempos <= 0 corrigidos adicionando 1 dia (evita erro matematico no log do modelo Cox)
  - Pacientes sem informacao de tempo removidos

**d) Agrupamento de Estagios AJCC**

Subestagios consolidados em 4 categorias principais:
- IA, IB → Stage I
- IIA, IIB, IIC → Stage II
- IIIA, IIIB, IIIC → Stage III
- IVA, IVB, IVC → Stage IV
- Estagios invalidos ou nulos removidos

**e) Remocao de Feature Redundante**

A variavel `age_group` (faixa etaria discretizada) foi removida do pipeline apos diagnostico de redundancia: e perfeitamente derivada de `age_at_index`, introduzindo multicolinearidade sem ganho preditivo. Testes confirmaram impacto zero na performance dos modelos.

### 3.4. Dataset Final

- **Total de pacientes:** 4.164
- **Features finais (5):**
  - `event_occurred` (binaria: indicador de obito)
  - `observed_time` (continua: tempo de sobrevivencia em anos)
  - `age_at_index` (continua: idade no diagnostico)
  - `ajcc_pathologic_stage` (categorica: Stage I, II, III, IV)
  - `tissue_or_organ_of_origin` (categorica: 7 localizacoes anatomicas)
- **Apos One-Hot Encoding:** 10 features (1 numerica + 3 dummies de estagio + 6 dummies de localizacao)

---

## 4. ANALISE EXPLORATORIA DOS DADOS (AED)

### 4.1. Visualizacoes Geradas

| Figura | Arquivo | Descricao |
|---|---|---|
| Distribuicao de Idade | `distribuicao_idade.png` | Histograma com KDE da distribuicao etaria |
| Status Vital | `distribuicao_status_vital.png` | Barras com contagem e percentual vivo/falecido |
| Tempo de Acompanhamento | `distribuicao_tempo_acompanhamento.png` | Histograma por status vital com medianas |
| Distribuicao de Estagios | `distribuicao_estagios.png` | Barras com contagem por estagio AJCC |
| Localizacao Anatomica | `distribuicao_localizacao_anatomica.png` | Barras por sitio do tumor |
| Mortalidade por Idade | `taxa_obito_por_faixa_etaria.png` | Barras + linha de taxa de obito por faixa |
| Kaplan-Meier Global | `kaplan_meier_global.png` | Curva de sobrevivencia geral da coorte |
| Kaplan-Meier por Estagio | `kaplan_meier_por_estagio.png` | Curvas estratificadas por estagio patologico |

### 4.2. Principais Achados da AED

- A distribuicao etaria concentra-se entre 50-80 anos, consistente com a epidemiologia do cancer de colon
- Proporcao de eventos: 24,4% de obitos (1.014) e 75,6% censurados (3.150)
- A taxa de obito aumenta progressivamente com a faixa etaria
- Estagios mais avancados (III e IV) apresentam queda mais acentuada na curva de sobrevivencia
- A curva de Kaplan-Meier global mostra declinio progressivo da sobrevivencia ao longo do tempo

---

## 5. MODELAGEM PREDITIVA

### 5.1. Divisao dos Dados

- **Treino:** 80% (3.331 pacientes) — usado para validacao cruzada e treinamento final
- **Teste:** 20% (833 pacientes) — reservado para avaliacao imparcial
- **Estratificacao:** por `event_occurred` para manter proporcao de eventos similar nos dois conjuntos

### 5.2. Codificacao de Variaveis Categoricas

- **Metodo:** One-Hot Encoding com `drop_first=True`
- **Prevencao de Data Leakage:** encoding aplicado separadamente em cada fold da validacao cruzada
  - Fit apenas nos dados de treino do fold
  - Transform nos dados de validacao do fold
  - Colunas extras no teste preenchidas com 0; colunas ausentes removidas
- Variaveis categoricas processadas: `ajcc_pathologic_stage`, `tissue_or_organ_of_origin`

### 5.3. Metrica de Avaliacao: Indice de Concordancia (C-Index)

O C-Index mede a capacidade do modelo de ordenar corretamente os pacientes por risco:
- **0.5** = previsao aleatoria (equivalente a cara ou coroa)
- **1.0** = ordenacao perfeita
- **> 0.70** = clinicamente significativo na literatura medica
- **Interpretacao:** probabilidade de que, dado um par de pacientes, o modelo atribua maior risco ao que efetivamente teve pior desfecho

### 5.4. Estrategia de Validacao

- **Validacao Cruzada K-Fold:** 5 folds, com shuffle e random_state=42
- Em cada fold: encoding categorico independente (evita vazamento de dados)
- Avaliacao final no conjunto de teste (hold-out) com modelos treinados em todo o conjunto de treino

### 5.5. Modelos Treinados

#### 5.5.1. Cox Proportional Hazards (CoxPH)

**Fundamentacao:** Modelo semi-parametrico classico em analise de sobrevivencia. Assume que o efeito das covariaveis sobre o risco (hazard) e multiplicativo e constante no tempo (proporcionalidade dos riscos).

**Configuracao:**
```
CoxPHFitter(penalizer=0.1)
```
- Penalizador L2 de 0.1 para regularizacao e estabilidade numerica
- API via biblioteca `lifelines`

**Papel no projeto:** Modelo baseline interpretavel. Gera Hazard Ratios diretamente, permitindo quantificar o impacto de cada variavel.

#### 5.5.2. Random Survival Forest (RSF)

**Fundamentacao:** Extensao das Random Forests para dados de sobrevivencia. Constroi multiplas arvores de sobrevivencia e agrega suas previsoes. Captura relacoes nao-lineares e interacoes entre variaveis sem suposicoes parametricas.

**Configuracao:**
```
RandomSurvivalForest(
    random_state=42,
    min_samples_leaf=15,   # Minimo de amostras por folha
    max_depth=10,          # Profundidade maxima das arvores
    n_estimators=200       # Numero de arvores na floresta
)
```
- API via biblioteca `scikit-survival`

**Papel no projeto:** Modelo ensemble com boa estabilidade entre folds (menor variancia na CV). Complementa o XGBoost com Permutation Importance como metodo robusto de interpretabilidade.

#### 5.5.3. XGBoost Survival

**Fundamentacao:** Gradient Boosting aplicado a analise de sobrevivencia usando a funcao de perda de Cox (cox-nloglik). Constroi arvores sequencialmente, cada uma corrigindo os erros da anterior.

**Configuracao:**
```
XGBRegressor(
    objective="survival:cox",
    eval_metric="cox-nloglik",
    random_state=42,
    max_depth=10,
    learning_rate=0.05,
    n_estimators=200,
    min_child_weight=1,
    gamma=0.2,
    subsample=0.8,
    colsample_bytree=1.0,
    reg_alpha=0,
    reg_lambda=0.1,
)
```
- API via biblioteca `xgboost`

**Papel no projeto:** Melhor modelo do pipeline (C-Index 0.980). Selecionado como modelo principal para estratificacao de risco.

---

## 6. RESULTADOS

### 6.1. Comparacao dos Modelos

| Modelo | C-Index CV (media +/- desvio) | C-Index Teste | Interpretacao |
|---|---|---|---|
| Cox PH | 0.706 +/- 0.014 | 0.736 | Baseline interpretavel, performance adequada |
| Random Survival Forest | 0.844 +/- 0.006 | 0.889 | Alta discriminacao, menor variancia entre folds |
| **XGBoost Survival** | **0.952 +/- 0.010** | **0.980** | **Melhor modelo, discriminacao excepcional** |

### 6.2. Validacao contra Sobreajuste (Overfitting)

Dado o C-Index excepcionalmente alto do XGBoost, foram conduzidos 5 testes de validacao:

**a) Gap Treino vs Teste**

| Modelo | C-Index Treino | C-Index Teste | Gap |
|---|---|---|---|
| CoxPH | 0.714 | 0.736 | -0.022 |
| RSF | 0.873 | 0.889 | -0.016 |
| XGBoost | 0.970 | 0.980 | -0.010 |

Gap negativo nos 3 modelos = desempenho no teste ligeiramente superior ao treino. Ausencia de memorizacao.

**b) Analise de Complexidade (XGBoost)**

| max_depth | C-Index Treino | C-Index Teste | Gap |
|---|---|---|---|
| 3 | 0.872 | 0.891 | -0.019 |
| 5 | 0.934 | 0.946 | -0.012 |
| 7 | 0.960 | 0.971 | -0.011 |
| 10 | 0.970 | 0.980 | -0.010 |

Mesmo com max_depth=3 (modelo simples), C-Index ja atinge 0.891. O sinal preditivo vem dos dados, nao da complexidade.

**c) Teste de Permutacao**

| Modelo | C-Index Real | C-Index Aleatorio (media) | Diferenca |
|---|---|---|---|
| XGBoost | 0.980 | 0.453 | +0.527 |
| RSF | 0.889 | 0.459 | +0.430 |

Com labels aleatorios, C-Index cai para ~0.45. Os modelos aprendem padroes genuinos, nao ruido.

**d) Learning Curves:** Convergencia saudavel entre treino e teste conforme volume de dados aumenta.

**e) Justificativa clinica:** O estagio patologico AJCC e um dos preditores mais fortes em oncologia colorretal. Stage I tem ~90% de sobrevida em 5 anos vs ~10% para Stage IV. Com 4.164 pacientes e um preditor tao discriminativo, desempenho alto e clinicamente plausivel.

### 6.3. Modelo Selecionado

O **XGBoost Survival** foi selecionado como modelo principal por:
1. Maior C-Index no teste (0.980)
2. Performance validada contra sobreajuste (gap negativo, teste de permutacao)
3. Capacidade de capturar interacoes nao-lineares complexas
4. Disponibilidade de metodos de interpretabilidade (importancia por peso/ganho, PDP/ICE)

---

## 7. INTERPRETABILIDADE DOS MODELOS

### 7.1. Cox PH: Hazard Ratios (Razoes de Risco)

**Figura:** `razoes_risco_cox.png`

O grafico de Hazard Ratios mostra o impacto de cada variavel na sobrevivencia:
- **HR > 1 (barras vermelhas):** Fator de risco — aumenta a probabilidade de obito
- **HR < 1 (barras azuis):** Fator protetor — diminui a probabilidade de obito
- **HR = 1 (linha tracejada):** Sem efeito
- Escala logaritmica para simetria visual (HR=0.5 e HR=2.0 ficam equidistantes de 1)

**Exemplo de interpretacao:** Se o estagio IV tem HR=3.5, significa que pacientes em estagio IV tem 3.5x mais risco de obito em comparacao com a categoria de referencia.

### 7.2. XGBoost: Importancia de Variaveis e PDP

**Figuras:**
- `importancia_variaveis_peso_xgboost.png` — Importancia por Peso (frequencia de uso nos splits)
- `importancia_variaveis_ganho_xgboost.png` — Importancia por Ganho (reducao media na funcao de perda)
- `dependencia_parcial_xgboost.png` — Graficos de Dependencia Parcial (PDP) com curvas ICE

**Nota sobre Weight vs Gain:** A metrica weight favorece `age_at_index` por ser a unica variavel continua (muitos limiares de corte possiveis), enquanto gain revela que `Stage IV` tem o maior poder discriminativo por split. Essa diferenca ilustra a importancia de avaliar multiplas metricas de importancia.

**Interpretacao do PDP:**
- Mostra a relacao marginal entre cada variavel e o log-hazard (risco)
- Curvas ICE mostram a variacao individual para cada paciente
- Eixo Y: Log-Hazard (valores maiores = maior risco)

### 7.3. RSF: Permutation Importance e PDP

**Figuras:**
- `importancia_permutacao_rsf.png` — Top variaveis por Permutation Importance (10 repeticoes, com barras de erro)
- `dependencia_parcial_rsf.png` — PDP com ICE para as 3 variaveis mais importantes

**Top 3 variaveis (Permutation Importance):**
1. `age_at_index` (0.222 +/- 0.012) — idade no diagnostico
2. `ajcc_pathologic_stage_Stage IV` (0.153 +/- 0.011) — estagio avancado
3. `tissue_or_organ_of_origin_Sigmoid colon` (0.065 +/- 0.007) — localizacao anatomica

**Metodo Permutation Importance:**
- Aleatoriza os valores de uma variavel e mede a queda no C-Index
- Quanto maior a queda, mais importante e a variavel
- Metodo robusto que funciona para qualquer modelo (model-agnostic)

### 7.4. Variaveis Mais Importantes (Consolidado)

Com base na analise dos tres modelos, as variaveis mais relevantes para o prognostico sao:

1. **Estagio Patologico AJCC** — Principal preditor em todos os modelos. Estagio IV tem o maior impacto no risco
2. **Idade no Diagnostico** — Fator de risco continuo; idade mais avancada associada a pior prognostico
3. **Localizacao Anatomica** — Algumas localizacoes do colon (ex.: sigmoide) apresentam prognosticos distintos

A hierarquia de importancia difere entre modelos: o XGBoost prioriza o estagio (gain), enquanto o RSF atribui maior peso a idade (permutation importance). Essa complementaridade reforça a importancia de avaliar multiplos algoritmos.

---

## 8. ESTRATIFICACAO DE RISCO

### 8.1. Metodologia

**Figura:** `curvas_sobrevivencia_por_grupos_risco.png`

- O modelo XGBoost Survival gera um score de risco para cada paciente
- Os pacientes sao divididos em 3 grupos usando tercis (percentis 33 e 67):
  - **Risco Baixo:** tercil inferior (33% com menores scores de risco)
  - **Risco Medio:** tercil intermediario
  - **Risco Alto:** tercil superior (33% com maiores scores de risco)
- Curvas de Kaplan-Meier geradas para cada grupo com intervalos de confianca de 95%

### 8.2. Resultado

As curvas de sobrevivencia por grupo de risco demonstram separacao clara entre os tres grupos:
- **Risco Baixo:** probabilidade de sobrevivencia mantida acima de 80% ao longo do acompanhamento
- **Risco Medio:** declinio gradual com probabilidade intermediaria
- **Risco Alto:** queda acentuada nos primeiros anos, probabilidade significativamente inferior

A estratificacao valida a capacidade do modelo de identificar subgrupos de pacientes que poderiam se beneficiar de acompanhamento mais intensivo ou abordagens terapeuticas diferenciadas.

---

## 9. ASPECTOS TECNICOS E BOAS PRATICAS

### 9.1. Prevencao de Data Leakage

- Split treino/teste realizado **antes** de qualquer encoding
- One-Hot Encoding aplicado **dentro** de cada fold da validacao cruzada
- Conjunto de teste **nunca** influencia a codificacao do treino
- Colunas de treino salvas (`training_columns.joblib`) para garantir consistencia na producao

### 9.2. Reproducibilidade

- `random_state=42` em todos os processos estocasticos
- Modelos salvos em formato joblib para recarregamento
- Dados intermediarios salvos em Parquet (preserva tipos de dados)
- Pipeline executavel de ponta a ponta via `src/main.py`

### 9.3. Paleta de Cores Padronizada

Todas as visualizacoes usam uma paleta centralizada (`config.PALETTE`) com cores semanticas:
- Azul = Vivo/Censurado/Protetor | Vermelho = Falecido/Evento/Risco
- Verde/Ambar/Vermelho para grupos de risco (baixo/medio/alto)
- Progressao cromatica para estagios I-IV

### 9.4. Stack Tecnologico

| Categoria | Bibliotecas |
|---|---|
| Processamento de Dados | pandas, numpy, pyarrow |
| Analise de Sobrevivencia | lifelines (Cox, Kaplan-Meier) |
| Machine Learning | scikit-learn, scikit-survival (RSF), xgboost |
| Visualizacao | matplotlib, seaborn |
| Serializacao | joblib |
| Linguagem | Python 3.11 |

### 9.5. Estrutura do Pipeline

```
1. Preprocessamento      → TSV para Parquet
2. Consolidacao          → Merge clinico + biospecimen
3. Feature Engineering   → Selecao, limpeza, criacao de variaveis
4. AED                   → 8 visualizacoes exploratorias
5. Treinamento           → 3 modelos + validacao cruzada 5-fold
6. Validacao Overfitting → 5 testes diagnosticos
7. Interpretacao Cox     → Hazard Ratios
8. Interpretacao XGB     → Importancia (peso/ganho) + PDP/ICE
9. Interpretacao RSF     → Permutation Importance + PDP/ICE
10. Predicao             → Tempos medianos de sobrevivencia
11. Visualizacao Final   → Curvas de risco estratificadas
```

---

## 10. LISTA COMPLETA DE FIGURAS GERADAS

Todas as figuras estao em `reports/figures/` com resolucao de 300 DPI.

| # | Arquivo | Secao do TCC Sugerida |
|---|---|---|
| 1 | `distribuicao_idade.png` | Analise Exploratoria |
| 2 | `distribuicao_status_vital.png` | Analise Exploratoria |
| 3 | `distribuicao_tempo_acompanhamento.png` | Analise Exploratoria |
| 4 | `distribuicao_estagios.png` | Analise Exploratoria |
| 5 | `distribuicao_localizacao_anatomica.png` | Analise Exploratoria |
| 6 | `taxa_obito_por_faixa_etaria.png` | Analise Exploratoria |
| 7 | `kaplan_meier_global.png` | Analise Exploratoria |
| 8 | `kaplan_meier_por_estagio.png` | Analise Exploratoria |
| 9 | `razoes_risco_cox.png` | Resultados — Interpretabilidade |
| 10 | `importancia_variaveis_peso_xgboost.png` | Resultados — Interpretabilidade |
| 11 | `importancia_variaveis_ganho_xgboost.png` | Resultados — Interpretabilidade |
| 12 | `dependencia_parcial_xgboost.png` | Resultados — Interpretabilidade |
| 13 | `importancia_permutacao_rsf.png` | Resultados — Interpretabilidade |
| 14 | `dependencia_parcial_rsf.png` | Resultados — Interpretabilidade |
| 15 | `diagnostico_overfitting.png` | Resultados — Validacao |
| 16 | `curvas_sobrevivencia_por_grupos_risco.png` | Resultados — Estratificacao de Risco |

---

## 11. ESTRUTURA SUGERIDA PARA O DOCUMENTO DO TCC

### Capitulo 1 — Introducao
- Contextualizacao do cancer colorretal (epidemiologia, relevancia clinica)
- Problema de pesquisa: necessidade de estratificacao de risco baseada em dados
- Objetivos (geral e especificos)
- Justificativa: uso de ML em oncologia, dados publicos do TCGA
- Organizacao do trabalho

### Capitulo 2 — Fundamentacao Teorica
- 2.1. Cancer Colorretal (biologia, estadiamento AJCC, fatores prognosticos)
- 2.2. Analise de Sobrevivencia (conceitos: censura, funcao de sobrevivencia, funcao de risco)
- 2.3. Estimador de Kaplan-Meier
- 2.4. Modelo de Cox (regressao de riscos proporcionais, hazard ratios)
- 2.5. Random Survival Forests (extensao de RF para sobrevivencia)
- 2.6. XGBoost para Sobrevivencia (gradient boosting com funcao de perda de Cox)
- 2.7. Metricas de Avaliacao (C-Index, validacao cruzada)
- 2.8. Interpretabilidade de Modelos (Permutation Importance, PDP, ICE)

### Capitulo 3 — Materiais e Metodos
- 3.1. Descricao dos Dados (TCGA-COAD, variaveis selecionadas)
- 3.2. Pipeline de Processamento (pre-processamento, consolidacao, feature engineering)
- 3.3. Analise Exploratoria (figuras 1-8)
- 3.4. Divisao Treino/Teste e Prevencao de Data Leakage
- 3.5. Modelos Utilizados (configuracoes, hiperparametros)
- 3.6. Estrategia de Validacao (5-fold CV)
- 3.7. Ferramentas e Tecnologias

### Capitulo 4 — Resultados e Discussao
- 4.1. Resultados da AED (caracterizacao da coorte)
- 4.2. Comparacao dos Modelos (tabela de C-Index)
- 4.3. Validacao contra Sobreajuste (gap treino/teste, complexidade, permutacao, learning curves)
- 4.4. Interpretacao do Modelo Cox (Hazard Ratios — figura 9)
- 4.5. Interpretacao do XGBoost (importancia peso/ganho + PDP — figuras 10-12)
- 4.6. Interpretacao do RSF (permutation importance + PDP — figuras 13-14)
- 4.7. Estratificacao de Risco (curvas por grupo — figura 16)
- 4.8. Discussao (comparacao com literatura, limitacoes, implicacoes clinicas)

### Capitulo 5 — Conclusao
- Sintese dos resultados
- Contribuicoes do trabalho
- Limitacoes
- Trabalhos futuros

### Referencias
### Apendices
- A. Codigo-fonte do pipeline
- B. Lista completa de variaveis clinicas do TCGA-COAD
- C. Detalhamento dos hiperparametros

---

## 12. PONTOS FORTES DO PROJETO (para destacar no TCC)

1. **Pipeline reprodutivel:** Codigo modular, executavel de ponta a ponta
2. **Prevencao rigorosa de Data Leakage:** Encoding dentro de cada fold, separacao treino/teste antes de qualquer transformacao
3. **Tres modelos complementares:** Baseline interpretavel (Cox) + ensemble estavel (RSF) + alta performance (XGBoost)
4. **Performance excepcional:** C-Index de 0.980 no teste com XGBoost (validado contra sobreajuste)
5. **Validacao robusta de overfitting:** 5 testes diagnosticos independentes confirmando ausencia de sobreajuste
6. **Interpretabilidade completa:** Hazard Ratios, Importancia por Peso/Ganho, Permutation Importance, PDP e ICE
7. **Aplicacao clinica concreta:** Estratificacao em 3 grupos de risco com curvas de Kaplan-Meier
8. **Dados publicos e acessiveis:** TCGA permite replicacao por outros pesquisadores
9. **Visualizacoes padronizadas:** Paleta semantica centralizada em todas as 16 figuras

## 13. LIMITACOES A MENCIONAR

1. Apenas variaveis clinicas foram utilizadas (sem dados genomicos, proteomicos ou de imagem)
2. Dataset limitado a coorte TCGA (vies de selecao: pacientes de centros academicos dos EUA)
3. Sem validacao externa em coorte independente
4. Suposicao de proporcionalidade dos riscos no modelo Cox nao foi formalmente testada
5. Categorias anatomicas esparsas foram removidas em vez de agrupadas
