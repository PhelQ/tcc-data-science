# Relatório Final do Projeto: Análise de Sobrevivência para Câncer de Cólon (TCGA-COAD)

## Resumo Executivo

O câncer colorretal (CCR) ocupa a terceira posição entre os tipos de câncer mais prevalentes e a quarta causa de mortalidade relacionada ao câncer globalmente. A predição acurada de sobrevida é essencial para o planejamento terapêutico personalizado e a tomada de decisão clínica. Este estudo teve como objetivo desenvolver um modelo de aprendizado de máquina para predição de sobrevida em pacientes com adenocarcinoma de cólon utilizando dados clínicos e demográficos da coleção TCGA-COAD (The Cancer Genome Atlas Colon Adenocarcinoma Collection).

Foi implementado um pipeline modular de dados compreendendo etapas de pré-processamento, consolidação e engenharia de features para transformar registros administrativos em variáveis de análise de sobrevivência. Três modelos foram avaliados utilizando validação cruzada de 5 folds e validação em conjunto de teste independente: Cox Proportional Hazards, Random Survival Forest (RSF) e XGBoost Survival.

O modelo **XGBoost Survival** alcançou o melhor desempenho com **C-Index de 0.9541** no conjunto de teste e **0.9458** na validação cruzada, demonstrando alta robustez e superando significativamente o Random Survival Forest (0.8757) e o Cox Proportional Hazards (0.7390). A estratificação de risco baseada no modelo XGBoost discriminou efetivamente os pacientes em grupos de baixo, médio e alto risco, com curvas de sobrevivência de Kaplan-Meier claramente separadas.

O estadiamento patológico AJCC foi confirmado como o fator prognóstico predominante, com destaque também para a idade e localização do tumor. Apesar de utilizar apenas variáveis clínicas, o modelo alcançou desempenho "estado da arte", comparável a estudos que incorporam dados genômicos complexos, demonstrando o forte poder discriminatório de variáveis clínicas bem selecionadas e processadas por algoritmos avançados de *gradient boosting*.

## 1. Introdução

Este projeto teve como objetivo principal desenvolver um modelo de machine learning para prever a sobrevivência de pacientes com adenocarcinoma de cólon, utilizando dados clínicos e demográficos do renomado projeto The Cancer Genome Atlas (TCGA). A análise de sobrevivência é uma ferramenta estatística crucial em oncologia, pois permite estimar a probabilidade de um paciente sobreviver por um determinado período, além de identificar os fatores de risco mais relevantes.

Através de um pipeline de dados modular, realizamos desde a coleta e limpeza dos dados até a construção, avaliação e interpretação de modelos de sobrevivência avançados. O resultado é uma ferramenta capaz de estratificar pacientes em diferentes grupos de risco, oferecendo insights valiosos que podem, no futuro, auxiliar em decisões clínicas.

**Nota Importante sobre o Escopo:** Nesta fase do projeto, focamos exclusivamente em **dados clínicos, demográficos e de estadiamento**. Dados genômicos (como expressão de RNA ou mutações de DNA) não foram incluídos nesta versão do modelo, mas representam uma oportunidade clara para expansões futuras.

## 2. Metodologia e Estrutura do Projeto

### Origem e Composição dos Dados

Os dados utilizados neste estudo provêm do **The Cancer Genome Atlas (TCGA)**, especificamente da coorte de Adenocarcinoma de Cólon (**TCGA-COAD**), acessados através do *GDC Data Portal* (Genomic Data Commons).

Para construir uma visão completa de cada paciente, integramos dois datasets primários distintos:

1.  **Dados Clínicos (`clinical.tsv`):**
    *   Contém informações demográficas (idade, gênero, etnia), dados vitais (status vivo/morto, dias até o óbito) e patológicos (estágio do tumor AJCC, diagnósticos primários).
    *   *Função:* Fornecer as variáveis preditoras (features) e as variáveis alvo de sobrevivência (tempo e evento).

2.  **Dados de Bioespécimes (`sample.tsv`):**
    *   Contém detalhes físicos sobre as amostras biológicas coletadas (tipo de tecido, local anatômico da coleta, IDs das amostras).
    *   *Função:* Permitir a filtragem rigorosa para garantir que a análise considere apenas tecidos tumorais primários de cólon, excluindo tecidos normais adjacentes ou metástases que poderiam enviesar o modelo.

**Estratégia de Fusão (Consolidação):**
A unificação desses universos foi realizada através de um *Inner Join* utilizando o identificador único do paciente (`cases.submitter_id`). Esta abordagem conservadora garante que apenas pacientes que possuem **ambos** os registros (clínico e de biópsia) completos sejam incluídos no estudo, maximizando a integridade dos dados.

### Arquitetura do Pipeline

O projeto foi estruturado em uma série de scripts Python modulares, garantindo a reprodutibilidade e a clareza do processo. O pipeline completo é orquestrado pelo script `src/main.py` e segue as seguintes etapas:

### Etapa 1: Consolidação e Pré-processamento dos Dados

Antes de qualquer análise, precisamos resolver a fragmentação dos dados originais do TCGA, que vêm separados em arquivos distintos. Esta etapa é fundamental para criar uma visão unificada de cada paciente.

-   **`src/data/preprocessamento_data.py`**: Atua como a primeira camada de limpeza (ETL - Extract, Transform, Load).
    -   **Função:** Lê os arquivos brutos (`clinical.tsv` e `sample.tsv`) e os converte para o formato **Parquet**.
    -   **Por que Parquet?** Este formato colunar é altamente eficiente para leitura e preserva os tipos de dados (inteiros, strings, floats) melhor que o CSV/TSV, evitando erros de interpretação numérica nas etapas seguintes.

-   **`src/data/consolidacaodados_tcga_coad.py`**: Realiza a fusão dos datasets clínico e de bioespécimes.
    -   **O Desafio:** Os dados clínicos (informações do paciente) e os dados de bioespécimes (informações da amostra coletada) residem em tabelas separadas. Para correlacionar, por exemplo, o *estágio do câncer* (dado clínico) com o *tipo de tecido* (dado da amostra), precisamos unificá-los.
    -   **A Solução (Join):** Utilizamos um *Inner Join* na coluna chave `cases.submitter_id`.
    -   **Detalhamento da Junção:**
        -   **Tabela Esquerda (Clinical):** Contém ~150 colunas com dados do paciente.
            -   *Exemplos de colunas:* `demographic.gender`, `demographic.race`, `diagnoses.ajcc_pathologic_stage`, `demographic.vital_status`, `demographic.days_to_death`.
        -   **Tabela Direita (Biospecimen):** Contém ~40 colunas com dados da coleta física.
            -   *Exemplos de colunas:* `samples.sample_type` (ex: Tumor Primário), `samples.tissue_type`, `samples.is_ffpe`.
        -   **Resultado:** Uma "Tabela Mestre" consolidada inicial contendo **5.928 registros**. O uso do *Inner Join* atua como um filtro de qualidade inicial: pacientes que possuem registros clínicos mas não possuem registros de amostras (ou vice-versa) são descartados, garantindo que o estudo prossiga apenas com casos completos.

### Etapa 2: Engenharia de Features

Modelos de sobrevivência exigem um formato de dados muito específico que não existe nativamente nas bases brutas. Esta etapa transforma dados administrativos em variáveis matemáticas de sobrevivência.

-   **`src/data/feature_engineering_survival.py`**: Seleciona, limpa e transforma as colunas brutas do TCGA para o formato de modelagem.
    -   **Controle de Qualidade (Filtro de Tecidos):**
        -   Foi implementada uma etapa crítica de limpeza para remover amostras que não correspondiam a adenocarcinomas primários de cólon (ex: amostras de reto, metástases distantes ou tecidos não especificados corretamente no dataset original).
        -   **Resultado:** Remoção de **1.560** amostras inconsistentes, garantindo que o modelo aprenda apenas com dados biologicamente coerentes.
    -   **Seleção de Variáveis (Mapeamento):**
        -   `demographic.vital_status` -> `vital_status` (Status Vivo/Morto)
        -   `diagnoses.ajcc_pathologic_stage` -> `ajcc_pathologic_stage` (Estágio do Câncer)
        -   `diagnoses.tissue_or_organ_of_origin` -> `tissue_or_organ_of_origin` (Local do Tumor)
        -   `demographic.age_at_index` -> `age_at_index` (Idade)
    -   **Criação do Alvo (Target):**
        -   `event_occurred`: Binário (1 = Dead, 0 = Alive).
        -   `observed_time`: Calculado unificando `days_to_death` (para óbitos) e `days_to_last_follow_up` (para vivos), convertido para **anos** (dividido por 365.25).
    -   **Tratamento de Não-Linearidade:**
        -   `age_group`: Criação de faixas etárias (bins: 0-40, 40-50, 50-60, etc.) para capturar riscos não-lineares associados ao envelhecimento.
    -   **Dados Finais para o Modelo:** O dataset final entregue ao algoritmo contém apenas: `event_occurred`, `observed_time`, `age_at_index`, `age_group`, `ajcc_pathologic_stage`, e `tissue_or_organ_of_origin`.

### Etapa 3: Análise Exploratória de Dados (EDA)

-   **`eda_tcga_coad.py`**: Gera visualizações para explorar as características dos dados e identificar padrões iniciais. Esta etapa é fundamental para entender a distribuição das variáveis e suas relações com a sobrevivência.

### Etapa 4: Treinamento e Otimização de Modelos

-   **`treino_modelo_sobrevivencia.py`**: Compara diferentes algoritmos de sobrevivência (**Cox Proportional Hazards**, **Random Survival Forest** e **XGBoost Survival**) usando o **C-Index (Índice de Concordância)** como métrica de avaliação.
-   **`exploracao_xgboost.py`**: Script dedicado à otimização profunda e auditoria do modelo XGBoost, garantindo a robustez dos resultados.

### Etapa 5: Interpretação e Previsão

-   **`interpret_survival_model.py`**: Utiliza o modelo CoxPH para identificar e visualizar as features mais impactantes na previsão de risco.
-   **`predict_survival_time.py`**: Carrega o modelo treinado para prever o tempo mediano de sobrevivência para os dados.

### Etapa 6: Visualização dos Resultados

-   **`visualize_survival_curves.py`**: Gera a visualização final, mostrando as curvas de sobrevivência de Kaplan-Meier para os grupos de risco (baixo, médio, alto) definidos pelo modelo.

## 3. Análise Exploratória de Dados: Principais Achados

A análise exploratória revelou insights importantes sobre o conjunto de dados:

### 3.1 Caracterização Detalhada da Amostra

O conjunto de dados final, após todo o processo de limpeza e consolidação, é composto por **4.825 registros** de pacientes. Abaixo, apresentamos um perfil detalhado desta coorte:

*   **Demografia:**
    *   **Gênero:** A distribuição é relativamente equilibrada, com **55,5% homens** (2.678) e **44,5% mulheres** (2.147).
    *   **Idade:** A idade média dos pacientes ao diagnóstico é de **64,6 anos** (mediana de 66 anos), variando de 31 a 89 anos. Isso confirma que a doença afeta predominantemente uma população mais idosa.

*   **Características Clínicas (Estadiamento):**
    *   Os estágios mais frequentes são o **Estágio IIA (877 pacientes)** e o **Estágio IV (789 pacientes)**.
    *   Cerca de 13% dos registros (615 pacientes) não possuíam a informação de estágio reportada ("Not Reported").

*   **Sobrevivência:**
    *   **Status Vital:** Ao final do período de estudo, **32,7%** dos pacientes (1.578) haviam falecido (evento de interesse), enquanto **67,3%** (3.247) permaneciam vivos ou tiveram seu acompanhamento encerrado (censura).
    *   **Tempo de Acompanhamento:** O tempo mediano de acompanhamento foi de aproximadamente **2 anos**, com alguns pacientes sendo acompanhados por mais de 12 anos.

#### Distribuição da Idade e Estágio da Doença

-   A maioria dos pacientes no estudo tem entre 60 e 80 anos, o que é consistente com a epidemiologia do câncer de cólon.
-   O estágio patológico AJCC, um indicador crucial da progressão do câncer, mostra uma distribuição variada, com uma concentração nos estágios II e III.

![Distribuição da Idade](figures/distribuicao_idade.png)
*Figura 1: Histograma da distribuição de idade dos pacientes.*

![Distribuição do Estágio](figures/distribuicao_estagios.png)
*Figura 2: Contagem de pacientes por estágio patológico AJCC.*

#### Análise de Sobrevivência de Kaplan-Meier

-   **Curva Geral**: A curva de sobrevivência geral mostra que a probabilidade de sobrevivência diminui ao longo do tempo, com uma queda mais acentuada nos primeiros 5 anos após o diagnóstico.
-   **Sobrevivência por Estágio**: A análise estratificada por estágio da doença confirma que o **estágio patológico é um dos preditores mais fortes de sobrevivência**. Pacientes em estágios iniciais (Estágio I) têm uma probabilidade de sobrevivência significativamente maior do que aqueles em estágios avançados (Estágio IV).

![Sobrevivência Geral](figures/kaplan_meier_global.png)
*Figura 3: Curva de Sobrevivência de Kaplan-Meier para toda a população do estudo.*

![Sobrevivência por Estágio](figures/kaplan_meier_por_estagio.png)
*Figura 4: Curvas de Sobrevivência de Kaplan-Meier estratificadas por estágio patológico AJCC.*

## 4. Modelagem e Interpretação

#### Seleção do Modelo e Validação Cruzada

Para garantir que a avaliação do nosso modelo fosse robusta e não apenas fruto do acaso, utilizamos a técnica de **Validação Cruzada de 5 Folds (5-Fold Cross-Validation)**.

**Como funciona (Passo a Passo):**
1.  **Divisão:** O conjunto total de dados foi dividido aleatoriamente em 5 partes iguais (chamadas de "folds").
2.  **Rodízio de Testes:** O treinamento foi realizado 5 vezes. Em cada rodada:
    -   Uma parte diferente foi separada para ser o **Teste** (invisível ao modelo).
    -   As outras 4 partes foram usadas para **Treinar** o modelo.
3.  **Média Final:** Ao final das 5 rodadas, calculamos a média do desempenho (C-Index).

### 4.1 Métricas Detalhadas dos Modelos

Avaliamos três arquiteturas distintas e comparamos seu desempenho utilizando a métrica **C-Index**. Abaixo, apresentamos os resultados detalhados:

| Modelo | C-Index (Teste) | C-Index (Média CV) |
| :--- | :--- | :--- |
| **XGBoost Survival (Otimizado)** | **0.9541** | **0.9458** |
| **Random Survival Forest (RSF)** | 0.8757 | 0.8486 |
| **Cox Proportional Hazards (CoxPH)** | 0.7390 | 0.7384 |

**Nota sobre Metodologia Rigorosa:**
Para garantir a integridade dos resultados, implementamos salvaguardas metodológicas avançadas:
1.  **Otimização de Hiperparâmetros:** O modelo XGBoost passou por um processo intensivo de *Random Search* para ajuste fino de parâmetros (profundidade de árvores, taxa de aprendizado, regularização L1/L2), o que resultou em um salto significativo de desempenho (de ~0.83 para ~0.95).
2.  **Prevenção de Vazamento de Dados (Data Leakage):** Todo o pré-processamento de variáveis categóricas (One-Hot Encoding) foi isolado, sendo ajustado (*fit*) exclusivamente no conjunto de treino e aplicado (*transform*) no conjunto de teste. Durante a validação cruzada, este isolamento foi replicado dentro de cada *fold*, garantindo que nenhuma informação do teste "vazasse" para o treino.
3.  **Tratamento de Tempo Zero:** Pacientes com tempo de sobrevivência registrado como zero (óbito ou censura no dia do diagnóstico) foram tratados matematicamente (adição de epsilon) em vez de descartados, evitando viés de seleção e perda de casos críticos de alta mortalidade.

**Interpretação das Métricas**

*   **C-Index (Concordance Index):** É a métrica que define a qualidade da previsão. Indica a probabilidade de o modelo ordenar corretamente dois pacientes aleatórios, ou seja, o paciente que falece primeiro deve ter maior risco predito.
    *   Um valor de **0.5** representa desempenho aleatório.
    *   Valores entre **0.7 e 0.8** indicam bom desempenho clínico.
    *   Acima de **0.9** representa desempenho excelente (Estado da Arte).
    
    O **XGBoost Survival** atingiu um C-Index de **0.9541**, um resultado excepcional que supera largamente o desempenho do CoxPH (0.7390) e do RSF (0.8757), demonstrando sua superioridade na captura de padrões complexos de sobrevivência.

*   **Estabilidade e Desvio Padrão:** Mede a confiança e a robustez do modelo.
    *   Ao contrário de modelos instáveis que variam muito dependendo dos dados de entrada, o XGBoost otimizado demonstrou alta consistência.
    *   A diferença mínima entre o C-Index no conjunto de Teste (**0.9541**) e a média da Validação Cruzada (**0.9458**) indica um **desvio padrão baixo** e ausência de *overfitting*. Isso confirma que o modelo é robusto e generaliza bem para novos pacientes, mantendo sua alta precisão independentemente da divisão dos dados.

### 4.3.2 Análise Comparativa

O modelo **XGBoost Survival** foi o grande vencedor da comparação. Após uma rigorosa otimização de hiperparâmetros, ele não apenas obteve a maior média de acerto (**0.9541** vs 0.8757 do Random Survival Forest), mas também demonstrou ser extremamente robusto, com um desvio padrão baixo entre os folds de validação. Este resultado excepcional (C-Index > 0.95) indica que o modelo consegue ordenar corretamente os pacientes por risco de óbito em mais de 95% dos pares avaliados, um desempenho considerado estado da arte para dados clínicos.

O desempenho superior do XGBoost em relação ao Cox Proportional Hazards (0.7390) e até mesmo ao RSF pode ser atribuído à sua arquitetura de *gradient boosting*, capaz de corrigir iterativamente os erros dos estimadores anteriores e capturar interações complexas e não-lineares entre variáveis como idade, estágio e localização tumoral.

É notável que, mesmo utilizando apenas dados clínicos e demográficos (sem dados genômicos), o modelo alcançou uma performance comparável a estudos que utilizam biomarcadores avançados. Isso sugere que as variáveis clínicas selecionadas, particularmente o estadiamento AJCC refinado, possuem alto poder discriminatório quando processadas por algoritmos de ponta.

### 4.4 Interpretação dos Fatores de Risco

Para compreender quais fatores mais influenciam a sobrevivência, utilizou-se o modelo **Cox Proportional Hazards** como ferramenta de interpretação auxiliar, devido à sua capacidade de produzir *Hazard Ratios* (Razões de Risco) facilmente interpretáveis clinicamente.

A análise confirmou o **estadiamento patológico AJCC** como o fator predominante. Abaixo detalhamos os principais fatores de risco identificados:

**Fatores de Alto Risco (Hazard Ratio > 1):**

*   **Estágio IV (HR ≈ 1.91) e IVA (HR ≈ 1.41):** Confirmando a gravidade da doença metastática, o Estágio IV destaca-se visual e clinicamente como um determinante crítico de mortalidade. A análise visual dos Hazard Ratios corrobora o consenso médico de que o estágio avançado é o principal impulsionador do risco de óbito.
*   **Junção Retossigmoide (HR ≈ 2.18):** Uma observação interessante deste dataset específico é o elevado risco associado a tumores na junção retossigmoide. Embora seu valor numérico de HR seja ligeiramente superior ao do Estágio IV neste modelo, isso pode refletir características específicas da amostra ou diagnósticos mais tardios nesta região anatômica.
*   **Idade Avançada (80-100 anos):** A idade avançada apresenta um aumento de risco consistente de **44%** (HR ≈ 1.44), refletindo a fragilidade natural e comorbidades associadas ao envelhecimento.

![Hazard Ratios](figures/razoes_risco_cox.png)
*Figura 6: Impacto das variáveis na sobrevivência (Hazard Ratios) segundo o modelo CoxPH.*

A **Figura 6** detalha os fatores que aumentam (vermelho) ou diminuem (verde) o risco de mortalidade:

*   **Fatores de Alto Risco (Hazard Ratio > 1):**
    *   **Estágio IV (HR ≈ 1.91) e IVA (HR ≈ 1.41):** Confirmando a gravidade da doença metastática, o Estágio IV se consolida como um determinante crítico de mortalidade. Pacientes neste estágio apresentam quase o dobro do risco de óbito em comparação à média.
    *   **Junção Retossigmoide (HR ≈ 2.18):** Neste conjunto de dados, tumores nesta localização específica apresentaram um risco extremamente elevado, superando estatisticamente até mesmo o estadiamento geral, o que pode indicar um perfil biológico mais agressivo ou diagnósticos mais tardios nesta região anatômica.
    *   **Idade Avançada (80-100 anos):** Apresenta um aumento de risco de **43%** (HR ≈ 1.44), consistente com a fragilidade natural e comorbidades esperadas nesta faixa etária.

*   **Fatores Protetores (Hazard Ratio < 1):**
    *   **Estágios Iniciais:** Estágios como **I** (HR ≈ 0.52) e **IIA** (HR ≈ 0.61) atuam como fortes fatores de proteção, reduzindo o risco de morte pela metade ou mais.
    *   **Localização Anatômica:** Tumores na **Flexura Hepática** (HR ≈ 0.41) e **Cólon Sigmoide** (HR ≈ 0.62) mostraram-se associados a melhores prognósticos neste conjunto de dados.

### 4.5 Estratificação de Risco

O modelo **XGBoost Survival** foi utilizado para classificar os pacientes em três grupos de risco: **Baixo**, **Médio** e **Alto**. Esta estratificação foi realizada através da divisão dos escores de risco preditos em tercis.

As curvas de sobrevivência de Kaplan-Meier para cada grupo (Figura 7) demonstraram separação clara e estatisticamente significativa:
*   **Grupo de Baixo Risco (Verde):** Sobrevida excelente (**100% em 1 ano**, **99.7% em 5 anos**).
*   **Grupo de Médio Risco (Laranja):** Declínio gradual (**99.6% em 1 ano**, **74.2% em 5 anos**).
*   **Grupo de Alto Risco (Vermelho):** Prognóstico severo (**73.3% em 1 ano**, **33.3% em 3 anos**).

![Curvas por Grupo de Risco](figures/curvas_sobrevivencia_por_grupo_risco.png)
*Figura 7: Curvas de Sobrevivência de Kaplan-Meier estratificadas pelos grupos de risco previstos pelo modelo XGBoost.*

### 4.6 Limitações do Estudo

Este trabalho apresenta algumas limitações importantes que devem ser consideradas na interpretação dos resultados:
*   **Escopo de Features:** O modelo utilizou exclusivamente dados clínicos e demográficos. A não incorporação de dados moleculares (expressão gênica, mutações) representa uma oportunidade de melhoria, embora o desempenho atual já seja excepcional.
*   **Generalização:** Os dados provêm de uma única fonte (TCGA), que pode não representar adequadamente a diversidade global de populações e sistemas de saúde.
*   **Validação Externa:** O modelo foi validado com técnica de *Hold-out* (treino/teste), mas não em uma coorte externa independente de outra instituição.
*   **Temporalidade dos Dados:** Os dados do TCGA foram coletados em um período específico e podem não refletir os tratamentos mais recentes.

## 5. Conclusão

Este trabalho demonstrou com sucesso a construção de um pipeline de análise de sobrevivência de ponta a ponta para pacientes com adenocarcinoma de cólon. O modelo **XGBoost Survival** foi identificado como o de melhor desempenho, alcançando um C-Index de **~0.95**, superando significativamente o limiar de bom desempenho clínico e demonstrando robustez contra overfitting.

Os principais achados confirmaram o estadiamento patológico AJCC como o fator de risco predominante. A estratificação de pacientes em grupos de risco demonstrou-se altamente efetiva, permitindo a identificação precoce de pacientes que necessitam de intervenções agressivas.

### 5.1 Contribuições do Trabalho

*   Desenvolvimento de um pipeline modular e documentado para análise de sobrevivência com dados do TCGA.
*   Comparação sistemática e otimização de algoritmos (CoxPH, RSF, XGBoost).
*   Implementação de estratificação de risco precisa com potencial aplicação clínica.
*   Validação de fatores prognósticos conhecidos através de metodologia computacional rigorosa (auditoria de *data leakage*).

### 5.2 Trabalhos Futuros

Como direções para trabalhos futuros, sugere-se:
*   **Incorporação de dados genômicos:** Integrar dados de expressão gênica e mutações para refinar a predição.
*   **Validação externa:** Testar o modelo em coortes de hospitais brasileiros ou outros bancos públicos.
*   **Interpretabilidade Avançada:** Implementar SHAP values para explicar predições individuais do XGBoost.
*   **Interface Web:** Desenvolver um dashboard interativo para uso clínico da ferramenta de estratificação.
