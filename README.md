# Análise de Sobrevivência para Câncer de Cólon (TCGA-COAD)

Este projeto aplica técnicas avançadas de Data Science e Machine Learning para prever a sobrevivência de pacientes com adenocarcinoma de cólon, utilizando dados clínicos e demográficos do **The Cancer Genome Atlas (TCGA)**.

## 🎯 Objetivo

Desenvolver um modelo capaz de estratificar pacientes em grupos de risco (baixo, médio, alto) e prever o tempo de sobrevivência, auxiliando na compreensão dos fatores prognósticos da doença.

## 📊 Principais Resultados

O modelo final (**Random Survival Forest**) alcançou um **C-Index de 0.85** (nos testes de validação), demonstrando alta capacidade de discriminação entre pacientes de diferentes riscos.


## 🛠️ Pipeline do Projeto

O projeto segue um pipeline modular e reprodutível:

1.  **Coleta e Consolidação**: Unificação de dados clínicos e de bioespécimes do TCGA.
2.  **Engenharia de Features e Limpeza**: 
    *   Filtragem rigorosa de amostras (mantendo apenas sítios anatômicos confirmados de cólon, removendo ~1.500 amostras inconsistentes ou de outros tecidos).
    *   Criação de variáveis de sobrevivência (tempo observado, censura, faixas etárias).
3.  **Análise Exploratória (EDA)**: Estudos detalhados sobre distribuição de idade, estágio e curvas de Kaplan-Meier.
4.  **Modelagem**: Treinamento e comparação de modelos:
    *   *Cox Proportional Hazards* (Foco em explicabilidade)
    *   *Random Survival Forest* (Foco em performance - **Modelo Vencedor**)
    *   *XGBoost Survival*
5.  **Interpretação**: Análise de Hazard Ratios e importância das variáveis.

## 🚀 Como Executar

### Pré-requisitos
*   Python 3.10+
*   Pip

### Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/PhelQ/tcc-data-science.git
    cd tcc-data-science
    ```

2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3.  Execute o pipeline completo:
    ```bash
    python src/main.py
    ```

## 📂 Estrutura do Repositório

```
├── data/               # Dados (ignorados no git por tamanho/privacidade)
├── models/             # Modelos treinados (ignorados)
├── reports/            # Relatórios e figuras geradas
│   ├── figures/        # Gráficos (PNG)
│   └── relatorio_do_projeto.md  # Relatório detalhado completo
├── src/                # Código fonte
│   ├── data/           # Scripts de processamento de dados
│   ├── eda/            # Scripts de análise exploratória
│   ├── modeling/       # Treinamento e avaliação de modelos
│   ├── visualization/  # Geração de gráficos finais
│   └── main.py         # Orquestrador do projeto
└── requirements.txt    # Dependências do projeto
```

## 📝 Relatório Completo

Para uma leitura aprofundada sobre a metodologia, análise estatística e discussão dos resultados, consulte o [Relatório do Projeto](reports/relatorio_do_projeto.md).

---
*Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) em Data Science.*

Este projeto aplica técnicas avançadas de Data Science e Machine Learning para prever a sobrevivência de pacientes com adenocarcinoma de cólon, utilizando dados clínicos e demográficos do **The Cancer Genome Atlas (TCGA)**.

## 🎯 Objetivo

Desenvolver um modelo capaz de estratificar pacientes em grupos de risco (baixo, médio, alto) e prever o tempo de sobrevivência, auxiliando na compreensão dos fatores prognósticos da doença.

## 📊 Principais Resultados

O modelo final (**XGBoost Survival Otimizado**) alcançou um **C-Index de 0.95** (nos testes de validação), superando significativamente o Random Survival Forest (0.87) e demonstrando uma capacidade excepcional de discriminação de risco.


## 🛠️ Pipeline do Projeto

O projeto segue um pipeline modular e reprodutível:

1.  **Coleta e Consolidação**: Unificação de dados clínicos e de bioespécimes do TCGA.
2.  **Engenharia de Features e Limpeza**: 
    *   Filtragem rigorosa de amostras (mantendo apenas sítios anatômicos confirmados de cólon, removendo ~1.500 amostras inconsistentes ou de outros tecidos).
    *   Criação de variáveis de sobrevivência (tempo observado, censura, faixas etárias).
3.  **Análise Exploratória (EDA)**: Estudos detalhados sobre distribuição de idade, estágio e curvas de Kaplan-Meier.
4.  **Modelagem**: Treinamento e comparação de modelos:
    *   *Cox Proportional Hazards* (Foco em explicabilidade)
    *   *Random Survival Forest* (Benchmark robusto)
    *   *XGBoost Survival* (Foco em performance máxima - **Modelo Vencedor**)
5.  **Interpretação**: Análise de Hazard Ratios, SHAP Values e importância das variáveis.

## 🚀 Como Executar

### Pré-requisitos
*   Python 3.10+
*   Pip

### Instalação

1.  Clone o repositório:
    ```bash
    git clone https://github.com/PhelQ/tcc-data-science.git
    cd tcc-data-science
    ```

2.  Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

3.  Execute o pipeline completo:
    ```bash
    python src/main.py
    ```

## 📂 Estrutura do Repositório

```
├── data/               # Dados (ignorados no git por tamanho/privacidade)
├── models/             # Modelos treinados (ignorados)
├── reports/            # Relatórios e figuras geradas
│   ├── figures/        # Gráficos (PNG)
│   └── relatorio_do_projeto.md  # Relatório detalhado completo
├── src/                # Código fonte
│   ├── data/           # Scripts de processamento de dados
│   ├── eda/            # Scripts de análise exploratória
│   ├── modeling/       # Treinamento e avaliação de modelos
│   ├── visualization/  # Geração de gráficos finais
│   └── main.py         # Orquestrador do projeto
└── requirements.txt    # Dependências do projeto
```

## 📝 Relatório Completo

Para uma leitura aprofundada sobre a metodologia, análise estatística e discussão dos resultados, consulte o [Relatório do Projeto](reports/relatorio_do_projeto.md).

---
*Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) em Data Science.*
