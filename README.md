# Análise de Sobrevivência para Câncer de Cólon (TCGA-COAD)

Este projeto aplica técnicas avançadas de Data Science e Machine Learning para prever a sobrevivência de pacientes com adenocarcinoma de cólon, utilizando dados clínicos e demográficos do **The Cancer Genome Atlas (TCGA)**.

## 🎯 Objetivo

Desenvolver modelos capazes de estratificar pacientes em grupos de risco (baixo, médio, alto) e prever o tempo de sobrevivência, auxiliando na compreensão dos fatores prognósticos da doença.

## 📊 Principais Resultados

### Comparação de Modelos (C-Index no Teste)

| Modelo | C-Index (CV) | C-Index (Teste) | Característica |
|--------|--------------|-----------------|----------------|
| **XGBoost Survival** | **0.952** | **0.979** | **Melhor Performance** |
| Random Survival Forest | 0.844 | 0.889 | Robustez / Baixa Variância |
| Cox Proportional Hazards | 0.706 | 0.736 | Interpretabilidade |

O **XGBoost Survival** alcançou o melhor desempenho com **C-Index de 0.979** no conjunto de teste, demonstrando alta capacidade de discriminação entre pacientes de diferentes riscos.

## 🛠️ Pipeline do Projeto

O projeto segue um pipeline modular e reprodutível:

1. **Pré-processamento**: Conversão dos dados brutos TSV do TCGA para formato Parquet
   - Dados clínicos e de bioespécimes processados separadamente

2. **Consolidação**: Unificação de dados clínicos e de bioespécimes
   - Cruzamento de bases por `cases.submitter_id` (ID único do paciente)
   - Filtragem rigorosa de amostras (~1.500 amostras removidas)

3. **Engenharia de Features e Limpeza**:
   - Filtragem de sítios anatômicos confirmados de cólon
   - Criação de variáveis de sobrevivência (tempo observado, censura)
   - One-hot encoding com prevenção de data leakage

4. **Análise Exploratória (EDA)**:
   - Distribuição de idade e estágio patológico
   - Taxa de óbito por faixa etária
   - Curvas de Kaplan-Meier globais e estratificadas por estágio

5. **Modelagem**:
   - **Cox Proportional Hazards**: Baseline interpretável
   - **Random Survival Forest**: Ensemble robusto (200 árvores, profundidade 10)
   - **XGBoost Survival**: Gradient boosting otimizado (200 estimadores, profundidade 10)
   - Validação cruzada 5-fold + holdout test (80/20)

6. **Interpretação**:
   - **Cox**: Hazard Ratios (razões de risco)
   - **XGBoost**: Feature importance (weight/gain) + Partial Dependence Plots
   - **RSF**: Permutation importance + Partial Dependence Plots

7. **Predição**: Estimativa do tempo mediano de sobrevivência por paciente

8. **Visualização**: Curvas de sobrevivência estratificadas por grupo de risco (baixo/médio/alto)

9. **Validação de Overfitting**: Diagnóstico completo com 5 testes (gap treino/teste, curvas de aprendizado, teste de permutação)

## 🚀 Como Executar

### Pré-requisitos
- Python 3.10+
- Pip

### Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/PhelQ/tcc-data-science.git
   cd tcc-data-science
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

3. Execute o pipeline completo (a partir da etapa de engenharia de features):
   ```bash
   python src/main.py
   ```

4. Ou execute módulos específicos:
   ```bash
   # Pré-processamento dos dados brutos
   python -m src.data.preprocessamento_data

   # Consolidação dos dados
   python -m src.data.consolidacaodados_tcga_coad

   # Apenas treinamento
   python -m src.modeling.train

   # Interpretação do Cox
   python -m src.modeling.interpret_cox

   # Interpretação do RSF
   python -m src.modeling.interpret_rsf

   # Interpretação do XGBoost
   python -m src.modeling.interpret_xgboost

   # Validação de overfitting
   python -m src.modeling.validate_overfitting
   ```

## 📂 Estrutura do Repositório

```
├── data/                    # Dados (ignorados no git)
│   ├── raw/                 # Dados brutos do TCGA (.tsv)
│   ├── interim/             # Dados intermediários (.parquet)
│   └── processed/           # Dados processados (.parquet)
├── models/                  # Modelos treinados (ignorados no git)
│   ├── coxph_model.joblib
│   ├── rsf_model.joblib
│   ├── xgb_model.joblib
│   ├── survival_model.joblib
│   └── training_columns.joblib
├── reports/                 # Relatórios e visualizações
│   ├── figures/             # Gráficos gerados (PNG)
│   │   ├── distribuicao_idade.png
│   │   ├── distribuicao_estagios.png
│   │   ├── distribuicao_localizacao_anatomica.png
│   │   ├── distribuicao_status_vital.png
│   │   ├── distribuicao_tempo_acompanhamento.png
│   │   ├── taxa_obito_por_faixa_etaria.png
│   │   ├── kaplan_meier_global.png
│   │   ├── kaplan_meier_por_estagio.png
│   │   ├── razoes_risco_cox.png
│   │   ├── importancia_variaveis_peso_xgboost.png
│   │   ├── importancia_variaveis_ganho_xgboost.png
│   │   ├── dependencia_parcial_xgboost.png
│   │   ├── importancia_permutacao_rsf.png
│   │   ├── dependencia_parcial_rsf.png
│   │   ├── curvas_sobrevivencia_por_grupos_risco.png
│   │   └── diagnostico_overfitting.png
│   └── model_comparison_results.csv
├── src/                     # Código fonte
│   ├── data/                # Processamento de dados
│   │   ├── preprocessamento_data.py
│   │   ├── consolidacaodados_tcga_coad.py
│   │   └── feature_engineering_survival.py
│   ├── aed/                 # Análise Exploratória de Dados
│   │   └── aed_tcga_coad.py
│   ├── modeling/            # Modelagem e interpretação
│   │   ├── train.py               # Treina os 3 modelos
│   │   ├── interpret_cox.py       # Interpretação Cox (Hazard Ratios)
│   │   ├── interpret_xgboost.py   # Interpretação XGBoost
│   │   ├── interpret_rsf.py       # Interpretação RSF
│   │   ├── predict.py             # Predições de sobrevivência
│   │   └── validate_overfitting.py # Diagnóstico de overfitting
│   ├── visualization/       # Visualizações
│   │   └── visualize_survival_curves.py
│   ├── config.py            # Configurações globais
│   ├── utils.py             # Funções utilitárias
│   └── main.py              # Orquestrador do pipeline
├── main.py                  # Entry point raiz
├── notebooks/               # Notebooks exploratórios
│   └── visualizacao.ipynb
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

## 📊 Principais Variáveis Preditivas

Com base na análise de importância dos modelos:

1. **Estágio Patológico** (Stage IV) - Maior impacto no risco
2. **Idade do Paciente** - Fator de risco contínuo
3. **Sítio Anatômico** - Localização específica do tumor (Cólon Sigmoide, Ceco, etc.)

## 🔬 Metodologia Técnica

- **Prevenção de Data Leakage**: One-hot encoding aplicado separadamente em cada fold
- **Validação Robusta**: Cross-validation 5-fold + holdout test
- **Métrica**: C-Index (Concordance Index) - probabilidade de ordenação correta de risco
- **Interpretabilidade**: Hazard Ratios, Permutation Importance, Partial Dependence Plots

## 📈 Próximos Passos

- [ ] Adicionar dados genômicos (mutações, expressão gênica)
- [ ] Implementar análise de sensibilidade
- [ ] Desenvolver interface web para predições
- [ ] Publicar modelo como API

## 📚 Referências

- The Cancer Genome Atlas (TCGA): [https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)
- scikit-survival: [https://scikit-survival.readthedocs.io/](https://scikit-survival.readthedocs.io/)
- lifelines: [https://lifelines.readthedocs.io/](https://lifelines.readthedocs.io/)

---

**Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) em Data Science**

**Autor**: Phelipe Quintas
**Repositório**: [github.com/PhelQ/tcc-data-science](https://github.com/PhelQ/tcc-data-science)
