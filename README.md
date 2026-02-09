# Análise de Sobrevivência para Câncer de Cólon (TCGA-COAD)

Este projeto aplica técnicas avançadas de Data Science e Machine Learning para prever a sobrevivência de pacientes com adenocarcinoma de cólon, utilizando dados clínicos e demográficos do **The Cancer Genome Atlas (TCGA)**.

## 🎯 Objetivo

Desenvolver modelos capazes de estratificar pacientes em grupos de risco (baixo, médio, alto) e prever o tempo de sobrevivência, auxiliando na compreensão dos fatores prognósticos da doença.

## 📊 Principais Resultados

### Comparação de Modelos (C-Index no Teste)

| Modelo | C-Index (CV) | C-Index (Teste) | Característica |
|--------|--------------|-----------------|----------------|
| Cox Proportional Hazards | 0.710 | 0.735 | Interpretabilidade |
| **Random Survival Forest** | **0.845** | **0.889** | **Melhor Performance** |
| XGBoost Survival | 0.804 | 0.849 | Eficiência |

O **Random Survival Forest** alcançou o melhor desempenho com **C-Index de 0.889** no conjunto de teste, demonstrando alta capacidade de discriminação entre pacientes de diferentes riscos.

## 🛠️ Pipeline do Projeto

O projeto segue um pipeline modular e reprodutível:

1. **Coleta e Consolidação**: Unificação de dados clínicos e de bioespécimes do TCGA
   - Cruzamento de bases por `cases.submitter_id` (ID único do paciente)
   - Filtragem rigorosa de amostras (~1.500 amostras removidas)

2. **Engenharia de Features e Limpeza**:
   - Filtragem de sítios anatômicos confirmados de cólon
   - Criação de variáveis de sobrevivência (tempo observado, censura, faixas etárias)
   - One-hot encoding com prevenção de data leakage

3. **Análise Exploratória (EDA)**:
   - Distribuição de idade, gênero, raça e estágio patológico
   - Curvas de Kaplan-Meier estratificadas
   - Análise de tempo de sobrevivência

4. **Modelagem**:
   - **Cox Proportional Hazards**: Baseline interpretável
   - **Random Survival Forest**: Ensemble robusto (200 árvores, profundidade 10)
   - **XGBoost Survival**: Gradient boosting otimizado
   - Validação cruzada 5-fold + holdout test (80/20)

5. **Interpretação**:
   - **Cox**: Hazard Ratios (razões de risco)
   - **XGBoost**: Feature importance (weight/gain) + Partial Dependence Plots
   - **RSF**: Permutation importance + Partial Dependence Plots

6. **Visualização**:
   - Curvas de sobrevivência por grupo de risco
   - Gráficos de explicabilidade para todos os modelos

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

3. Execute o pipeline completo:
   ```bash
   python src/main.py
   ```

4. Ou execute módulos específicos:
   ```bash
   # Apenas treinamento
   python -m src.modeling.train

   # Interpretação do RSF
   python -m src.modeling.interpret_rsf

   # Interpretação do XGBoost
   python -m src.modeling.interpret_xgboost
   ```

## 📂 Estrutura do Repositório

```
├── data/                    # Dados (ignorados no git)
│   ├── raw/                 # Dados brutos do TCGA
│   └── processed/           # Dados processados
├── models/                  # Modelos treinados (ignorados)
│   ├── coxph_model.pkl
│   ├── rsf_model.pkl
│   ├── xgboost_model.pkl
│   └── best_survival_model.pkl
├── reports/                 # Relatórios e visualizações
│   ├── figures/             # Gráficos gerados (PNG)
│   │   ├── razoes_risco_cox.png
│   │   ├── xgboost_native_importance_pt.png
│   │   ├── dependencia_parcial_xgboost.png
│   │   ├── rsf_permutation_importance_pt.png
│   │   └── dependencia_parcial_rsf.png
│   └── model_comparison_results.csv
├── src/                     # Código fonte
│   ├── data/                # Processamento de dados
│   │   ├── consolidacaodados_tcga_coad.py
│   │   ├── preprocessamento_data.py
│   │   └── feature_engineering_survival.py
│   ├── aed/                 # Análise Exploratória de Dados
│   │   └── aed_tcga_coad.py
│   ├── modeling/            # Modelagem e interpretação
│   │   ├── train.py         # Treina os 3 modelos
│   │   ├── interpret_cox.py # Interpretação Cox (Hazard Ratios)
│   │   ├── interpret_xgboost.py # Interpretação XGBoost
│   │   ├── interpret_rsf.py # Interpretação RSF
│   │   └── predict.py       # Predições de sobrevivência
│   ├── visualization/       # Visualizações
│   │   └── visualize_survival_curves.py
│   ├── config.py            # Configurações globais
│   ├── utils.py             # Funções utilitárias
│   └── main.py              # Orquestrador do pipeline
├── notebooks/               # Notebooks exploratórios
│   └── visualizacao.ipynb
├── requirements.txt         # Dependências do projeto
└── README.md                # Este arquivo
```

## 📊 Principais Variáveis Preditivas

Com base na análise de importância dos modelos:

1. **Estágio Patológico** (Stage IV) - Maior impacto no risco
2. **Idade do Paciente** - Fator de risco contínuo
3. **Sítio Anatômico** - Localização específica do tumor (Sigmoid colon, Cecum, etc.)
4. **Grupos Etários** - Discretização da idade

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

## 📝 Relatório Completo

Para uma leitura aprofundada sobre a metodologia, análise estatística e discussão dos resultados, consulte o [Relatório do Projeto](reports/relatorio_do_projeto.md).

## 📚 Referências

- The Cancer Genome Atlas (TCGA): [https://www.cancer.gov/tcga](https://www.cancer.gov/tcga)
- scikit-survival: [https://scikit-survival.readthedocs.io/](https://scikit-survival.readthedocs.io/)
- lifelines: [https://lifelines.readthedocs.io/](https://lifelines.readthedocs.io/)

---

**Desenvolvido como parte do Trabalho de Conclusão de Curso (TCC) em Data Science**

**Autor**: Phelipe Quintas
**Repositório**: [github.com/PhelQ/tcc-data-science](https://github.com/PhelQ/tcc-data-science)
