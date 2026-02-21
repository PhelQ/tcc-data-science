"""
Interpretação do Modelo XGBoost Survival.

Gera análises de explicabilidade:
- Importância de variáveis (weight e gain)
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor, plot_importance
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import train_test_split
from sksurv.metrics import concordance_index_censored

from src import config
from src.utils import save_plot
from src.modeling.train import load_and_split_data, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def explore_xgboost():
    logging.info("--- Iniciando Sessão de Exploração do XGBoost ---")
    
    # 1. Carregar Dados
    logging.info(f"Carregando dados de {config.FEATURES_SURVIVAL_PATH}...")
    X, y, cat_features = load_and_split_data(config.FEATURES_SURVIVAL_PATH)
    
    # 2. Split Treino/Teste
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y["event"]
    )
    
    # 3. Encoding (Fit Treino, Transform Teste)
    X_train, X_test = encode_features(X_train_raw, X_test_raw, cat_features)
    
    # Preparar y para XGBoost
    y_train_xgb = np.where(y_train["event"], y_train["time"], -y_train["time"])
    
    # 4. Hiperparâmetros 
    best_params = {
        'max_depth': 10,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'min_child_weight': 1,
        'gamma': 0.2,
        'subsample': 0.8,
        'colsample_bytree': 1.0,
        'reg_alpha': 0,
        'reg_lambda': 0.1,
        'objective': 'survival:cox',
        'eval_metric': 'cox-nloglik',
        'n_jobs': -1,
        'random_state': 42
    }
    
    # 5. Treinar Melhor Modelo no Treino Completo
    logging.info("Treinando melhor modelo no conjunto de treino completo...")
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train_xgb)
    
    # 6. Avaliar no Teste
    test_preds = final_model.predict(X_test)
    test_score = concordance_index_censored(y_test["event"], y_test["time"], test_preds)[0]
    logging.info(f"Score Final no Teste (Hold-out): {test_score:.4f}")
    
    # 7. Análise de Importância (Nativa XGBoost)
    logging.info("Gerando gráficos de importância nativos...")
    
    # Plot de Peso (Weight)
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_importance(final_model, max_num_features=20, importance_type='weight',
                    height=0.5, ax=ax, color=config.PALETTE['primary'])
    ax.set_title(f"Importância das Variáveis (Peso) - C-Index: {test_score:.3f}")
    ax.set_xlabel("F Score (Peso)")
    ax.set_ylabel("Variáveis")
    plt.tight_layout()
    save_plot(fig, config.FIGURES_DIR, "importancia_variaveis_peso_xgboost.png")
    
    # Plot de Ganho (Gain)
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_importance(final_model, max_num_features=20, importance_type='gain',
                    height=0.5, ax=ax, color=config.PALETTE['primary'])
    ax.set_title(f"Importância das Variáveis (Ganho de Informação) - C-Index: {test_score:.3f}")
    ax.set_xlabel("Ganho Médio")
    ax.set_ylabel("Variáveis")
    plt.tight_layout()
    save_plot(fig, config.FIGURES_DIR, "importancia_variaveis_ganho_xgboost.png")

    # 8. Gráficos de Dependência Parcial (PDP)
    logging.info("Gerando Gráficos de Dependência Parcial (PDP)...")
    try:
        # Selecionar as top 3 variáveis baseadas no ganho (gain)
        importance_scores = final_model.get_booster().get_score(importance_type='gain')
        top_3_features = sorted(importance_scores, key=importance_scores.get, reverse=True)[:3]
        
        logging.info(f"Variáveis selecionadas para PDP: {top_3_features}")
        
        # Criar uma função wrapper para prever em escala logarítmica (log-hazard)
        # Isso evita que a escala exploda para 600.000 e permite ver a tendência real.
        import xgboost as xgb
        class XGBoostLogWrapper:
            def __init__(self, model):
                self.model = model
                self._estimator_type = "regressor"
                self.classes_ = []
            def fit(self, X, y):
                return self
            def predict(self, X):
                # Usando DMatrix para garantir compatibilidade com o booster
                dmat = xgb.DMatrix(X)
                return self.model.get_booster().predict(dmat, output_margin=True)

        wrapped_model = XGBoostLogWrapper(final_model)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6)) # Criando 3 colunas explicitamente
        display = PartialDependenceDisplay.from_estimator(
            wrapped_model,
            X_train,
            features=top_3_features,
            kind="both",
            subsample=100,
            ax=ax,
            percentiles=(0.05, 0.95),
            pd_line_kw={'color': config.PALETTE['primary'], 'linewidth': 2.5},
            ice_lines_kw={'color': config.PALETTE['primary'], 'alpha': 0.08, 'linewidth': 0.5},
        )
        
        plt.suptitle("Impacto das Variáveis no Log-Risco (PDP/ICE)", fontsize=16, fontweight='bold')
        for a in ax.flatten():
            a.set_ylabel("Log-Hazard (Risco Relativo)")
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_plot(fig, config.FIGURES_DIR, "dependencia_parcial_xgboost.png")
        logging.info("Gráficos de PDP corrigidos e gerados com sucesso!")
        
    except Exception as e:
        logging.error(f"Erro ao gerar gráficos de PDP: {e}")

    logging.info("Exploração concluída! Confira os gráficos em reports/figures/")
    
def main():
    explore_xgboost()

if __name__ == "__main__":
    main()
