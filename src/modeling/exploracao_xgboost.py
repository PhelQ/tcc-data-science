import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer
from sksurv.metrics import concordance_index_censored

from src import config
from src.utils import save_plot
from src.modeling.treino_modelo_sobrevivencia import load_and_split_data, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def c_index_scorer(estimator, X, y):
    """Scorer customizado para RandomizedSearchCV (C-Index)."""
    y_pred = estimator.predict(X)
    try:
        event = y["event"]
        time = y["time"]
    except:
        event = y["event"]
        time = y["time"]
        
    return concordance_index_censored(event, time, y_pred)[0]

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
    
    # 4. Melhores Parâmetros
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
    
    # 7. Análise de Importância (Nativa XGBoost) - TRADUZIDO
    logging.info("Gerando gráficos de importância nativos (Traduzidos)...")
    
    # Plot Weight
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_importance(final_model, max_num_features=20, importance_type='weight', height=0.5, ax=ax)
    ax.set_title(f"Importância das Variáveis (Peso) - C-Index: {test_score:.3f}")
    ax.set_xlabel("F Score (Peso)")
    ax.set_ylabel("Variáveis")
    plt.tight_layout()
    save_plot(fig, config.FIGURES_DIR, "xgboost_native_importance_pt.png")
    
    # Plot Gain
    fig, ax = plt.subplots(figsize=(12, 10))
    plot_importance(final_model, max_num_features=20, importance_type='gain', height=0.5, ax=ax)
    ax.set_title(f"Importância das Variáveis (Ganho de Informação) - C-Index: {test_score:.3f}")
    ax.set_xlabel("Ganho Médio")
    ax.set_ylabel("Variáveis")
    plt.tight_layout()
    save_plot(fig, config.FIGURES_DIR, "xgboost_native_importance_gain_pt.png")

    # 8. SHAP (Tentativa com Fix de Compatibilidade)
    logging.info("Tentando gerar SHAP values (com fix de compatibilidade)...")
    try:
        # Hack para compatibilidade XGBoost > 1.0 / SHAP
        # O problema é que o TreeExplainer as vezes falha ao ler o base_score do modelo salvo
        # Passar o booster diretamente ajuda
        booster = final_model.get_booster()
        
        # Workaround específico para 'utf-8' error em algumas versões
        model_bytearray = booster.save_raw()[4:]
        def myfun(self=None, **kwargs):
            return model_bytearray
        booster.save_raw = myfun
        
        # Criar Explainer
        explainer = shap.TreeExplainer(booster)
        shap_values = explainer.shap_values(X_train)
        
        # Summary Plot Traduzido
        plt.figure(figsize=(12, 10))
        shap.summary_plot(shap_values, X_train, show=False)
        plt.title("Impacto das Variáveis na Sobrevivência (SHAP)")
        plt.xlabel("Valor SHAP (Impacto na saída do modelo)")
        # Traduzir a barra de cores (manual, difícil no summary_plot padrão, mas o título ajuda)
        # O summary_plot padrão usa "Feature value: Low <-> High" em inglês.
        # Podemos tentar sobrescrever o label do eixo X novamente por garantia
        plt.xlabel("Valor SHAP (Impacto na Predição de Risco)")
        
        plt.tight_layout()
        save_plot(plt.gcf(), config.FIGURES_DIR, "shap_summary_pt.png")
        logging.info("SHAP gerado com sucesso!")
        
    except Exception as e:
        logging.error(f"Erro ao gerar SHAP: {e}")
        logging.warning("Seguindo apenas com os plots nativos.")

    logging.info("Exploração concluída! Confira os gráficos em reports/figures/")
    
if __name__ == "__main__":
    explore_xgboost()
