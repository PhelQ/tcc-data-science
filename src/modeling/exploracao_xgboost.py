import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import make_scorer
from sksurv.metrics import concordance_index_censored

from src import config
from src.utils import save_plot
from src.modeling.treino_modelo_sobrevivencia import load_and_split_data, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def c_index_scorer(estimator, X, y):
    """Scorer customizado para RandomizedSearchCV (C-Index).
    
    O y passado pelo sklearn será o y_train (estruturado ou array).
    O XGBoost espera y numérico para fit, mas o scorer recebe o y original do split.
    """
    # Predição de risco (XGBoost survival output é log hazard ratio)
    y_pred = estimator.predict(X)
    
    # Extrair evento e tempo do array estruturado ou dataframe
    # O RandomizedSearchCV do sklearn vai passar y como array estruturado se fizermos o split antes
    # Mas precisamos garantir compatibilidade.
    
    try:
        event = y["event"]
        time = y["time"]
    except:
        # Fallback se for array numpy estruturado
        event = y["event"]
        time = y["time"]
        
    return concordance_index_censored(event, time, y_pred)[0]

def explore_xgboost():
    logging.info("--- Iniciando Sessão de Exploração Avançada do XGBoost ---")
    
    # 1. Carregar Dados
    logging.info(f"Carregando dados de {config.FEATURES_SURVIVAL_PATH}...")
    X, y, cat_features = load_and_split_data(config.FEATURES_SURVIVAL_PATH)
    
    # 2. Split Treino/Teste
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y["event"]
    )
    
    # 3. Encoding (Fit Treino, Transform Teste)
    X_train, X_test = encode_features(X_train_raw, X_test_raw, cat_features)
    
    # Preparar y para XGBoost (Target numérico para fit: positivo=tempo se vivo, negativo=tempo se morto? Não.)
    # Para 'survival:cox', o XGBoost espera:
    # labels: valores de tempo. Se censurado, valor negativo?
    # A documentação oficial diz: "y > 0 para evento, y < 0 para censura" para 'survival:cox'.
    # Vamos converter.
    
    y_train_xgb = np.where(y_train["event"], y_train["time"], -y_train["time"])
    y_test_xgb = np.where(y_test["event"], y_test["time"], -y_test["time"])
    
    # 4. Definir Espaço de Hiperparâmetros (Grid Radical)
    # MELHORES PARAMETROS ENCONTRADOS (C-Index ~0.97)
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
    
    # Pular Random Search para ser rápido na análise de feature importance
    logging.info("Usando melhores parâmetros pré-encontrados para análise rápida...")
    best_score = 0.9638 # Histórico

    logging.info(f"--- Melhor Resultado (Histórico): {best_score:.4f} ---")
    
    # 5. Treinar Melhor Modelo no Treino Completo
    logging.info("Treinando melhor modelo no conjunto de treino completo...")
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train_xgb)
    
    # 6. Avaliar no Teste
    test_preds = final_model.predict(X_test)
    test_score = concordance_index_censored(y_test["event"], y_test["time"], test_preds)[0]
    logging.info(f"Score Final no Teste (Hold-out): {test_score:.4f}")
    
    # 7. Análise de Importância (Nativa XGBoost)
    # Substituindo SHAP (que deu erro de compatibilidade) por plot_importance nativo
    logging.info("Gerando gráfico de importância das features (Nativo XGBoost)...")
    
    from xgboost import plot_importance
    
    plt.figure(figsize=(12, 10))
    plot_importance(final_model, max_num_features=20, importance_type='weight', height=0.5)
    plt.title(f"XGBoost Feature Importance (Weight) - C-Index: {test_score:.3f}")
    plt.tight_layout()
    save_plot(plt.gcf(), config.FIGURES_DIR, "xgboost_native_importance.png")
    
    # Também salvar importância por ganho (Gain) - geralmente mais informativo para relevância
    plt.figure(figsize=(12, 10))
    plot_importance(final_model, max_num_features=20, importance_type='gain', height=0.5)
    plt.title(f"XGBoost Feature Importance (Gain) - C-Index: {test_score:.3f}")
    plt.tight_layout()
    save_plot(plt.gcf(), config.FIGURES_DIR, "xgboost_native_importance_gain.png")

    logging.info("Exploração concluída! Confira os gráficos em reports/figures/")
    
if __name__ == "__main__":
    explore_xgboost()