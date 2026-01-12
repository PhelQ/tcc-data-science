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

def torture_xgboost():
    logging.info("--- Iniciando Sessão de Tortura do XGBoost ---")
    
    # 1. Carregar Dados
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
    param_dist = {
        'max_depth': [3, 4, 5, 6, 8, 10],              # Profundidade
        'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.2], # Taxa de aprendizado
        'n_estimators': [100, 200, 300, 500, 1000],    # Número de árvores
        'min_child_weight': [1, 3, 5, 7],              # Peso mínimo na folha
        'gamma': [0, 0.1, 0.2, 0.5],                   # Redução mínima de perda
        'subsample': [0.6, 0.8, 1.0],                  # Amostra de linhas
        'colsample_bytree': [0.6, 0.8, 1.0],           # Amostra de colunas
        'reg_alpha': [0, 0.1, 0.5, 1.0],               # L1 Regularization
        'reg_lambda': [0.1, 1.0, 5.0, 10.0]            # L2 Regularization
    }
    
    xgb = XGBRegressor(objective='survival:cox', eval_metric='cox-nloglik', n_jobs=-1, random_state=42)
    
    # Nota: RandomizedSearchCV do sklearn não aceita y estruturado nativamente no 'fit' para dividir folds se não for array simples.
    # Mas se passarmos y_train (estruturado) e o estimador esperar y_xgb (float), vai dar erro no fit interno.
    # Solução: Fazer o loop de busca manualmente ou usar um wrapper. 
    # Vou fazer um Random Search manual simples para ter controle total sobre o formato do target.
    
    logging.info("Iniciando Random Search Manual (20 iterações)...")
    
    best_score = -1
    best_params = {}
    best_model = None
    
    import random
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    
    n_iter = 20
    results = []
    
    for i in range(n_iter):
        # Amostrar parâmetros
        params = {k: random.choice(v) for k, v in param_dist.items()}
        params['objective'] = 'survival:cox'
        params['eval_metric'] = 'cox-nloglik'
        params['n_jobs'] = -1
        params['random_state'] = 42
        
        cv_scores = []
        
        for train_idx, val_idx in kf.split(X_train):
            # Split CV
            X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
            
            # Converter y para formato XGBoost
            y_cv_train_xgb = np.where(y_cv_train["event"], y_cv_train["time"], -y_cv_train["time"])
            
            # Treinar
            model = XGBRegressor(**params)
            try:
                model.fit(X_cv_train, y_cv_train_xgb, verbose=False)
                
                # Avaliar (C-Index)
                preds = model.predict(X_cv_val)
                
                # Tratamento de segurança para valores infinitos/NaN
                if not np.isfinite(preds).all():
                    logging.warning(f"Predições infinitas detectadas com params {params}. Ignorando fold.")
                    score = 0.0
                else:
                    score = concordance_index_censored(y_cv_val["event"], y_cv_val["time"], preds)[0]
            except Exception as e:
                logging.warning(f"Erro no treino/avaliação com params {params}: {e}")
                score = 0.0
                
            cv_scores.append(score)
            
        mean_score = np.mean(cv_scores) if cv_scores else 0.0
        logging.info(f"Iter {i+1}/{n_iter}: Score={mean_score:.4f} | Params={params}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            
    logging.info(f"--- Melhor Resultado: {best_score:.4f} ---")
    logging.info(f"Melhores Parâmetros: {best_params}")
    
    # 5. Treinar Melhor Modelo no Treino Completo
    logging.info("Treinando melhor modelo no conjunto de treino completo...")
    final_model = XGBRegressor(**best_params)
    final_model.fit(X_train, y_train_xgb)
    
    # 6. Avaliar no Teste
    test_preds = final_model.predict(X_test)
    test_score = concordance_index_censored(y_test["event"], y_test["time"], test_preds)[0]
    logging.info(f"Score Final no Teste (Hold-out): {test_score:.4f}")
    
    # 7. Análise SHAP
    logging.info("Gerando análise SHAP...")
    # Usar TreeExplainer explicitamente para modelos baseados em árvore
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer(X_test)
    
    # Plot Summary (Beeswarm)
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title(f"SHAP Summary - XGBoost Otimizado (C-Index: {test_score:.3f})")
    plt.tight_layout()
    save_plot(plt.gcf(), config.FIGURES_DIR, "xgboost_shap_summary.png")
    
    # Plot Bar (Importância Global)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("Importância das Features (SHAP Global)")
    plt.tight_layout()
    save_plot(plt.gcf(), config.FIGURES_DIR, "xgboost_shap_importance.png")
    
    logging.info("Tortura concluída! Confira os gráficos em reports/figures/")

if __name__ == "__main__":
    torture_xgboost()