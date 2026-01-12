"""
Treina e avalia múltiplos modelos de sobrevivência (CoxPH, RSF, XGBoost),
gerando um relatório comparativo e salvando todos os artefatos.
"""

import logging
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from xgboost import XGBRegressor

from src import config
from src.utils import load_data, save_data, save_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_split_data(data_path: str) -> tuple:
    """Carrega os dados e realiza o split Treino/Teste preservando tipos.
    
    NÃO realiza One-Hot Encoding aqui para evitar Data Leakage.
    """
    logging.info(f"Carregando dados de {data_path}...")
    df = load_data(data_path)

    X = df.drop(["event_occurred", "observed_time"], axis=1)
    y = np.array(
        list(zip(df["event_occurred"], df["observed_time"])),
        dtype=[("event", "bool"), ("time", "float64")],
    )
    
    # Identificar colunas categóricas para processamento posterior
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    logging.info(f"Features categóricas identificadas: {categorical_features}")
    
    return X, y, categorical_features

def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame, cat_features: list) -> tuple:
    """Realiza One-Hot Encoding fitando APENAS no treino e transformando o teste.
    
    Evita Data Leakage garantindo que o modelo não conheça categorias
    que só existem no conjunto de teste.
    """
    logging.info("Realizando One-Hot Encoding (Fit no Treino, Transform no Teste)...")
    
    # Se não houver features categóricas, retorna como está
    if not cat_features:
        return X_train, X_test

    # Usamos pd.get_dummies no treino para definir as colunas base
    X_train_enc = pd.get_dummies(X_train, columns=cat_features, drop_first=True, dummy_na=False)
    
    # Aplicamos as mesmas transformações no teste
    X_test_enc = pd.get_dummies(X_test, columns=cat_features, drop_first=True, dummy_na=False)
    
    # Alinhamento de colunas (Garantiar que Teste tenha exatamente as colunas do Treino)
    # 1. Adicionar colunas faltantes no teste (preenchidas com 0)
    missing_cols = set(X_train_enc.columns) - set(X_test_enc.columns)
    for c in missing_cols:
        X_test_enc[c] = 0
        
    # 2. Remover colunas extras no teste (que não existiam no treino)
    extra_cols = set(X_test_enc.columns) - set(X_train_enc.columns)
    if extra_cols:
        logging.warning(f"Ignorando colunas inéditas no teste: {extra_cols}")
        X_test_enc = X_test_enc.drop(columns=extra_cols)
        
    # 3. Reordenar para garantir a mesma ordem
    X_test_enc = X_test_enc[X_train_enc.columns]
    
    return X_train_enc, X_test_enc


def get_models_config() -> dict:
    """Retorna a configuração dos modelos a serem treinados."""
    return {
        "CoxPH": CoxPHFitter(penalizer=0.1),
        "RandomSurvivalForest": RandomSurvivalForest(
            random_state=42, 
            min_samples_leaf=15, 
            max_depth=10, 
            n_estimators=200
        ),
        "XGBoostSurvival": XGBRegressor(
            objective="survival:cox",
            random_state=42,
            eval_metric="cox-nloglik",
            max_depth=3,            # Evita overfitting
            learning_rate=0.05,     # Aprendizado mais lento e estável
            min_child_weight=5,     # Exige mais amostras para criar nó
            reg_alpha=0.5,          # Regularização L1
            reg_lambda=0.5          # Regularização L2
        ),
    }


def fit_model(model_name: str, model, X: pd.DataFrame, y: np.ndarray):
    """Função auxiliar para treinar um modelo (abstrai diferenças de API)."""
    if model_name == "CoxPH":
        fit_df = X.copy()
        fit_df["event_occurred"] = y["event"]
        fit_df["observed_time"] = y["time"]
        model.fit(fit_df, duration_col="observed_time", event_col="event_occurred")
    elif model_name == "XGBoostSurvival":
        y_xgb = np.where(y["event"], y["time"], -y["time"])
        model.fit(X, y_xgb)
    else:  # RandomSurvivalForest
        model.fit(X, y)
    return model


def predict_risk(model_name: str, model, X: pd.DataFrame) -> np.ndarray:
    """Função auxiliar para prever risco (abstrai diferenças de API)."""
    if model_name == "CoxPH":
        return model.predict_partial_hazard(X)
    elif model_name == "XGBoostSurvival":
        return model.predict(X)
    else:  # RandomSurvivalForest
        return model.predict(X)


def evaluate_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> float:
    """Treina e avalia um único modelo (usado na validação cruzada)."""
    try:
        # Clona e treina
        # Nota: Sklearn e XGBoost têm clone, mas CoxPHFitter não segue scikit-learn API estritamente
        # Vamos assumir que recebemos uma instância nova ou reinicializável
        # Como fit_model altera o estado, na CV precisamos garantir que não vazamos info.
        # A função cross_validate_models instancia novos modelos ou usa clone se possível.
        # Aqui simplificamos: o chamador garante a instância limpa.
        
        fit_model(model_name, model, X_train, y_train)
        y_pred = predict_risk(model_name, model, X_val)
        
        return concordance_index_censored(y_val["event"], y_val["time"], y_pred)[0]

    except Exception as e:
        logging.error(f"Erro ao avaliar {model_name}: {e}")
        return 0.0


def cross_validate_models(X: pd.DataFrame, y: np.ndarray, cat_features: list) -> dict:
    """Avalia múltiplos modelos usando validação cruzada K-Fold com pipeline correto."""
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    model_names = get_models_config().keys()
    results = {name: [] for name in model_names}

    logging.info("--- Iniciando Validação Cruzada (5-Fold) ---")

    for model_name in model_names:
        logging.info(f"Avaliando: {model_name}")
        fold = 1
        for train_index, val_index in cv.split(X, y):
            # Split nos dados brutos
            X_train_cv_raw, X_val_cv_raw = X.iloc[train_index], X.iloc[val_index]
            y_train_cv, y_val_cv = y[train_index], y[val_index]
            
            # Feature Engineering isolado por fold (Evita Data Leakage)
            X_train_cv, X_val_cv = encode_features(X_train_cv_raw, X_val_cv_raw, cat_features)
            
            # Instancia novo modelo limpo
            model = get_models_config()[model_name]

            c_index = evaluate_model(model_name, model, X_train_cv, y_train_cv, X_val_cv, y_val_cv)
            if c_index > 0.0:
                results[model_name].append(c_index)
            fold += 1

    final_results = {}
    for model_name, c_indices in results.items():
        if c_indices:
            final_results[model_name] = {
                "mean_c_index": np.mean(c_indices),
                "std_c_index": np.std(c_indices),
            }

    return final_results


def main():
    """Função principal para executar o pipeline comparativo."""
    logging.info("Iniciando pipeline comparativo de modelos de sobrevivência...")
    
    # 1. Carregar e Split Inicial (Treino/Teste)
    # O split acontece ANTES de qualquer encoding para garantir isolamento total do teste
    X, y, cat_features = load_and_split_data(config.FEATURES_SURVIVAL_PATH)
    
    # 2. Split Treino/Teste (80/20)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y["event"]
    )
    logging.info(f"Dados divididos: Treino={len(X_train_raw)}, Teste={len(X_test_raw)}")
    
    # 3. Validação Cruzada no Treino (Nested CV logic)
    # Passamos os dados brutos e a lista de features categóricas para que o encoding
    # ocorra dentro de cada fold
    cv_results = cross_validate_models(X_train_raw, y_train, cat_features)
    
    # 4. Treinamento Final e Avaliação no Teste
    logging.info("--- Treinando modelos finais e avaliando no Teste ---")
    
    # Agora sim aplicamos o encoding final no conjunto de treino completo e no teste
    X_train_final, X_test_final = encode_features(X_train_raw, X_test_raw, cat_features)
    
    # Salvar datasets processados (snapshot do que foi usado no treino final)
    train_df = X_train_final.copy()
    train_df["event_occurred"] = y_train["event"]
    train_df["observed_time"] = y_train["time"]
    save_data(train_df, config.TRAIN_DATA_PATH)
    
    test_df = X_test_final.copy()
    test_df["event_occurred"] = y_test["event"]
    test_df["observed_time"] = y_test["time"]
    save_data(test_df, config.TEST_DATA_PATH)
    
    models = get_models_config()
    final_metrics = []
    
    best_model_name = None
    best_test_c_index = -1.0
    
    for name, model in models.items():
        # Treinar em TODO o conjunto de treino
        logging.info(f"Treinando {name}...")
        fit_model(name, model, X_train_final, y_train)
        
        # Avaliar no Teste
        y_pred_test = predict_risk(name, model, X_test_final)
        c_index_test = concordance_index_censored(y_test["event"], y_test["time"], y_pred_test)[0]
        
        # Recuperar métricas de CV
        cv_mean = cv_results.get(name, {}).get("mean_c_index", 0.0)
        cv_std = cv_results.get(name, {}).get("std_c_index", 0.0)
        
        logging.info(f"  > {name}: Test C-Index = {c_index_test:.4f} (CV Mean: {cv_mean:.4f})")
        
        final_metrics.append({
            "model_name": name,
            "cv_mean_c_index": cv_mean,
            "cv_std_c_index": cv_std,
            "test_c_index": c_index_test
        })
        
        # Salvar modelo específico
        if name == "CoxPH":
            save_model(model, config.COXPH_MODEL_PATH)
        elif name == "RandomSurvivalForest":
            save_model(model, config.RSF_MODEL_PATH)
        elif name == "XGBoostSurvival":
            save_model(model, config.XGB_MODEL_PATH)
            
        # Verificar se é o melhor
        if c_index_test > best_test_c_index:
            best_test_c_index = c_index_test
            best_model_name = name

    # 5. Salvar Relatório
    results_df = pd.DataFrame(final_metrics)
    results_path = f"{config.REPORTS_DIR}/model_comparison_results.csv"
    save_data(results_df, results_path)
    logging.info(f"Relatório comparativo salvo em: {results_path}")
    
    # 6. Salvar Melhor Modelo como Padrão
    logging.info(f"Melhor modelo selecionado (baseado no Teste): {best_model_name}")
    
    # Recarregar o melhor modelo salvo para salvar como padrão
    if best_model_name == "CoxPH":
        # CoxPHFitter não carrega fácil com load_model se não for pickle, mas aqui está em memória
        best_model_instance = models["CoxPH"]
    elif best_model_name == "RandomSurvivalForest":
        best_model_instance = models["RandomSurvivalForest"]
    else:
        best_model_instance = models["XGBoostSurvival"]
        
    save_model(best_model_instance, config.SURVIVAL_MODEL_PATH)
    logging.info(f"Melhor modelo salvo como padrão em: {config.SURVIVAL_MODEL_PATH}")
    
    # Salvar colunas de treino (Importante: usar colunas finais pós-encoding)
    save_model(list(X_train_final.columns), config.TRAINING_COLUMNS_PATH)
    logging.info("Pipeline comparativo concluído com sucesso!")


if __name__ == "__main__":
    main()
