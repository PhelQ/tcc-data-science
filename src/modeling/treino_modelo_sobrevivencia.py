"""
Treina e avalia múltiplos modelos de sobrevivência, salvando o melhor.
"""

import logging

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from xgboost import XGBRegressor

from src import config
from src.utils import load_data, save_data, save_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(data_path: str) -> tuple:
    """Carrega e pré-processa os dados para modelagem.

    Retorna:
        - X_processed (pd.DataFrame): DataFrame de features pronto para o modelo.
        - y (np.ndarray): Array estruturado para análise de sobrevivência.
    """
    logging.info(f"Carregando dados de {data_path}...")
    df = load_data(data_path)

    X = df.drop(["event_occurred", "observed_time"], axis=1)
    y = np.array(
        list(zip(df["event_occurred"], df["observed_time"])),
        dtype=[("event", "bool"), ("time", "float64")],
    )

    logging.info("Pré-processando features (one-hot encoding)...")
    categorical_features = X.select_dtypes(include=["object", "category"]).columns
    X_processed = pd.get_dummies(
        X, columns=categorical_features, drop_first=True, dummy_na=False
    )

    return X_processed, y


def evaluate_model(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> float:
    """Treina e avalia um único modelo de sobrevivência, retornando o C-Index."""
    try:
        if model_name == "CoxPH":
            train_df = X_train.copy()
            train_df["event_occurred"] = y_train["event"]
            train_df["observed_time"] = y_train["time"]
            model.fit(train_df, duration_col="observed_time", event_col="event_occurred")
            y_pred = model.predict_partial_hazard(X_val)

        elif model_name == "XGBoostSurvival":
            y_train_xgb = np.where(y_train["event"], y_train["time"], -y_train["time"])
            model.fit(X_train, y_train_xgb)
            y_pred = model.predict(X_val)

        else:  # RandomSurvivalForest
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

        return concordance_index_censored(y_val["event"], y_val["time"], y_pred)[0]

    except Exception as e:
        logging.error(f"Erro ao avaliar {model_name}: {e}")
        return 0.0


def cross_validate_models(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Avalia múltiplos modelos de sobrevivência usando validação cruzada K-Fold."""
    models = {
        "CoxPH": CoxPHFitter(penalizer=0.1),
        "RandomSurvivalForest": RandomSurvivalForest(random_state=42),
        "XGBoostSurvival": XGBRegressor(
            objective="survival:cox", random_state=42, eval_metric="cox-nloglik"
        ),
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {model_name: [] for model_name in models}

    for model_name, model in models.items():
        logging.info(f"--- Avaliando modelo: {model_name} ---")
        for train_index, val_index in cv.split(X, y):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y[train_index], y[val_index]

            c_index = evaluate_model(model_name, model, X_train, y_train, X_val, y_val)
            if c_index > 0.0:
                results[model_name].append(c_index)

    final_results = {}
    for model_name, c_indices in results.items():
        if c_indices:
            final_results[model_name] = {
                "mean_c_index": np.mean(c_indices),
                "std_c_index": np.std(c_indices),
            }

    return final_results


def train_final_model(model_name: str, model, X: pd.DataFrame, y: np.ndarray):
    """Treina um modelo de sobrevivência com todos os dados."""
    logging.info(f"--- Treinando modelo final ({model_name}) com todos os dados ---")
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


def train_and_save_models(
    X: pd.DataFrame, y: np.ndarray, best_model_name: str
) -> None:
    """Treina o melhor modelo e um modelo CoxPH para interpretação, e os salva."""
    models = {
        "CoxPH": CoxPHFitter(penalizer=0.1),
        "RandomSurvivalForest": RandomSurvivalForest(random_state=42),
        "XGBoostSurvival": XGBRegressor(
            objective="survival:cox", random_state=42, eval_metric="cox-nloglik"
        ),
    }

    best_model = train_final_model(best_model_name, models[best_model_name], X, y)
    save_model(best_model, config.SURVIVAL_MODEL_PATH)
    logging.info(f"Melhor modelo ({best_model_name}) salvo em: {config.SURVIVAL_MODEL_PATH}")

    if best_model_name == "CoxPH":
        save_model(best_model, config.COXPH_MODEL_PATH)
    else:
        cox_model = train_final_model("CoxPH", models["CoxPH"], X, y)
        save_model(cox_model, config.COXPH_MODEL_PATH)
    
    logging.info(f"Modelo CoxPH para interpretação salvo em: {config.COXPH_MODEL_PATH}")


def main():
    """Função principal para executar o pipeline de treinamento e avaliação."""
    logging.info("Iniciando o pipeline de treinamento de modelos de sobrevivência...")
    
    X, y = load_and_preprocess_data(config.FEATURES_SURVIVAL_PATH)
    results = cross_validate_models(X, y)

    if not results:
        logging.warning("Nenhum modelo foi avaliado com sucesso. Encerrando o pipeline.")
        return

    results_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    results_df.rename(columns={"index": "model_name"}, inplace=True)
    
    best_model_name = results_df.loc[results_df["mean_c_index"].idxmax()]["model_name"]
    best_c_index = results_df["mean_c_index"].max()

    logging.info("--- Resultados da Avaliação do Modelo (Validação Cruzada) ---")
    for _, row in results_df.iterrows():
        logging.info(
            f"  - {row['model_name']}: C-Index Médio: {row['mean_c_index']:.4f} "
            f"(± {row['std_c_index']:.4f})"
        )
    
    results_path = f"{config.REPORTS_DIR}/model_evaluation_results.csv"
    save_data(results_df, results_path)
    logging.info(f"Resultados da avaliação salvos em: {results_path}")

    logging.info(f"Melhor modelo selecionado: {best_model_name} (C-Index: {best_c_index:.4f})")
    train_and_save_models(X, y, best_model_name)

    save_model(list(X.columns), config.TRAINING_COLUMNS_PATH)
    logging.info(f"Colunas de treinamento salvas em: {config.TRAINING_COLUMNS_PATH}")
    logging.info("Pipeline de treinamento de modelos de sobrevivência concluído!")


if __name__ == "__main__":
    main()