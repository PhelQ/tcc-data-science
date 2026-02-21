"""
Predição de Tempo de Sobrevivência.

Carrega o melhor modelo treinado e realiza predições de tempo mediano
de sobrevivência para cada paciente. Salva os resultados em CSV.
"""

import logging

import numpy as np
import pandas as pd

from src import config
from src.modeling.train import load_and_split_data
from src.utils import load_data, load_model, save_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_for_prediction(X: pd.DataFrame, cat_features: list) -> pd.DataFrame:
    """Realiza o encoding simplificado para predição."""
    logging.info("Realizando One-Hot Encoding para predição...")
    return pd.get_dummies(X, columns=cat_features, drop_first=True)

def align_columns(X_processed: pd.DataFrame, training_columns: list) -> pd.DataFrame:
    """Alinha as colunas do DataFrame de entrada com as colunas de treinamento."""
    logging.info("Alinhando colunas com o modelo treinado...")
    training_cols_index = pd.Index(training_columns)
    
    # Adicionar colunas faltantes
    missing_cols = training_cols_index.difference(X_processed.columns)
    for col in missing_cols:
        X_processed[col] = 0

    # Remover colunas extras
    extra_cols = X_processed.columns.difference(training_cols_index)
    if not extra_cols.empty:
        X_processed = X_processed.drop(columns=list(extra_cols))

    return X_processed[training_cols_index]


def predict_survival_time(model, X: pd.DataFrame) -> list:
    """Prevê o tempo mediano de sobrevivência para cada paciente via RSF.

    Usa predict_survival_function do RSF para obter a curva de sobrevivência
    de cada paciente e extrai o tempo mediano (ponto onde S(t) cruza 0.5).

    Args:
        model: RandomSurvivalForest treinado.
        X: DataFrame com as features processadas para previsão.

    Returns:
        Lista com os tempos medianos de sobrevivência previstos.
    """
    logging.info("Calculando o tempo mediano de sobrevivência via RSF...")
    survival_funcs = model.predict_survival_function(X)
    medians = []
    for fn in survival_funcs:
        # fn.x = pontos de tempo, fn.y = probabilidades de sobrevivência
        idx = np.searchsorted(-fn.y, -0.5)
        if idx < len(fn.x):
            medians.append(fn.x[idx])
        else:
            medians.append(fn.x[-1])
    return medians


def main():
    """Função principal para carregar modelo, dados, prever e salvar os resultados."""
    logging.info("Iniciando o pipeline de previsão de tempo de sobrevivência...")

    model = load_model(config.RSF_MODEL_PATH)
    training_columns = load_model(config.TRAINING_COLUMNS_PATH)

    # Carregar dados usando a função correta
    X, y, cat_features = load_and_split_data(config.FEATURES_SURVIVAL_PATH)
    
    # Processar
    X_processed = preprocess_for_prediction(X, cat_features)
    X_aligned = align_columns(X_processed, training_columns)

    predicted_times = predict_survival_time(model, X_aligned)

    # Carrega os dados brutos novamente para usar como base para os resultados
    df_raw = load_data(config.FEATURES_SURVIVAL_PATH)
    results_df = df_raw.drop(["event_occurred", "observed_time"], axis=1, errors="ignore").copy()
    results_df["OS"] = [val[0] for val in y]
    results_df["OS_time"] = [val[1] for val in y]
    results_df["predicted_survival_time"] = predicted_times

    save_data(results_df, config.PREDICTED_SURVIVAL_TIME_PATH)
    logging.info(f"Resultados da previsão salvos em: {config.PREDICTED_SURVIVAL_TIME_PATH}")
    logging.info("Pipeline de previsão concluído!")


if __name__ == "__main__":
    main()


