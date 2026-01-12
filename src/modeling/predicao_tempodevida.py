"""
Este script carrega um modelo de sobrevivência CoxPH treinado e dados clínicos para prever
o tempo mediano de sobrevivência para cada paciente. Os resultados são salvos em um arquivo CSV.
"""

import logging

import pandas as pd

from src import config
from src.modeling.treino_modelo_sobrevivencia import load_and_preprocess_data
from src.utils import load_data, load_model, save_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def align_columns(X_processed: pd.DataFrame, training_columns: list) -> pd.DataFrame:
    """Alinha as colunas do DataFrame de entrada com as colunas de treinamento."""
    logging.info("Alinhando colunas com o modelo treinado...")
    training_cols_index = pd.Index(training_columns)
    missing_cols = training_cols_index.difference(X_processed.columns)
    for col in missing_cols:
        X_processed[col] = 0

    extra_cols = X_processed.columns.difference(training_cols_index)
    X_processed = X_processed.drop(columns=list(extra_cols))

    return X_processed[training_cols_index]


def predict_survival_time(model, X: pd.DataFrame) -> list:
    """Prevê o tempo mediano de sobrevivência para cada paciente.

    Args:
        model: Um modelo de sobrevivência treinado com um método `predict_median`.
        X: Um DataFrame com as features processadas para previsão.

    Returns:
        Uma lista com os tempos medianos de sobrevivência previstos.
    """
    logging.info("Calculando o tempo mediano de sobrevivência...")
    median_survival_times = model.predict_median(X)
    return median_survival_times.tolist()


def main():
    """Função principal para carregar modelo, dados, prever e salvar os resultados."""
    logging.info("Iniciando o pipeline de previsão de tempo de sobrevivência...")

    model = load_model(config.COXPH_MODEL_PATH)
    training_columns = load_model(config.TRAINING_COLUMNS_PATH)

    X_processed, y = load_and_preprocess_data(config.FEATURES_SURVIVAL_PATH)
    X_aligned = align_columns(X_processed, training_columns)

    predicted_times = predict_survival_time(model, X_aligned)

    # Carrega os dados brutos novamente para usar como base para os resultados
    df_raw = load_data(config.FEATURES_SURVIVAL_PATH)
    results_df = df_raw.drop(["event_occurred", "observed_time"], axis=1, errors="ignore").copy()
    results_df["OS"] = y["event"]
    results_df["OS_time"] = y["time"]
    results_df["predicted_survival_time"] = predicted_times

    save_data(results_df, config.PREDICTED_SURVIVAL_TIME_PATH)
    logging.info(f"Resultados da previsão salvos em: {config.PREDICTED_SURVIVAL_TIME_PATH}")
    logging.info("Pipeline de previsão concluído!")


if __name__ == "__main__":
    main()


