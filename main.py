"""
Script principal para executar todo o pipeline.
"""

import logging

from src.analysis import (
    eda_tcga_coad,
    interpret_survival_model,
    predict_survival_time,
    train_survival_model,
    visualize_survival_curves,
)
from src.data import (
    consolidacaodados_tcga_coad,
    feature_engineering_survival,
    preprocessamento_data,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Executa o pipeline completo de processamento de dados e modelagem."""
    try:
        logging.info("Iniciando o pipeline completo...")

        # Etapas de processamento de dados
        logging.info("--- Etapa 1: Pré-processamento de dados ---")
        preprocessamento_data.main()

        logging.info("--- Etapa 2: Consolidando dados brutos ---")
        consolidacaodados_tcga_coad.main()

        logging.info("--- Etapa 3: Engenharia de features ---")
        feature_engineering_survival.main()

        # Etapas de análise e modelagem
        logging.info("--- Etapa 4: Análise Exploratória de Dados (EDA) ---")
        eda_tcga_coad.main()

        logging.info("--- Etapa 5: Treinando modelos de sobrevivência ---")
        train_survival_model.main()

        logging.info("--- Etapa 6: Interpretando o modelo Cox ---")
        interpret_survival_model.main()

        logging.info("--- Etapa 7: Prevendo o tempo de sobrevivência ---")
        predict_survival_time.main()

        logging.info("--- Etapa 8: Visualizando curvas de sobrevivência ---")
        visualize_survival_curves.main()

        logging.info("Pipeline concluído com sucesso!")

    except Exception as e:
        logging.error(f"Ocorreu um erro inesperado durante a execução do pipeline: {e}")
        raise e


if __name__ == "__main__":
    main()