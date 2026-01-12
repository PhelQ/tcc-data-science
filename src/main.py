import logging

from src.aed import aed_tcga_coad
from src.modeling import (
    interpret_modelos_sobrevivencia,
    predicao_tempodevida,
    treino_modelo_sobrevivencia,
)
from src.visualization import visualize_survival_curves
from src.data import feature_engineering_survival

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    """Função principal para orquestrar o pipeline de análise de sobrevivência."""
    logging.info("Iniciando o pipeline completo de análise de sobrevivência...")

    # 1. Engenharia de features
    logging.info("Etapa 1: Engenharia de Features")
    feature_engineering_survival.main()

    # 2. Análise Exploratória de Dados (EDA)
    logging.info("Etapa 2: Análise Exploratória de Dados")
    aed_tcga_coad.main()

    # 3. Treinamento e Avaliação de Modelos
    logging.info("Etapa 3: Treinamento e Avaliação de Modelos")
    treino_modelo_sobrevivencia.main()

    # 4. Interpretação do Modelo
    logging.info("Etapa 4: Interpretação do Modelo")
    interpret_modelos_sobrevivencia.main()

    # 5. Previsão do Tempo de Sobrevivência
    logging.info("Etapa 5: Previsão do Tempo de Sobrevivência")
    predicao_tempodevida.main()

    # 6. Visualização das Curvas de Sobrevivência
    logging.info("Etapa 6: Visualização das Curvas de Sobrevivência")
    visualize_survival_curves.main()

    logging.info("Pipeline completo de análise de sobrevivência concluído!")

if __name__ == "__main__":
    main()