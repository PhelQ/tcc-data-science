import logging
import pandas as pd
from src import config
from src.utils import load_data, save_data

logging.basicConfig(level=logging.INFO)

def main():
    """
    Carrega os dados brutos (clinical e sample) e os salva no formato parquet
    sem realizar modificações nas colunas.
    """
    logging.info("--- Iniciando pipeline de pré-processamento de dados ---")

    # Carrega os dados a partir dos caminhos definidos no config
    logging.info(f"Processando arquivo clínico: {config.CLINICAL_RAW_PATH}")
    df_clinical = load_data(config.CLINICAL_RAW_PATH)
    
    logging.info(f"Processando arquivo de amostra: {config.BIOSPECIMEN_RAW_PATH}")
    df_biospecimen = load_data(config.BIOSPECIMEN_RAW_PATH)

    # Salva os dados limpos (convertidos para parquet)
    logging.info(f"Salvando dados clínicos em {config.CLINICAL_CLEANED_PATH}")
    save_data(df_clinical, config.CLINICAL_CLEANED_PATH)

    logging.info(f"Salvando dados de biospecimen em {config.BIOSPECIMEN_CLEANED_PATH}")
    save_data(df_biospecimen, config.BIOSPECIMEN_CLEANED_PATH)

    logging.info("--- Pipeline de pré-processamento de dados concluído ---")

if __name__ == "__main__":
    main()
