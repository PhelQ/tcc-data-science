import logging
import pandas as pd
from src import config
from src.utils import load_data, save_data

logging.basicConfig(level=logging.INFO)

def consolidate_data(clinical_df: pd.DataFrame, biospecimen_df: pd.DataFrame, join_key: str) -> pd.DataFrame:
    """
    Consolida os dados clínicos e de biospecimen usando a chave de junção especificada.
    """
    logging.info(f"Iniciando a consolidação de dados na chave '{join_key}'...")

    # Verifica se a chave de junção existe em ambos os dataframes
    if join_key not in clinical_df.columns or join_key not in biospecimen_df.columns:
        raise ValueError(f"A chave de junção '{join_key}' não foi encontrada em um dos dataframes.")

    # Realiza a junção
    df_merged = pd.merge(
        clinical_df,
        biospecimen_df,
        on=join_key,
        how="inner"
    )

    logging.info(f"Shape do dataframe consolidado: {df_merged.shape}")
    return df_merged

def main():
    """
    Executa o pipeline de consolidação de dados.
    """
    logging.info("--- Iniciando pipeline de consolidação de dados ---")
    df_clinical_cleaned = load_data(config.CLINICAL_CLEANED_PATH)
    df_biospecimen_cleaned = load_data(config.BIOSPECIMEN_CLEANED_PATH)
    
    # A chave de junção correta é 'cases.submitter_id'
    join_key = "cases.submitter_id"
    
    df_consolidated = consolidate_data(df_clinical_cleaned, df_biospecimen_cleaned, join_key)
    save_data(df_consolidated, config.CONSOLIDATED_DATA_PATH)
    logging.info(f"Dados consolidados salvos em: {config.CONSOLIDATED_DATA_PATH}")
    logging.info("--- Pipeline de consolidação de dados concluído ---")

if __name__ == "__main__":
    main()

