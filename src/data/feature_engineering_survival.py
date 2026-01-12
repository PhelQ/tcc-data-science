"""
Cria features para o modelo de sobrevivência a partir dos dados consolidados.
"""

import logging
import pandas as pd
from src import config
from src.utils import load_data, save_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def rename_and_select_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e renomeia as colunas de sobrevivência."""
    logging.info("Selecionando e renomeando colunas...")
    return df[list(config.SURVIVAL_COLUMNS_MAP.keys())].rename(columns=config.SURVIVAL_COLUMNS_MAP)

def filter_colon_samples(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra apenas amostras de tecido de cólon."""
    logging.info("Filtrando amostras de tecido de cólon...")
    valid_sites = [
        "Sigmoid colon", "Ascending colon", "Colon, NOS", "Cecum", 
        "Transverse colon", "Descending colon", "Rectosigmoid junction", 
        "Hepatic flexure of colon", "Splenic flexure of colon"
    ]
    initial_len = len(df)
    df_filtered = df[df["tissue_or_organ_of_origin"].isin(valid_sites)]
    logging.info(f"Removidas {initial_len - len(df_filtered)} amostras não-cólon.")
    return df_filtered

def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas para o tipo numérico."""
    logging.info("Convertendo colunas para numérico...")
    for col in ["days_to_last_follow_up", "days_to_death", "age_at_index"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def create_event_and_time_vars(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as variáveis de evento e tempo observado."""
    logging.info("Criando variáveis de evento e tempo...")
    df["event_occurred"] = (df["vital_status"] == "Dead").astype(int)
    df["observed_time"] = df[["days_to_last_follow_up", "days_to_death"]].max(axis=1) / 365.25
    return df[df["observed_time"] > 0]

def discretize_age(df: pd.DataFrame) -> pd.DataFrame:
    """Discretizes age into bins."""
    logging.info("Discretizing age into bins...")
    age_bins = [0, 40, 50, 60, 70, 80, 100]
    age_labels = [f"age_{age_bins[i]}_{age_bins[i+1]}" for i in range(len(age_bins)-1)]
    df["age_group"] = pd.cut(df["age_at_index"], bins=age_bins, labels=age_labels, right=False)
    return df

def translate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Traduz os nomes das colunas para o português."""
    logging.info("Traduzindo nomes de colunas...")
    return df.rename(columns={
        "age_at_index": "idade",
        "ajcc_pathologic_stage": "estagio_patologico_ajcc",
        "tissue_or_organ_of_origin": "tecido_ou_orgao_de_origem",
    })

def main():
    """Executa o pipeline de criação de features de sobrevivência."""
    logging.info("--- Iniciando pipeline de criação de features de sobrevivência ---")

    df_consolidated = load_data(config.CONSOLIDATED_DATA_PATH)
    
    df_survival = (
        df_consolidated.pipe(rename_and_select_columns)
        .pipe(filter_colon_samples)
        .pipe(convert_to_numeric)
        .pipe(create_event_and_time_vars)
        .pipe(discretize_age)
    )
    
    df_final = df_survival[config.FINAL_SURVIVAL_FEATURES]

    save_data(df_final, config.FEATURES_SURVIVAL_PATH)
    logging.info(f"Shape do dataframe final: {df_final.shape}")
    logging.info(f"Colunas do dataframe final: {df_final.columns.tolist()}")
    logging.info("--- Pipeline de criação de features de sobrevivência concluído ---")

if __name__ == "__main__":
    main()
