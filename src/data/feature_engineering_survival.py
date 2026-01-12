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
    """Cria as variáveis de evento e tempo observado de forma robusta."""
    logging.info("Criando variáveis de evento e tempo...")
    
    # 1. Definir evento
    df["event_occurred"] = (df["vital_status"] == "Dead").astype(int)
    
    # 2. Calcular tempo observado com lógica condicional estrita
    # Se morto -> days_to_death
    # Se vivo -> days_to_last_follow_up
    # Se ambos nulos ou inconsistentes -> NaN (serão removidos depois)
    
    import numpy as np
    
    # Inicializa com NaN
    df["observed_days"] = np.nan
    
    # Máscaras
    mask_dead = df["event_occurred"] == 1
    mask_alive = df["event_occurred"] == 0
    
    # Atribuição condicional
    df.loc[mask_dead, "observed_days"] = df.loc[mask_dead, "days_to_death"]
    df.loc[mask_alive, "observed_days"] = df.loc[mask_alive, "days_to_last_follow_up"]
    
    # Remover pacientes sem informação de tempo
    initial_count = len(df)
    df = df.dropna(subset=["observed_days"])
    dropped_nan = initial_count - len(df)
    if dropped_nan > 0:
        logging.warning(f"Removidos {dropped_nan} pacientes sem informação de tempo (NaN).")
    
    # Tratar tempos <= 0 (Diagnóstico e óbito/seguimento no mesmo dia)
    # Adicionamos 1 dia (1/365.25 anos) para evitar erro matemático em modelos Cox (log(0))
    # e para não descartar esses casos críticos de mortalidade imediata.
    mask_zero_neg = df["observed_days"] <= 0
    n_zeros = mask_zero_neg.sum()
    if n_zeros > 0:
        logging.info(f"Corrigindo {n_zeros} registros com tempo <= 0 adicionando 1 dia.")
        df.loc[mask_zero_neg, "observed_days"] = 1.0
        
    # Converter para anos
    df["observed_time"] = df["observed_days"] / 365.25
    
    return df

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
