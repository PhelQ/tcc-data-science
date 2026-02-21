"""
Arquivo de configuração para o projeto.
"""

import os

# Diretório raiz do projeto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Diretórios de dados
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# Caminhos para dados brutos
CLINICAL_RAW_PATH = os.path.join(
    RAW_DATA_DIR, 'clinical.project-tcga-coad.2025-10-21', 'clinical.tsv'
)
BIOSPECIMEN_RAW_PATH = os.path.join(
    RAW_DATA_DIR, 'biospecimen.project-tcga-coad.2025-11-04', 'sample.tsv'
)

# Diretório de modelos
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Diretório de relatórios
REPORTS_DIR = os.path.join(ROOT_DIR, 'reports')
REPORTS_EDA_DIR = os.path.join(REPORTS_DIR, 'eda')
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures')

# Caminhos para dados intermediários
CLINICAL_CLEANED_PATH = os.path.join(
    INTERIM_DATA_DIR, 'tcga_coad_clinical_cleaned.parquet'
)
BIOSPECIMEN_CLEANED_PATH = os.path.join(
    INTERIM_DATA_DIR, 'biospecimen_cleaned.parquet'
)
CONSOLIDATED_DATA_PATH = os.path.join(
    INTERIM_DATA_DIR, 'tcga_coad_consolidated.parquet'
)

# Caminhos para dados processados
FEATURES_FINAL_PATH = os.path.join(
    PROCESSED_DATA_DIR, 'tcga_coad_features_final.parquet'
)
FEATURES_SURVIVAL_PATH = os.path.join(
    PROCESSED_DATA_DIR, 'tcga_coad_features_survival.parquet'
)
TRAIN_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR, 'tcga_coad_train.parquet'
)
TEST_DATA_PATH = os.path.join(
    PROCESSED_DATA_DIR, 'tcga_coad_test.parquet'
)

# Caminhos para modelos
COXPH_MODEL_PATH = os.path.join(MODELS_DIR, 'coxph_model.joblib')
RSF_MODEL_PATH = os.path.join(MODELS_DIR, 'rsf_model.joblib')
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_model.joblib')
SURVIVAL_MODEL_PATH = os.path.join(MODELS_DIR, 'survival_model.joblib')
TRAINING_COLUMNS_PATH = os.path.join(MODELS_DIR, 'training_columns.joblib')

# Caminhos para relatórios
PREDICTED_SURVIVAL_TIME_PATH = os.path.join(
    REPORTS_DIR, 'predicted_survival_time.csv'
)
COX_HAZARD_RATIOS_PATH = os.path.join(
    FIGURES_DIR, 'razoes_risco_cox.png'
)

# Mapeamento de colunas para a engenharia de features de sobrevivência
SURVIVAL_COLUMNS_MAP = {
    "demographic.vital_status": "vital_status",
    "diagnoses.days_to_last_follow_up": "days_to_last_follow_up",
    "demographic.days_to_death": "days_to_death",
    "demographic.age_at_index": "age_at_index",
    "diagnoses.ajcc_pathologic_stage": "ajcc_pathologic_stage",
    "diagnoses.tissue_or_organ_of_origin": "tissue_or_organ_of_origin",
}

# Features finais para o modelo de sobrevivência
FINAL_SURVIVAL_FEATURES = [
    "event_occurred",
    "observed_time",
    "age_at_index",
    "ajcc_pathologic_stage",
    "tissue_or_organ_of_origin",
]

# ============================================================
# Paleta de Cores Padronizada do Projeto
# ============================================================
PALETTE = {
    # --- Cores Semânticas (significado fixo em todo o projeto) ---
    'alive': '#3498DB',          # Azul — Vivo / Censurado / Protetor
    'dead': '#E74C3C',           # Vermelho — Falecido / Evento / Risco
    'alive_dark': '#2471A3',     # Azul escuro — medianas, linhas de destaque
    'dead_dark': '#C0392B',      # Vermelho escuro — medianas, linhas de destaque

    # --- Cor Primária (barras neutras, histogramas gerais) ---
    'primary': '#2C6FAC',

    # --- Grupos de Risco (prognóstico: bom → ruim) ---
    'risk_low': '#27AE60',       # Verde
    'risk_medium': '#F39C12',    # Âmbar
    'risk_high': '#E74C3C',      # Vermelho

    # --- Estágios Patológicos (I→IV: progressão de severidade) ---
    'stages': ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C'],
    'stage_labels': ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],

    # --- Paleta Categórica Geral (localizações, faixas etárias etc.) ---
    'categorical': [
        '#2C6FAC', '#27AE60', '#E67E22', '#9B59B6',
        '#1ABC9C', '#34495E', '#E74C3C',
    ],

    # --- Acento (taxa de óbito, destaques secundários) ---
    'accent': '#E67E22',

    # --- Neutro (elementos não reportados, faltantes) ---
    'neutral': '#95A5A6',
}