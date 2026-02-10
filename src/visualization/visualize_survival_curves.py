"""
Gera as visualizações das curvas de sobrevivência.
"""

import logging
import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
import seaborn as sns

from src import config
from src.utils import load_data, save_plot, load_model
from src.modeling.predict import align_columns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _setup_plot(ax, title, xlabel, ylabel):
    """Configura os elementos de um gráfico matplotlib."""
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()

def preprocess_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pré-processa features para o modelo de sobrevivência (one-hot encoding)."""
    logging.info("Pré-processando features (one-hot encoding)...")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_processed

def plot_survival_by_risk_group(df: pd.DataFrame, figures_dir: str):
    """Gera e salva as curvas de sobrevivência por grupos de risco (alto, médio, baixo)."""
    logging.info("Gerando curvas de sobrevivência por grupos de risco previstos pelo modelo...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = {
        "baixo": config.PALETTE['risk_low'],
        "médio": config.PALETTE['risk_medium'],
        "alto": config.PALETTE['risk_high'],
    }

    # A FutureWarning é esperada aqui e será tratada em versões futuras do pandas
    for group, data in df.groupby("risk_group"):
        kmf = KaplanMeierFitter()
        kmf.fit(data["observed_time"], data["event_occurred"], label=f"Risco {group.capitalize()}")
        kmf.plot_survival_function(ax=ax, color=colors[group], ci_show=True)

    _setup_plot(
        ax,
        "Curvas de Sobrevivência de Kaplan-Meier por Grupo de Risco",
        "Tempo (Anos)",
        "Probabilidade de Sobrevivência",
    )
    save_plot(fig, figures_dir, "curvas_sobrevivencia_por_grupos_risco.png")


def main():
    """Executa o pipeline de visualização de sobrevivência."""
    logging.info("Iniciando a geração de visualizações de sobrevivência...")

    # Configurar estilo visual
    sns.set_theme(style="whitegrid")

    # Carregar dados e modelo
    df = load_data(config.FEATURES_SURVIVAL_PATH)
    model = load_model(config.SURVIVAL_MODEL_PATH)
    training_columns = load_model(config.TRAINING_COLUMNS_PATH)

    # Preparar dados para previsão
    X = df.drop(columns=["event_occurred", "observed_time"])
    X_processed = preprocess_features(X)
    X_aligned = align_columns(X_processed, training_columns)

    # Prever risco e adicionar ao dataframe
    risk_scores = model.predict(X_aligned)
    df["risk_group"] = pd.qcut(risk_scores, q=3, labels=["baixo", "médio", "alto"])

    # Gerar e salvar visualizações
    plot_survival_by_risk_group(df, config.FIGURES_DIR)

    logging.info("Visualizações de sobrevivência geradas com sucesso!")


if __name__ == "__main__":
    main()
