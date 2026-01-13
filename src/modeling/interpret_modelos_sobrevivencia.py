"""
Interpreta o modelo de sobrevivência CoxPH e visualiza os Hazard Ratios.
"""

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.utils import load_model, save_plot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def get_hazard_ratios(model, training_columns: list) -> pd.DataFrame:
    """Extrai e calcula os Hazard Ratios (Razões de Risco) do modelo CoxPH."""
    # Use exp(coef) directly from summary, which IS the hazard ratio
    # Do not overwrite index with training_columns to avoid potential mismatch
    summary = model.summary.copy()
    
    # Rename for clarity if needed, or just use exp(coef)
    summary["Hazard Ratio"] = summary["exp(coef)"]
    
    # Verify if indices roughly match expectation (optional, but good for debugging)
    # logging.info(f"Features in model: {summary.index.tolist()[:5]}")
    
    return summary.sort_values(by="coef", key=abs, ascending=False)


def _clean_feature_names(feature_names: pd.Index) -> list:
    """Limpa os nomes das features para uma melhor visualização."""
    cleaned_names = []
    prefixes_to_remove = [
        "tissue_or_organ_of_origin_",
        "ajcc_pathologic_stage_",
        "gender_",
        "race_",
        "ethnicity_",
        "age_group_"
    ]
    for name in feature_names:
        new_name = name
        for prefix in prefixes_to_remove:
            if new_name.startswith(prefix):
                new_name = new_name.replace(prefix, "")
                break
        cleaned_names.append(new_name.replace("_", " ").title())
    return cleaned_names


def plot_hazard_ratios(hazard_ratios: pd.DataFrame, output_path: str) -> None:
    """Plota as 20 features mais importantes com base nos Hazard Ratios."""
    import matplotlib.ticker as ticker

    top_features = hazard_ratios.head(20).copy()
    cleaned_labels = _clean_feature_names(top_features.index)
    top_features["Feature Name"] = cleaned_labels
    
    # Define cor baseada no risco
    top_features["Tipo de Efeito"] = top_features["Hazard Ratio"].apply(
        lambda x: "Fator de Risco (Piora)" if x > 1 else "Fator Protetor (Melhora)"
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot usando hue para cores automáticas e legenda correta
    sns.barplot(
        data=top_features,
        x="Hazard Ratio",
        y="Feature Name",
        hue="Tipo de Efeito",
        palette={"Fator de Risco (Piora)": "#e74c3c", "Fator Protetor (Melhora)": "#2ecc71"},
        ax=ax,
        dodge=False 
    )

    # Linha de referência em 1
    ax.axvline(x=1, color="black", linestyle="--", linewidth=1.5, alpha=0.7)
    
    # Escala logarítmica para simetria visual (0.5 vs 2.0)
    ax.set_xscale("log")
    
    # Ajustar ticks do eixo X para serem legíveis
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())

    ax.set_title(
        "Impacto das Variáveis na Sobrevivência (Hazard Ratios)",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Hazard Ratio (Escala Log)\n< 1: Reduz Risco (Melhor) | > 1: Aumenta Risco (Pior)", fontsize=12)
    ax.set_ylabel("", fontsize=12)
    
    # Move a legenda para não cobrir dados
    ax.legend(title="Interpretação", loc="lower right")

    plt.tight_layout()
    save_plot(fig, config.FIGURES_DIR, "razoes_risco_cox.png")


def main():
    """Função principal para executar a interpretação do modelo."""
    logging.info("Iniciando a interpretação do modelo de sobrevivência...")

    model = load_model(config.COXPH_MODEL_PATH)
    training_columns = load_model(config.TRAINING_COLUMNS_PATH)

    hazard_ratios = get_hazard_ratios(model, training_columns)
    plot_hazard_ratios(hazard_ratios, config.FIGURES_DIR)

    logging.info(
        f"Interpretação do modelo concluída! Gráfico salvo em: {config.COX_HAZARD_RATIOS_PATH}"
    )


if __name__ == "__main__":
    main()

