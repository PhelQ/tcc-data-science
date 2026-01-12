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
    summary = model.summary
    summary["Hazard Ratio"] = model.hazard_ratios_
    summary.index = training_columns
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
    top_features = hazard_ratios.head(20)
    cleaned_labels = _clean_feature_names(top_features.index)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))

    colors = ["#3498db" if hr < 1 else "#e74c3c" for hr in top_features["Hazard Ratio"]]

    sns.barplot(
        x=top_features["Hazard Ratio"],
        y=cleaned_labels,
        palette=colors,
        ax=ax,
        orient="h",
    )

    ax.axvline(x=1, color="black", linestyle="--", linewidth=1.5)

    ax.set_title(
        "Top 20 Fatores de Risco - Modelo CoxPH",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Hazard Ratio (Razão de Risco)", fontsize=12)
    ax.set_ylabel("Variável", fontsize=12)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3498db", label="Fator Protetor (HR < 1)"),
        plt.Rectangle((0, 0), 1, 1, color="#e74c3c", label="Fator de Risco (HR > 1)"),
        plt.Line2D([], [], color='black', linestyle='--', label='Sem Efeito (HR=1)')
    ]
    ax.legend(handles=legend_handles, loc="best")

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

