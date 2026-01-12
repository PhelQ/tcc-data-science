"""
Análise Exploratória de Dados para Predição de Sobrevivência no TCGA-COAD.
"""
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lifelines import KaplanMeierFitter

from src import config
from src.utils import load_data, save_plot

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def plot_age_distribution(df, output_dir):
    """Gera o histograma da distribuição de idade."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df['age_at_index'], kde=True, bins=30, color='skyblue')
    plt.title('Distribuição de Idade dos Pacientes')
    plt.xlabel('Idade no Diagnóstico')
    plt.ylabel('Frequência')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(plt.gcf(), output_dir, "distribuicao_idade.png")

def plot_age_by_vital_status(df, output_dir):
    """Gera o boxplot de idade por status vital."""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='vital_status', y='age_at_index', data=df, palette='Set2')
    plt.title('Distribuição de Idade por Status Vital')
    plt.xlabel('Status Vital')
    plt.ylabel('Idade no Diagnóstico')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(plt.gcf(), output_dir, "distribuicao_idade_por_status_vital.png")

def plot_stage_distribution(df, output_dir):
    """Gera o gráfico de contagem para os estágios patológicos (AJCC)."""
    # Limpeza de dados para visualização
    plot_df = df.copy()
    plot_df['ajcc_pathologic_stage'] = plot_df['ajcc_pathologic_stage'].replace("'--", "Não Reportado")
    
    # Ordenar estágios naturalmente se possível, caso contrário por contagem
    order = plot_df['ajcc_pathologic_stage'].value_counts().index

    plt.figure(figsize=(12, 7))
    ax = sns.countplot(y='ajcc_pathologic_stage', data=plot_df, order=order, palette="viridis")
    
    # Adicionar rótulos nas barras
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    plt.title('Distribuição dos Estágios Patológicos (AJCC)')
    plt.xlabel('Contagem')
    plt.ylabel('Estágio')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    save_plot(plt.gcf(), output_dir, "distribuicao_estagios.png")

def plot_overall_survival(df, output_dir):
    """Plota a curva de sobrevivência Kaplan-Meier global."""
    kmf = KaplanMeierFitter()
    kmf.fit(df['observed_time'], event_observed=df['event_occurred'], label='Coorte Global')
    
    plt.figure(figsize=(10, 6))
    kmf.plot(ci_show=True, linewidth=2)
    plt.title('Sobrevivência Global (Kaplan-Meier)')
    plt.xlabel('Tempo (Anos)')
    plt.ylabel('Probabilidade de Sobrevivência')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(plt.gcf(), output_dir, "kaplan_meier_global.png")

def plot_survival_by_stage(df, output_dir):
    """Plota curvas de sobrevivência Kaplan-Meier estratificadas por estágio do câncer."""
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(111)

    # Limpar nomes dos estágios para a legenda
    plot_df = df.copy()
    plot_df['ajcc_pathologic_stage'] = plot_df['ajcc_pathologic_stage'].replace("'--", "Não Reportado")

    for stage, grouped_df in plot_df.groupby('ajcc_pathologic_stage'):
        # Filtrar grupos muito pequenos para evitar poluição visual
        if len(grouped_df) > 10:
            kmf = KaplanMeierFitter()
            kmf.fit(grouped_df['observed_time'], grouped_df['event_occurred'], label=stage)
            kmf.plot(ax=ax, ci_show=False)

    plt.title('Probabilidade de Sobrevivência por Estágio Patológico (AJCC)')
    plt.xlabel('Tempo (Anos)')
    plt.ylabel('Probabilidade de Sobrevivência')
    plt.legend(title='Estágio', loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    save_plot(plt.gcf(), output_dir, "kaplan_meier_por_estagio.png")

def main():
    logging.info("Iniciando Análise Exploratória de Dados (EDA)...")
    
    try:
        df = load_data(config.FEATURES_SURVIVAL_PATH)
        
        # Pré-processamento para EDA
        df["age_at_index"] = pd.to_numeric(df["age_at_index"], errors="coerce")
        df["vital_status"] = df["event_occurred"].map({1: "Falecido", 0: "Vivo"})

        # Gerar gráficos
        plot_age_distribution(df, config.FIGURES_DIR)
        plot_age_by_vital_status(df, config.FIGURES_DIR)
        plot_stage_distribution(df, config.FIGURES_DIR)
        plot_overall_survival(df, config.FIGURES_DIR)
        plot_survival_by_stage(df, config.FIGURES_DIR)

        logging.info(f"EDA concluída com sucesso. Figuras salvas em: {config.FIGURES_DIR}")

    except Exception as e:
        logging.error(f"Falha na EDA: {e}")
        raise

if __name__ == "__main__":
    main()
