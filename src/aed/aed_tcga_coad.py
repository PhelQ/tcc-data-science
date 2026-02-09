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

def plot_tissue_origin_distribution(df, output_dir):
    """Gera o gráfico de distribuição por localização anatômica do tumor."""
    tissue_map = {
        "Sigmoid colon": "Cólon Sigmoide",
        "Ascending colon": "Cólon Ascendente",
        "Colon, NOS": "Cólon (NOS)",
        "Cecum": "Ceco",
        "Transverse colon": "Cólon Transverso",
        "Descending colon": "Cólon Descendente",
        "Rectosigmoid junction": "Junção Retossigmoide",
    }
    plot_df = df.copy()
    plot_df['localizacao'] = plot_df['tissue_or_organ_of_origin'].map(tissue_map)
    order = plot_df['localizacao'].value_counts().index

    plt.figure(figsize=(12, 7))
    ax = sns.countplot(y='localizacao', data=plot_df, order=order, palette="viridis")
    for container in ax.containers:
        ax.bar_label(container, label_type='edge', padding=3)

    plt.title('Distribuição por Localização Anatômica do Tumor')
    plt.xlabel('Contagem')
    plt.ylabel('Localização')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    save_plot(plt.gcf(), output_dir, "distribuicao_localizacao_anatomica.png")


def plot_mortality_by_age_group(df, output_dir):
    """Gera o gráfico de taxa de óbito por faixa etária."""
    age_labels = {
        "age_0_40": "0-40",
        "age_40_50": "40-50",
        "age_50_60": "50-60",
        "age_60_70": "60-70",
        "age_70_80": "70-80",
        "age_80_100": "80-100",
    }
    stats = []
    for grp in sorted(df['age_group'].unique()):
        subset = df[df['age_group'] == grp]
        n_total = len(subset)
        n_obitos = subset['event_occurred'].sum()
        stats.append({
            'Faixa Etária': age_labels.get(str(grp), str(grp)),
            'Total': n_total,
            'Óbitos': n_obitos,
            'Taxa de Óbito (%)': n_obitos / n_total * 100,
        })
    stats_df = pd.DataFrame(stats)

    fig, ax1 = plt.subplots(figsize=(10, 7))

    x = range(len(stats_df))
    bars = ax1.bar(x, stats_df['Total'], color='#4c72b0', alpha=0.7, label='Total de Pacientes')
    ax1.bar_label(bars, labels=[str(v) for v in stats_df['Total']], padding=3, fontsize=10)
    ax1.set_xlabel('Faixa Etária', fontsize=12)
    ax1.set_ylabel('Número de Pacientes', fontsize=12, color='#4c72b0')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stats_df['Faixa Etária'])
    ax1.tick_params(axis='y', labelcolor='#4c72b0')

    ax2 = ax1.twinx()
    line = ax2.plot(x, stats_df['Taxa de Óbito (%)'], color='#c44e52', marker='o',
                    linewidth=2.5, markersize=8, label='Taxa de Óbito (%)')
    for i, val in enumerate(stats_df['Taxa de Óbito (%)']):
        ax2.annotate(f'{val:.1f}%', (i, val), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=10, color='#c44e52', fontweight='bold')
    ax2.set_ylabel('Taxa de Óbito (%)', fontsize=12, color='#c44e52')
    ax2.tick_params(axis='y', labelcolor='#c44e52')
    ax2.set_ylim(0, max(stats_df['Taxa de Óbito (%)']) * 1.3)

    plt.title('Distribuição de Pacientes e Taxa de Óbito por Faixa Etária', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_plot(fig, output_dir, "taxa_obito_por_faixa_etaria.png")


def plot_vital_status_distribution(df, output_dir):
    """Gera o gráfico de distribuição do status vital (evento de óbito)."""
    counts = df['event_occurred'].value_counts().sort_index()
    labels = ['Vivo / Censurado', 'Falecido']
    values = [counts.get(0, 0), counts.get(1, 0)]
    total = sum(values)
    pcts = [v / total * 100 for v in values]
    colors = ['#2ecc71', '#e74c3c']

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values, color=colors, edgecolor='white', width=0.5)
    for bar, val, pct in zip(bars, values, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                f'{val}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Número de Pacientes', fontsize=12)
    ax.set_title(f'Distribuição do Status Vital (n={total})', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_plot(fig, output_dir, "distribuicao_status_vital.png")


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
        plot_tissue_origin_distribution(df, config.FIGURES_DIR)
        plot_mortality_by_age_group(df, config.FIGURES_DIR)
        plot_vital_status_distribution(df, config.FIGURES_DIR)
        plot_overall_survival(df, config.FIGURES_DIR)
        plot_survival_by_stage(df, config.FIGURES_DIR)

        logging.info(f"EDA concluída com sucesso. Figuras salvas em: {config.FIGURES_DIR}")

    except Exception as e:
        logging.error(f"Falha na EDA: {e}")
        raise

if __name__ == "__main__":
    main()
