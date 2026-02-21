"""
Interpretação do Modelo Random Survival Forest.

Gera análises de explicabilidade:
- Permutation Importance (robusta e confiável)
- Partial Dependence Plots (PDP)
- Individual Conditional Expectation (ICE)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

from src import config
from src.utils import save_plot
from src.modeling.train import load_and_split_data, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def explore_rsf():
    """Executa análise completa de explicabilidade do Random Survival Forest."""
    logging.info("--- Iniciando Exploração do Random Survival Forest ---")

    # 1. Carregar Dados
    logging.info(f"Carregando dados de {config.FEATURES_SURVIVAL_PATH}...")
    X, y, cat_features = load_and_split_data(config.FEATURES_SURVIVAL_PATH)

    # 2. Split Treino/Teste
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y["event"]
    )

    # 3. Encoding (Fit Treino, Transform Teste)
    X_train, X_test = encode_features(X_train_raw, X_test_raw, cat_features)

    # 4. Treinar o modelo com os mesmos parâmetros usados no pipeline principal
    logging.info("Treinando Random Survival Forest...")
    rsf_model = RandomSurvivalForest(
        random_state=42,
        min_samples_leaf=15,
        max_depth=10,
        n_estimators=200,
        n_jobs=-1
    )
    rsf_model.fit(X_train, y_train)

    # 5. Avaliar no Teste
    test_preds = rsf_model.predict(X_test)
    test_score = concordance_index_censored(y_test["event"], y_test["time"], test_preds)[0]
    logging.info(f"C-Index no Teste: {test_score:.4f}")

    # 6. Permutation Importance (Método mais robusta e confiável)
    logging.info("Calculando Permutation Importance (pode demorar alguns minutos)...")

    def rsf_scorer(model, X, y):
        """Scorer customizado para permutation importance."""
        preds = model.predict(X)
        return concordance_index_censored(y["event"], y["time"], preds)[0]

    perm_importance = permutation_importance(
        rsf_model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring=rsf_scorer,
        n_jobs=-1
    )

    feature_importance_perm = pd.DataFrame({
        'feature': X_train.columns,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(
        feature_importance_perm['feature'],
        feature_importance_perm['importance_mean'],
        xerr=feature_importance_perm['importance_std'],
        color=config.PALETTE['primary'],
        alpha=0.8
    )
    ax.set_xlabel("Importância (Permutation Importance)", fontsize=12)
    ax.set_ylabel("Variáveis", fontsize=12)
    ax.set_title(
        f"Importância por Permutação - Random Survival Forest\nC-Index: {test_score:.3f}",
        fontsize=16,
        fontweight='bold'
    )
    ax.invert_yaxis()
    plt.tight_layout()
    save_plot(fig, config.FIGURES_DIR, "importancia_permutacao_rsf.png")
    logging.info("Gráfico de Permutation Importance salvo!")

    # 7. Partial Dependence Plots (PDP)
    logging.info("Gerando Gráficos de Dependência Parcial (PDP)...")
    try:
        # Selecionar as top 3 variáveis baseadas na permutation importance
        top_3_features = feature_importance_perm['feature'].head(3).tolist()
        logging.info(f"Variáveis selecionadas para PDP: {top_3_features}")

        # Criar wrapper compatível com sklearn
        class RSFWrapper:
            """Wrapper para tornar RSF compatível com PartialDependenceDisplay."""
            def __init__(self, model):
                self.model = model
                self._estimator_type = "regressor"
                self.classes_ = []

            def fit(self, X, y):
                return self

            def predict(self, X):
                return self.model.predict(X)

        wrapped_model = RSFWrapper(rsf_model)

        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        display = PartialDependenceDisplay.from_estimator(
            wrapped_model,
            X_train,
            features=top_3_features,
            kind="both",  # PDP + ICE
            subsample=100,
            ax=ax,
            percentiles=(0.05, 0.95),
            random_state=42,
            pd_line_kw={'color': config.PALETTE['primary'], 'linewidth': 2.5},
            ice_lines_kw={'color': config.PALETTE['primary'], 'alpha': 0.08, 'linewidth': 0.5},
        )

        plt.suptitle(
            "Impacto das Variáveis no Risco - Random Survival Forest (PDP/ICE)",
            fontsize=16,
            fontweight='bold'
        )
        for a in ax.flatten():
            a.set_ylabel("Risco Predito (Maior = Pior Prognóstico)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_plot(fig, config.FIGURES_DIR, "dependencia_parcial_rsf.png")
        logging.info("Gráficos de PDP gerados com sucesso!")

    except Exception as e:
        logging.error(f"Erro ao gerar gráficos de PDP: {e}")

    # 8. Resumo das Top Features
    logging.info("\n" + "="*60)
    logging.info("TOP 10 VARIÁVEIS MAIS IMPORTANTES (Permutation Importance):")
    logging.info("="*60)
    for idx, row in feature_importance_perm.head(10).iterrows():
        logging.info(f"{row['feature']:40s} {row['importance_mean']:.6f} ± {row['importance_std']:.6f}")

    logging.info("\n--- Exploração do RSF concluída! Confira os gráficos em reports/figures/ ---")


def main():
    explore_rsf()

if __name__ == "__main__":
    main()
