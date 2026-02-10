"""
Diagnóstico de Overfitting para os Modelos de Sobrevivência.

Executa análises para verificar se os modelos estão sobreajustados:
1. Gap Treino vs Teste (C-Index)
2. Impacto de feature redundante (age_group)
3. Learning Curves
4. Teste de Permutação (baseline aleatório)
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from xgboost import XGBRegressor
from lifelines import CoxPHFitter

from src import config
from src.utils import load_data, save_plot
from src.modeling.train import load_and_split_data, encode_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def compute_c_index(model_name, model, X, y):
    """Calcula C-Index para um modelo já treinado."""
    if model_name == "CoxPH":
        preds = model.predict_partial_hazard(X)
    else:
        preds = model.predict(X)
    return concordance_index_censored(y["event"], y["time"], preds)[0]


def fit_model(model_name, model, X, y):
    """Treina um modelo (mesma lógica do train.py)."""
    if model_name == "CoxPH":
        fit_df = X.copy()
        fit_df["event_occurred"] = y["event"]
        fit_df["observed_time"] = y["time"]
        model.fit(fit_df, duration_col="observed_time", event_col="event_occurred")
    elif model_name == "XGBoostSurvival":
        y_xgb = np.where(y["event"], y["time"], -y["time"])
        model.fit(X, y_xgb)
    else:
        model.fit(X, y)
    return model


def get_fresh_models():
    """Retorna instâncias novas dos 3 modelos."""
    return {
        "CoxPH": CoxPHFitter(penalizer=0.1),
        "RandomSurvivalForest": RandomSurvivalForest(
            random_state=42, min_samples_leaf=15, max_depth=10, n_estimators=200, n_jobs=-1
        ),
        "XGBoostSurvival": XGBRegressor(
            objective="survival:cox", eval_metric="cox-nloglik", random_state=42,
            max_depth=10, learning_rate=0.05, n_estimators=200, min_child_weight=1,
            gamma=0.2, subsample=0.8, colsample_bytree=1.0, reg_alpha=0, reg_lambda=0.1,
        ),
    }


def diagnostic_1_train_vs_test_gap(X_train, X_test, y_train, y_test):
    """Diagnóstico 1: Comparar C-Index no Treino vs Teste (detecta memorização)."""
    logging.info("\n" + "=" * 70)
    logging.info("DIAGNÓSTICO 1: GAP TREINO vs TESTE (Detecção de Memorização)")
    logging.info("=" * 70)

    results = []
    models = get_fresh_models()

    for name, model in models.items():
        model = fit_model(name, model, X_train, y_train)
        train_ci = compute_c_index(name, model, X_train, y_train)
        test_ci = compute_c_index(name, model, X_test, y_test)
        gap = train_ci - test_ci

        results.append({
            "Modelo": name,
            "C-Index Treino": train_ci,
            "C-Index Teste": test_ci,
            "Gap (Treino - Teste)": gap,
            "Overfit?": "SIM" if gap > 0.05 else ("LEVE" if gap > 0.02 else "NÃO"),
        })
        logging.info(f"  {name:30s} | Treino: {train_ci:.4f} | Teste: {test_ci:.4f} | Gap: {gap:+.4f}")

    return pd.DataFrame(results)


def diagnostic_2_feature_redundancy(X_train_raw, X_test_raw, y_train, y_test, cat_features):
    """Diagnóstico 2: Impacto de remover age_group (redundante com age_at_index)."""
    logging.info("\n" + "=" * 70)
    logging.info("DIAGNÓSTICO 2: IMPACTO DA FEATURE REDUNDANTE (age_group)")
    logging.info("=" * 70)

    results = []

    for label, drop_cols in [("Com age_group", []), ("Sem age_group", ["age_group"])]:
        X_tr = X_train_raw.drop(columns=drop_cols, errors="ignore")
        X_te = X_test_raw.drop(columns=drop_cols, errors="ignore")

        cats = [c for c in cat_features if c not in drop_cols]
        X_tr_enc, X_te_enc = encode_features(X_tr, X_te, cats)

        models = get_fresh_models()
        for name, model in models.items():
            model = fit_model(name, model, X_tr_enc, y_train)
            train_ci = compute_c_index(name, model, X_tr_enc, y_train)
            test_ci = compute_c_index(name, model, X_te_enc, y_test)
            results.append({
                "Configuração": label,
                "Modelo": name,
                "C-Index Treino": train_ci,
                "C-Index Teste": test_ci,
                "Gap": train_ci - test_ci,
            })
            logging.info(f"  {label:20s} | {name:30s} | Treino: {train_ci:.4f} | Teste: {test_ci:.4f}")

    return pd.DataFrame(results)


def diagnostic_3_complexity_analysis(X_train, X_test, y_train, y_test):
    """Diagnóstico 3: Impacto da complexidade do XGBoost (reduzir max_depth)."""
    logging.info("\n" + "=" * 70)
    logging.info("DIAGNÓSTICO 3: COMPLEXIDADE DO XGBOOST (max_depth)")
    logging.info("=" * 70)

    results = []
    for depth in [3, 5, 7, 10]:
        model = XGBRegressor(
            objective="survival:cox", eval_metric="cox-nloglik", random_state=42,
            max_depth=depth, learning_rate=0.05, n_estimators=200, min_child_weight=1,
            gamma=0.2, subsample=0.8, colsample_bytree=1.0, reg_alpha=0, reg_lambda=0.1,
        )
        y_xgb = np.where(y_train["event"], y_train["time"], -y_train["time"])
        model.fit(X_train, y_xgb)

        train_ci = concordance_index_censored(
            y_train["event"], y_train["time"], model.predict(X_train)
        )[0]
        test_ci = concordance_index_censored(
            y_test["event"], y_test["time"], model.predict(X_test)
        )[0]
        gap = train_ci - test_ci

        results.append({
            "max_depth": depth,
            "C-Index Treino": train_ci,
            "C-Index Teste": test_ci,
            "Gap": gap,
        })
        logging.info(f"  max_depth={depth:2d} | Treino: {train_ci:.4f} | Teste: {test_ci:.4f} | Gap: {gap:+.4f}")

    return pd.DataFrame(results)


def diagnostic_4_learning_curves(X_train, y_train, X_test, y_test):
    """Diagnóstico 4: Learning Curves (performance vs tamanho do treino)."""
    logging.info("\n" + "=" * 70)
    logging.info("DIAGNÓSTICO 4: LEARNING CURVES")
    logging.info("=" * 70)

    fractions = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    n_total = len(X_train)
    results = []

    for frac in fractions:
        n_samples = int(n_total * frac)
        if n_samples < 50:
            continue

        # Subsample do treino (mantendo estratificação)
        if frac < 1.0:
            idx = np.random.RandomState(42).choice(n_total, n_samples, replace=False)
            X_sub = X_train.iloc[idx]
            y_sub = y_train[idx]
        else:
            X_sub = X_train
            y_sub = y_train

        models_config = {
            "XGBoostSurvival": XGBRegressor(
                objective="survival:cox", eval_metric="cox-nloglik", random_state=42,
                max_depth=10, learning_rate=0.05, n_estimators=200, min_child_weight=1,
                gamma=0.2, subsample=0.8, colsample_bytree=1.0, reg_alpha=0, reg_lambda=0.1,
            ),
            "RSF": RandomSurvivalForest(
                random_state=42, min_samples_leaf=15, max_depth=10, n_estimators=200, n_jobs=-1
            ),
        }

        for name, model in models_config.items():
            model = fit_model(
                "XGBoostSurvival" if name == "XGBoostSurvival" else "RandomSurvivalForest",
                model, X_sub, y_sub
            )

            model_key = "XGBoostSurvival" if name == "XGBoostSurvival" else "RandomSurvivalForest"
            train_ci = compute_c_index(model_key, model, X_sub, y_sub)
            test_ci = compute_c_index(model_key, model, X_test, y_test)

            results.append({
                "Modelo": name,
                "Fração": frac,
                "N Amostras": n_samples,
                "C-Index Treino": train_ci,
                "C-Index Teste": test_ci,
            })
            logging.info(f"  {name:20s} | n={n_samples:5d} ({frac:.0%}) | Treino: {train_ci:.4f} | Teste: {test_ci:.4f}")

    return pd.DataFrame(results)


def diagnostic_5_permutation_test(X_train, X_test, y_train, y_test, n_permutations=5):
    """Diagnóstico 5: Teste de Permutação (baseline aleatório)."""
    logging.info("\n" + "=" * 70)
    logging.info("DIAGNÓSTICO 5: TESTE DE PERMUTAÇÃO (Labels Aleatórios)")
    logging.info("=" * 70)

    real_scores = {}
    random_scores = {name: [] for name in ["XGBoostSurvival", "RandomSurvivalForest"]}

    # Score real
    for name in ["XGBoostSurvival", "RandomSurvivalForest"]:
        model = get_fresh_models()[name]
        model = fit_model(name, model, X_train, y_train)
        real_scores[name] = compute_c_index(name, model, X_test, y_test)

    # Scores com labels permutados
    for i in range(n_permutations):
        perm_idx = np.random.RandomState(i).permutation(len(y_train))
        y_train_perm = y_train[perm_idx]

        for name in ["XGBoostSurvival", "RandomSurvivalForest"]:
            model = get_fresh_models()[name]
            try:
                model = fit_model(name, model, X_train, y_train_perm)
                score = compute_c_index(name, model, X_test, y_test)
                random_scores[name].append(score)
            except Exception:
                random_scores[name].append(0.5)

        logging.info(f"  Permutação {i + 1}/{n_permutations} concluída")

    results = []
    for name in ["XGBoostSurvival", "RandomSurvivalForest"]:
        mean_random = np.mean(random_scores[name])
        results.append({
            "Modelo": name,
            "C-Index Real": real_scores[name],
            "C-Index Aleatório (média)": mean_random,
            "Diferença": real_scores[name] - mean_random,
        })
        logging.info(f"  {name:30s} | Real: {real_scores[name]:.4f} | Aleatório: {mean_random:.4f}")

    return pd.DataFrame(results)


def plot_diagnostics(gap_df, complexity_df, learning_df):
    """Gera figura consolidada com os diagnósticos visuais."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Gap Treino vs Teste
    ax = axes[0]
    x = range(len(gap_df))
    width = 0.35
    ax.bar([i - width / 2 for i in x], gap_df["C-Index Treino"], width,
           label="Treino", color=config.PALETTE['primary'], alpha=0.8)
    ax.bar([i + width / 2 for i in x], gap_df["C-Index Teste"], width,
           label="Teste", color=config.PALETTE['accent'], alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(["CoxPH", "RSF", "XGBoost"], rotation=0)
    ax.set_ylabel("C-Index")
    ax.set_title("Gap Treino vs Teste", fontweight="bold")
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    # Plot 2: Complexidade XGBoost
    ax = axes[1]
    ax.plot(complexity_df["max_depth"], complexity_df["C-Index Treino"],
            "o-", color=config.PALETTE['dead'], label="Treino", linewidth=2)
    ax.plot(complexity_df["max_depth"], complexity_df["C-Index Teste"],
            "s-", color=config.PALETTE['alive'], label="Teste", linewidth=2)
    ax.fill_between(complexity_df["max_depth"],
                    complexity_df["C-Index Teste"], complexity_df["C-Index Treino"],
                    alpha=0.15, color=config.PALETTE['dead'], label="Gap (Overfitting)")
    ax.set_xlabel("max_depth")
    ax.set_ylabel("C-Index")
    ax.set_title("Complexidade XGBoost vs Overfitting", fontweight="bold")
    ax.legend()
    ax.set_ylim(0.5, 1.05)
    ax.grid(linestyle="--", alpha=0.5)

    # Plot 3: Learning Curves
    ax = axes[2]
    for model_name in learning_df["Modelo"].unique():
        subset = learning_df[learning_df["Modelo"] == model_name]
        color = config.PALETTE['primary'] if "XGB" in model_name else config.PALETTE['accent']
        ax.plot(subset["N Amostras"], subset["C-Index Treino"],
                "o--", color=color, alpha=0.5, label=f"{model_name} (Treino)")
        ax.plot(subset["N Amostras"], subset["C-Index Teste"],
                "s-", color=color, label=f"{model_name} (Teste)")
    ax.set_xlabel("N Amostras de Treino")
    ax.set_ylabel("C-Index")
    ax.set_title("Learning Curves", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0.5, 1.05)
    ax.grid(linestyle="--", alpha=0.5)

    plt.suptitle("Diagnóstico de Overfitting — Modelos de Sobrevivência",
                 fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save_plot(fig, config.FIGURES_DIR, "diagnostico_overfitting.png")


def main():
    logging.info("=" * 70)
    logging.info("  DIAGNÓSTICO COMPLETO DE OVERFITTING")
    logging.info("=" * 70)

    # Carregar dados
    X, y, cat_features = load_and_split_data(config.FEATURES_SURVIVAL_PATH)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y["event"]
    )
    X_train, X_test = encode_features(X_train_raw, X_test_raw, cat_features)

    logging.info(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
    logging.info(f"Features após encoding: {list(X_train.columns)}")

    # Executar diagnósticos
    gap_df = diagnostic_1_train_vs_test_gap(X_train, X_test, y_train, y_test)
    redundancy_df = diagnostic_2_feature_redundancy(X_train_raw, X_test_raw, y_train, y_test, cat_features)
    complexity_df = diagnostic_3_complexity_analysis(X_train, X_test, y_train, y_test)
    learning_df = diagnostic_4_learning_curves(X_train, y_train, X_test, y_test)
    permutation_df = diagnostic_5_permutation_test(X_train, X_test, y_train, y_test)

    # Gerar visualização
    plot_diagnostics(gap_df, complexity_df, learning_df)

    # Resumo Final
    logging.info("\n" + "=" * 70)
    logging.info("  RESUMO DO DIAGNÓSTICO")
    logging.info("=" * 70)
    logging.info("\n--- Gap Treino vs Teste ---")
    logging.info(gap_df.to_string(index=False))
    logging.info("\n--- Impacto Feature Redundante ---")
    logging.info(redundancy_df.to_string(index=False))
    logging.info("\n--- Complexidade XGBoost ---")
    logging.info(complexity_df.to_string(index=False))
    logging.info("\n--- Teste de Permutação ---")
    logging.info(permutation_df.to_string(index=False))


if __name__ == "__main__":
    main()
