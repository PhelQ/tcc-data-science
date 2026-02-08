# CLAUDE.md

## Project Overview

Survival analysis pipeline for colorectal cancer (TCGA-COAD). Python data science project that predicts patient survival and stratifies risk groups using clinical/demographic data from The Cancer Genome Atlas.

**Language:** Python 3.10+

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python src/main.py

# Lint
flake8 src/

# Format (Black is installed but no config file — uses defaults)
black src/

# Launch notebook
jupyter notebook notebooks/visualizacao.ipynb
```

There is no test suite (no pytest/unittest). Validation is done via logging output and notebook exploration.

## Code Style

- **Formatter:** Black (line length 88)
- **Linter:** flake8 with Black-compatible settings (`max-line-length = 88`, `extend-ignore = E203`)
- **Naming:** Portuguese for file names, variables, and comments (e.g., `preprocessamento_data.py`, `treino_modelo_sobrevivencia.py`)
- **Functions/variables:** snake_case
- **Logging:** Every module uses `logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')`

## Architecture

```
src/
├── config.py                 # Centralized paths, column mappings, feature lists
├── utils.py                  # I/O helpers: load/save data (parquet/CSV/TSV), models (joblib), plots
├── main.py                   # Pipeline orchestrator — runs all stages in order
├── data/                     # Data processing: TSV→Parquet, merge clinical+biospecimen, feature engineering
├── aed/                      # Exploratory data analysis (EDA)
├── modeling/                 # Train CoxPH, RSF, XGBoost; predict survival; interpret models
└── visualization/            # Kaplan-Meier curves by risk group
```

**Pipeline stages (executed in order by `main.py`):**
1. `preprocessamento_data` — Raw TSV to Parquet
2. `consolidacaodados_tcga_coad` — Merge clinical + biospecimen data
3. `feature_engineering_survival` — Create survival features
4. `aed_tcga_coad` — Generate EDA plots
5. `treino_modelo_sobrevivencia` — Train 3 survival models
6. `interpret_modelos_sobrevivencia` — Analyze hazard ratios
7. `predicao_tempodevida` — Predict survival times
8. `visualize_survival_curves` — Plot Kaplan-Meier curves

## Key Patterns

- Each module exposes a `main()` function as its entry point
- No `__init__.py` files — imports use `from src.data import ...` style (requires PYTHONPATH or running from project root)
- All file paths are centralized in `src/config.py` relative to `ROOT_DIR`
- Data flows: raw (TSV) → interim (Parquet) → processed (engineered features)
- One-Hot Encoding is fitted on the training set only (prevents data leakage)
- Models are serialized with joblib to `models/` directory
- Plots saved to `reports/figures/` at 300 DPI

## Data & Models (git-ignored)

- `data/raw/`, `data/interim/`, `data/processed/` — not in repo
- `models/*.joblib` — not in repo
- `reports/*.csv` — not in repo

Raw data comes from TCGA (The Cancer Genome Atlas) and must be downloaded separately.
