"""
Funções utilitárias para o projeto.
"""

import logging
import os

import joblib
import pandas as pd
from matplotlib import pyplot as plt


def load_data(file_path: str) -> pd.DataFrame:
    """Carrega dados de um arquivo.

    Args:
        file_path (str): O caminho para o arquivo.

    Returns:
        pd.DataFrame: Os dados carregados.
    """
    try:
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith('.tsv'):
            return pd.read_csv(file_path, sep='\t')
        else:
            raise ValueError("Formato de arquivo não suportado")
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {file_path}")
        raise


def save_data(data: pd.DataFrame, file_path: str):
    """Salva os dados em um arquivo.

    Args:
        data (pd.DataFrame): Os dados a serem salvos.
        file_path (str): O caminho para o arquivo.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    if file_path.endswith('.parquet'):
        data.to_parquet(file_path, index=False)
    elif file_path.endswith('.csv'):
        data.to_csv(file_path, index=False)
    else:
        raise ValueError("Formato de arquivo não suportado")


def load_model(file_path: str):
    """Carrega um modelo de um arquivo.

    Args:
        file_path (str): O caminho para o arquivo.

    Returns:
        O modelo carregado.
    """
    try:
        return joblib.load(file_path)
    except FileNotFoundError:
        logging.error(f"Modelo não encontrado: {file_path}")
        raise


def save_model(model, file_path: str):
    """Salva um modelo em um arquivo.

    Args:
        model: O modelo a ser salvo.
        file_path (str): O caminho para o arquivo.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)


def save_plot(fig: plt.Figure, output_dir: str, filename: str, dpi: int = 300):
    """Salva uma figura Matplotlib em um arquivo.

    Args:
        fig (plt.Figure): A figura a ser salva.
        output_dir (str): O diretório para salvar o gráfico.
        filename (str): O nome do arquivo.
        dpi (int): A resolução do gráfico salvo.
    """
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    try:
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
        logging.info(f"Gráfico salvo em: {file_path}")
    except Exception as e:
        logging.error(f"Erro ao salvar o gráfico em {file_path}: {e}")
    finally:
        plt.close(fig)