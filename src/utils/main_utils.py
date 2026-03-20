"""Shared I/O and data-wrangling utilities for the ML pipeline."""
import os
import sys

import dill
import numpy as np
import pandas as pd
import yaml

from src.exception import MyException
from src.logger import logging


def read_yaml_file(file_path: str) -> dict:
    """Load a YAML file and return its contents as a dict."""
    try:
        logging.info(f"Reading YAML: {file_path}")
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """Write a Python object to a YAML file."""
    try:
        logging.info(f"Writing YAML: {file_path}")
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> object:
    """Deserialise a dill-pickled object from disk."""
    try:
        logging.info(f"Loading object: {file_path}")
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """Serialise an object to disk with dill."""
    try:
        logging.info(f"Saving object: {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info(f"Object saved: {file_path}")
    except Exception as e:
        raise MyException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.ndarray) -> None:
    """Save a numpy array to a binary .npy file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.ndarray:
    """Load a numpy array from a binary .npy file."""
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def read_csv(file_path: str) -> pd.DataFrame:
    """Read a CSV into a DataFrame. Shared across components."""
    try:
        logging.info(f"Reading CSV: {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        raise MyException(e, sys) from e


def resolve_target_column(df: pd.DataFrame, configured: str) -> str:
    """Return the first matching target column name or raise ValueError."""
    for col in (configured, "life_ratio"):
        if col and col in df.columns:
            logging.info(f"Resolved target column: {col}")
            return col
    raise ValueError(
        f"Target column not found. Configured: '{configured}'. "
        f"Available: {list(df.columns)}"
    )


def split_features_target(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into feature matrix X and target vector y."""
    return df.drop(columns=[target_col]), df[target_col]