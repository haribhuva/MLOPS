"""Cleans, engineers features, and saves transformed train/test artifacts."""
import os
import sys

import pandas as pd

from src.constants import SCHEMA_FILE_PATH
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_csv, read_yaml_file


class DataTransformation:
    """Applies cleaning and feature engineering to raw ingested data."""

    _CFG_RENAME_COLS = "rename_columns"
    _CFG_DROP_COLS = "drop_columns"
    _CFG_TARGET = "target_column"

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        try:
            self._config = data_transformation_config
            self._validation_artifact = data_validation_artifact
            self._ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            self._model_config = read_yaml_file(self._config.transformed_object_file_path.replace(
                os.path.basename(self._config.transformed_object_file_path), ""
            ).rstrip(os.sep).rstrip(os.sep) + "")
            logging.info("DataTransformation initialized.")
        except Exception:
            # Gracefully handle missing model config at init — it's optional for this stage
            self._model_config = {}
            logging.info("DataTransformation initialized (no model config loaded).")

    def _clean(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Drop unwanted columns, rename, remove dead features."""
        try:
            logging.info("Cleaning started.")

            rename_columns = self._schema_config.get(self._CFG_RENAME_COLS) or {}
            if rename_columns:
                train_df = train_df.rename(columns=rename_columns)
                test_df = test_df.rename(columns=rename_columns)
                logging.info(f"Renamed columns: {list(rename_columns.values())}")

            # life_ratio = cycle / max_cycle_per_unit — normalised RUL proxy
            train_df["life_ratio"] = (
                train_df["cycle"] / train_df.groupby("unit_id")["cycle"].transform("max")
            )
            test_df["life_ratio"] = (
                test_df["cycle"] / test_df.groupby("unit_id")["cycle"].transform("max")
            )
            logging.info("Added life_ratio feature.")

            drop_columns = self._schema_config.get(self._CFG_DROP_COLS) or []
            drop_columns = [rename_columns.get(c, c) for c in drop_columns]
            train_df = train_df.drop(columns=[c for c in drop_columns if c in train_df.columns])
            test_df = test_df.drop(columns=[c for c in drop_columns if c in test_df.columns])
            logging.info(f"Dropped schema columns: {drop_columns}")

            # Drop dead columns (single unique value — carry zero information)
            target_col = "life_ratio"
            dead_columns = [
                c for c in train_df.columns
                if train_df[c].nunique() <= 1 and c != target_col
            ]
            train_df = train_df.drop(columns=dead_columns)
            test_df = test_df.drop(columns=dead_columns)
            logging.info(f"Dropped dead columns: {dead_columns}")

            before = len(train_df)
            train_df = train_df.dropna()
            rows_dropped = before - len(train_df)
            if rows_dropped > 0:
                logging.warning(f"Dropped {rows_dropped} rows with missing values.")

            logging.info("Cleaning completed.")
            return train_df, test_df

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        """Public entry point — validate, clean, and save transformed data."""
        try:
            logging.info("DataTransformation started.")

            if not self._validation_artifact.validation_status:
                raise ValueError(self._validation_artifact.message)

            train_df = read_csv(self._ingestion_artifact.trained_data_path)
            test_df = read_csv(self._ingestion_artifact.test_data_path)

            train_df, test_df = self._clean(train_df, test_df)

            os.makedirs(
                os.path.dirname(self._config.transformed_train_file_path),
                exist_ok=True,
            )
            train_df.to_csv(self._config.transformed_train_file_path, index=False)
            test_df.to_csv(self._config.transformed_test_file_path, index=False)
            logging.info("Saved transformed train and test data as CSV.")

            artifact = DataTransformationArtifact(
                transformed_train_file_path=self._config.transformed_train_file_path,
                transformed_test_file_path=self._config.transformed_test_file_path,
                transformed_object_file_path=self._config.transformed_object_file_path,
            )
            logging.info("DataTransformation completed.")
            return artifact

        except Exception as e:
            logging.error("DataTransformation failed.")
            raise MyException(e, sys) from e