import os
import sys

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, save_object
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact, DataIngestionArtifact
from src.constants import TARGET_COLUMN, SCHEMA_FILE_PATH


class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact   = data_validation_artifact
            self.data_ingestion_artifact    = data_ingestion_artifact
            self._schema_config             = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys) from e

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop unwanted columns, rename, remove dead features."""
        try:
            # 1. Rename
            rename_columns = self._schema_config.get("rename_columns") or {}
            if rename_columns:
                df = df.rename(columns=rename_columns)
                logging.info(f"Renamed columns: {rename_columns}")

            # 2. Drop explicitly listed columns
            drop_columns = self._schema_config.get("drop_columns") or []
            drop_columns = [rename_columns.get(c, c) for c in drop_columns]
            df = df.drop(columns=[c for c in drop_columns if c in df.columns])
            logging.info(f"Dropped columns: {drop_columns}")

            # 3. Drop dead columns (single unique value — carry zero information)
            dead_columns = [c for c in df.columns if df[c].nunique() <= 1 and c != TARGET_COLUMN]
            df = df.drop(columns=dead_columns)
            logging.info(f"Dropped dead columns: {dead_columns}")

            # 4. Drop missing values
            before = len(df)
            df = df.dropna()
            logging.info(f"Dropped {before - len(df)} rows with missing values.")

            return df

        except Exception as e:
            raise MyException(e, sys) from e

    def _scale(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Scale numerical features. Fit on train, apply to both."""
        try:
            pass_through_columns = self._schema_config.get("pass_through_columns") or []
            rename_columns       = self._schema_config.get("rename_columns") or {}
            pass_through_columns = [rename_columns.get(c, c) for c in pass_through_columns]

            # Columns to scale = everything except target and pass_through
            scale_columns = [
                c for c in train_df.columns
                if c not in pass_through_columns and c != TARGET_COLUMN
            ]

            scaler = MinMaxScaler()
            train_df[scale_columns] = scaler.fit_transform(train_df[scale_columns])
            test_df[scale_columns]  = scaler.transform(test_df[scale_columns])
            logging.info(f"Scaled columns: {scale_columns}")

            # Save scaler for inference
            save_object(self.data_transformation_config.transformed_object_file_path, scaler)
            logging.info("Saved scaler object.")

            return train_df, test_df

        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Data Transformation Started.")
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            train_df = self.read_data(self.data_ingestion_artifact.trained_data_path)
            test_df  = self.read_data(self.data_ingestion_artifact.test_data_path)
            logging.info("Read train and test data.")

            # Clean
            train_df = self._clean(train_df)
            test_df  = self._clean(test_df)

            # Scale
            train_df, test_df = self._scale(train_df, test_df)

            # Save as CSV — human readable, directly usable
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_transformation_config.transformed_train_file_path, index=False)
            test_df.to_csv(self.data_transformation_config.transformed_test_file_path,   index=False)
            logging.info("Saved transformed train and test data as CSV.")

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

            logging.info("Data Transformation Completed.")
            return data_transformation_artifact

        except Exception as e:
            logging.info("Data transformation failed.")
            raise MyException(e, sys) from e