import os
import sys

import pandas as pd
from pandas import DataFrame

from src.logger import logging
from src.exception import MyException
from src.utils.main_utils import read_yaml_file, write_yaml_file
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationconfig
from src.constants import SCHEMA_FILE_PATH


class DataValidation:
    """Validates ingested train/test data against the expected schema."""

    def __init__(self, data_validation_config: DataValidationconfig, data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            # Load the expected schema from the YAML config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e, sys)

    def validate_number_of_columns(self, dataframe: DataFrame) -> bool:
        """Check whether the dataframe has the same number of columns as the schema."""
        try:
            status = len(dataframe.columns) == len(self._schema_config["columns"])
            logging.info(f"Column count validation — passed: {status}")
            return status
        except Exception as e:
            raise MyException(e, sys)

    def _get_schema_column_names(self) -> list:
        """Extract column names from schema (each entry is a dict like {col_name: dtype})."""
        return [list(col.keys())[0] for col in self._schema_config["columns"]]

    def is_columns_exist(self, df: DataFrame) -> bool:
        """Check whether all schema-defined columns are present in the dataframe."""
        try:
            dataframe_columns = df.columns
            schema_column_names = self._get_schema_column_names()
            missing_columns = [
                col for col in schema_column_names if col not in dataframe_columns
            ]
            if missing_columns:
                logging.info(f"Missing columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            raise MyException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> DataFrame:
        """Read a CSV file and return it as a DataFrame."""
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Run all validation checks on train and test data, save report, and return artifact."""
        try:
            validation_error_message = ""
            logging.info("Starting data validation process...")

            # Read the ingested train and test datasets
            train_df = DataValidation.read_data(self.data_ingestion_artifact.trained_data_path)
            test_df = DataValidation.read_data(self.data_ingestion_artifact.test_data_path)

            # --- Column count validation ---
            if not self.validate_number_of_columns(dataframe=train_df):
                validation_error_message += (
                    f"Train dataframe column count mismatch. "
                    f"Expected: {len(self._schema_config['columns'])}, Found: {len(train_df.columns)}. "
                )
            else:
                logging.info("Train dataframe has the required number of columns.")

            if not self.validate_number_of_columns(dataframe=test_df):
                validation_error_message += (
                    f"Test dataframe column count mismatch. "
                    f"Expected: {len(self._schema_config['columns'])}, Found: {len(test_df.columns)}. "
                )
            else:
                logging.info("Test dataframe has the required number of columns.")

            # --- Required columns presence validation ---
            if not self.is_columns_exist(df=train_df):
                validation_error_message += "Train dataframe is missing required columns. "
            else:
                logging.info("Train dataframe has all required columns.")

            if not self.is_columns_exist(df=test_df):
                validation_error_message += "Test dataframe is missing required columns. "
            else:
                logging.info("Test dataframe has all required columns.")

            # Determine overall validation status
            validation_status = len(validation_error_message) == 0

            # Build the validation report dict and save as YAML
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_message if not validation_status else "All validations passed.",
            }
            write_yaml_file(
                file_path=self.data_validation_config.validation_report_file_path,
                content=validation_report,
            )
            logging.info(f"Validation report saved at: {self.data_validation_config.validation_report_file_path}")

            # Create and return the data validation artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message,
                validation_report_file_path=self.data_validation_config.validation_report_file_path,
            )
            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise MyException(e, sys) from e