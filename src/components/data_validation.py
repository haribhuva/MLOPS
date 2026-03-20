"""Validates ingested train/test data against the expected schema."""
import sys

import pandas as pd

from src.constants import SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import MyException
from src.logger import logging
from src.utils.main_utils import read_yaml_file, write_yaml_file, read_csv


class DataValidation:
    """Checks column count and column presence against schema.yaml."""

    _CFG_COLUMNS = "columns"

    def __init__(
        self,
        data_validation_config: DataValidationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ) -> None:
        try:
            self._config = data_validation_config
            self._ingestion_artifact = data_ingestion_artifact
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("DataValidation initialized.")
        except Exception as e:
            raise MyException(e, sys) from e

    def _get_schema_column_names(self) -> list[str]:
        """Extract column names from schema (each entry is {col_name: dtype})."""
        return [list(col.keys())[0] for col in self._schema_config[self._CFG_COLUMNS]]

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """Check whether the dataframe has the same number of columns as the schema."""
        try:
            status = len(dataframe.columns) == len(self._schema_config[self._CFG_COLUMNS])
            logging.info(f"Column count validation — passed: {status}")
            return status
        except Exception as e:
            raise MyException(e, sys) from e

    def is_columns_exist(self, df: pd.DataFrame) -> bool:
        """Check whether all schema-defined columns are present in the dataframe."""
        try:
            schema_column_names = self._get_schema_column_names()
            missing_columns = [
                col for col in schema_column_names if col not in df.columns
            ]
            if missing_columns:
                logging.info(f"Missing columns: {missing_columns}")
                return False
            return True
        except Exception as e:
            raise MyException(e, sys) from e

    def initiate_data_validation(self) -> DataValidationArtifact:
        """Run all validation checks, save report, and return artifact."""
        try:
            logging.info("DataValidation started.")
            validation_error_message = ""

            train_df = read_csv(self._ingestion_artifact.trained_data_path)
            test_df = read_csv(self._ingestion_artifact.test_data_path)

            if not self.validate_number_of_columns(dataframe=train_df):
                validation_error_message += (
                    f"Train column count mismatch. "
                    f"Expected: {len(self._schema_config[self._CFG_COLUMNS])}, "
                    f"Found: {len(train_df.columns)}. "
                )

            if not self.validate_number_of_columns(dataframe=test_df):
                validation_error_message += (
                    f"Test column count mismatch. "
                    f"Expected: {len(self._schema_config[self._CFG_COLUMNS])}, "
                    f"Found: {len(test_df.columns)}. "
                )

            if not self.is_columns_exist(df=train_df):
                validation_error_message += "Train dataframe is missing required columns. "

            if not self.is_columns_exist(df=test_df):
                validation_error_message += "Test dataframe is missing required columns. "

            validation_status = len(validation_error_message) == 0

            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_message if not validation_status else "All validations passed.",
            }
            write_yaml_file(
                file_path=self._config.validation_report_file_path,
                content=validation_report,
            )
            logging.info(f"Validation report saved: {self._config.validation_report_file_path}")

            artifact = DataValidationArtifact(
                validation_status=validation_status,
                message=validation_error_message,
                validation_report_file_path=self._config.validation_report_file_path,
            )
            logging.info("DataValidation completed.")
            return artifact

        except Exception as e:
            logging.error("DataValidation failed.")
            raise MyException(e, sys) from e