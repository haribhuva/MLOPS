"""Fetches raw train/test data from PostgreSQL and saves ingestion artifacts."""
import os
import sys

import pandas as pd

from src.configuration.postgresql_connection import postgresql_client
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import MyException
from src.logger import logging


class DataIngestion:
    """Fetches raw data from the database and persists it as CSV artifacts."""

    def __init__(self, data_ingestion_config: DataIngestionConfig) -> None:
        try:
            self._config = data_ingestion_config
            logging.info("DataIngestion initialized.")
        except Exception as e:
            raise MyException(e, sys) from e

    def _fetch_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Query PostgreSQL for train and test tables."""
        logging.info("Fetching data from PostgreSQL.")
        conn = postgresql_client()
        try:
            train_df = pd.read_sql_query(
                f"SELECT * FROM {self._config.train_collection_name};", conn
            )
            test_df = pd.read_sql_query(
                f"SELECT * FROM {self._config.test_collection_name};", conn
            )
        finally:
            conn.close()
        logging.info("Data fetched successfully from PostgreSQL.")
        return train_df, test_df

    def _save_artifacts(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> None:
        """Persist raw data to the feature store and ingested directories."""
        logging.info("Saving ingestion artifacts.")
        os.makedirs(os.path.dirname(self._config.feature_store_file_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._config.training_file_path), exist_ok=True)

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df.to_csv(self._config.feature_store_file_path, index=False)
        logging.info(f"Feature store saved: {self._config.feature_store_file_path}")

        train_df.to_csv(self._config.training_file_path, index=False)
        test_df.to_csv(self._config.testing_file_path, index=False)
        logging.info(f"Train saved: {self._config.training_file_path}")
        logging.info(f"Test saved: {self._config.testing_file_path}")

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """Public entry point — fetch, save, and return the artifact."""
        try:
            logging.info("DataIngestion started.")
            train_df, test_df = self._fetch_dataframes()
            self._save_artifacts(train_df, test_df)

            artifact = DataIngestionArtifact(
                trained_data_path=self._config.training_file_path,
                test_data_path=self._config.testing_file_path,
            )
            logging.info("DataIngestion completed.")
            return artifact
        except Exception as e:
            logging.error("DataIngestion failed.")
            raise MyException(e, sys) from e