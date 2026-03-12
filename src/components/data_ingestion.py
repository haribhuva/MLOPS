from src.configuration.postgresql_connection import postgresql_client
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import configure_logger

import os
import sys
import pandas as pd


class DataAccess:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.logger = configure_logger()
        self.logger.info("DataAccess class initialized with DataIngestionConfig.")

    def export_data_collection_as_dataframe(self) -> tuple:
        try:
            self.logger.info("Attempting to fetch data from PostgreSQL DB..")

            query_train = f"SELECT * FROM {self.data_ingestion_config.train_collection_name};"
            query_test = f"SELECT * FROM {self.data_ingestion_config.test_collection_name};"

            conn = postgresql_client()
            train_df = pd.read_sql_query(query_train, conn)
            test_df = pd.read_sql_query(query_test, conn)
            conn.close()

            self.logger.info("Data fetched successfully from PostgreSQL DB.")

            # Create artifact directories and save data
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            ingested_dir = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)
            os.makedirs(ingested_dir, exist_ok=True)

            # Save combined data to feature store
            combined_df = pd.concat([train_df, test_df], ignore_index=True)
            combined_df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            self.logger.info(f"Feature store saved at: {self.data_ingestion_config.feature_store_file_path}")

            # Save train and test data to ingested directory
            train_df.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_df.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            self.logger.info(f"Train data saved at: {self.data_ingestion_config.training_file_path}")
            self.logger.info(f"Test data saved at: {self.data_ingestion_config.testing_file_path}")

            # Return a proper artifact with the file paths
            data_ingestion_artifact = DataIngestionArtifact(
                trained_data_path=self.data_ingestion_config.training_file_path,
                test_data_path=self.data_ingestion_config.testing_file_path,
            )
            return data_ingestion_artifact

        except Exception as e:
            self.logger.error("Failed to fetch data from PostgreSQL DB.")
            raise MyException(e, sys)
        
if __name__ == "__main__":
    data_ingestion_config = DataIngestionConfig()
    data_access = DataAccess(data_ingestion_config)
    train_df, test_df = data_access.export_data_collection_as_dataframe()
    data_access.logger.info("Data fetched and saved successfully.")
    print(train_df.head())
    print(test_df.head())