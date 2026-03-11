import os
from dotenv import load_dotenv

load_dotenv()

POSTGRE_SQL_PASSWORD = os.getenv("POSTGRE_SQL_PASSWORD")

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_TRAIN_COLLECTION_NAME: str = "train_FD001"
DATA_INGESTION_TEST_COLLECTION_NAME: str = "test_FD001"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
# DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.25