import os
from dotenv import load_dotenv
from datetime import date

load_dotenv()

DATABASE_NAME = "mlops"
USER = "postgres"
HOST = "localhost"
POSTGRE_SQL_PASSWORD = os.getenv("POSTGRE_SQL_PASSWORD")

PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

TARGET_COLUMN = "cycle"
CURRENT_YEAR = date.today().year
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"

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

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.keras"
MODEL_TRAINER_MODEL_CONFIG_FILE_PATH: str = os.path.join("config", "model.yaml")

"""
MODEL EVALUATION related constant start with MODEL_EVALUATION var name
"""
MODEL_EVALUATION_DIR_NAME: str = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME: str = "evaluation_report.yaml"