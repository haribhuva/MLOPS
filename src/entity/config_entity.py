"""Dataclass configs defining artifact directory structure for each pipeline stage."""
import os
from src.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")


@dataclass
class TrainingPipelineConfig:
    """Root config anchoring all stage sub-directories."""
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config: TrainingPipelineConfig = TrainingPipelineConfig()


@dataclass
class DataIngestionConfig:
    """Paths for raw data ingestion artifacts."""
    data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
    feature_store_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
    training_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
    testing_file_path: str = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
    train_collection_name: str = DATA_INGESTION_TRAIN_COLLECTION_NAME
    test_collection_name: str = DATA_INGESTION_TEST_COLLECTION_NAME


@dataclass
class DataValidationConfig:
    """Paths for data validation report artifacts."""
    data_validation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME)
    validation_report_file_path: str = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME)


@dataclass
class DataTransformationConfig:
    """Paths for transformed data and preprocessing object artifacts."""
    data_transformation_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                    TRAIN_FILE_NAME)
    transformed_test_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                   TEST_FILE_NAME)
    transformed_object_file_path: str = os.path.join(data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                     PREPROCESSING_OBJECT_FILE_NAME)


@dataclass
class ModelTrainerConfig:
    """Paths for trained model artifacts and model config reference."""
    model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME)
    trained_model_file_path: str = os.path.join(model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR,
                                                 MODEL_TRAINER_TRAINED_MODEL_NAME)
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH


@dataclass
class ModelEvaluationConfig:
    """Paths for evaluation reports, plots, and metrics artifacts."""
    model_evaluation_dir: str = os.path.join(training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME)
    evaluation_report_file_path: str = os.path.join(model_evaluation_dir, MODEL_EVALUATION_REPORT_FILE_NAME)
    random_split_evaluation_dir: str = os.path.join(model_evaluation_dir, MODEL_EVALUATION_RANDOM_SPLIT_DIR_NAME)
    external_holdout_evaluation_dir: str = os.path.join(model_evaluation_dir, MODEL_EVALUATION_EXTERNAL_HOLDOUT_DIR_NAME)
    metrics_file_name: str = MODEL_EVALUATION_METRICS_FILE_NAME
    summary_file_name: str = MODEL_EVALUATION_SUMMARY_FILE_NAME
    model_config_file_path: str = MODEL_TRAINER_MODEL_CONFIG_FILE_PATH