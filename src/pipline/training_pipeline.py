"""Orchestrates the full training pipeline: ingest → validate → transform → train → evaluate."""
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_evaluation import ModelEvaluation
from src.components.model_trainer import ModelTrainer
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    DataValidationArtifact,
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
)
from src.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from src.exception import MyException
from src.logger import logging


class TrainPipeline:
    """Runs every pipeline stage in sequence, passing artifacts between stages."""

    def __init__(self) -> None:
        self._ingestion_config = DataIngestionConfig()
        self._validation_config = DataValidationConfig()
        self._transformation_config = DataTransformationConfig()
        self._trainer_config = ModelTrainerConfig()
        self._evaluation_config = ModelEvaluationConfig()

    def _start_data_ingestion(self) -> DataIngestionArtifact:
        """Fetch raw data from PostgreSQL and persist as CSV."""
        try:
            logging.info("Pipeline stage: data ingestion.")
            ingestion = DataIngestion(data_ingestion_config=self._ingestion_config)
            return ingestion.initiate_data_ingestion()
        except Exception as e:
            raise MyException(e, sys) from e

    def _start_data_validation(
        self, data_ingestion_artifact: DataIngestionArtifact
    ) -> DataValidationArtifact:
        """Validate ingested data against schema."""
        try:
            logging.info("Pipeline stage: data validation.")
            validation = DataValidation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_validation_config=self._validation_config,
            )
            return validation.initiate_data_validation()
        except Exception as e:
            raise MyException(e, sys) from e

    def _start_data_transformation(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_artifact: DataValidationArtifact,
    ) -> DataTransformationArtifact:
        """Clean and feature-engineer the validated data."""
        try:
            logging.info("Pipeline stage: data transformation.")
            transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self._transformation_config,
                data_validation_artifact=data_validation_artifact,
            )
            return transformation.initiate_data_transformation()
        except Exception as e:
            raise MyException(e, sys) from e

    def _start_model_trainer(
        self, data_transformation_artifact: DataTransformationArtifact
    ) -> ModelTrainerArtifact:
        """Select and train the best model."""
        try:
            logging.info("Pipeline stage: model training.")
            trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self._trainer_config,
            )
            return trainer.initiate_model_trainer()
        except Exception as e:
            raise MyException(e, sys) from e

    def _start_model_evaluation(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        model_trainer_artifact: ModelTrainerArtifact,
    ) -> ModelEvaluationArtifact:
        """Evaluate the trained model on the test set."""
        try:
            logging.info("Pipeline stage: model evaluation.")
            evaluation = ModelEvaluation(
                model_evaluation_config=self._evaluation_config,
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_artifact=model_trainer_artifact,
            )
            return evaluation.initiate_model_evaluation()
        except Exception as e:
            raise MyException(e, sys) from e

    def run_pipeline(self) -> None:
        """Execute every stage in sequence."""
        try:
            logging.info("TrainPipeline started.")

            ingestion_artifact = self._start_data_ingestion()
            validation_artifact = self._start_data_validation(ingestion_artifact)
            transformation_artifact = self._start_data_transformation(
                ingestion_artifact, validation_artifact
            )
            trainer_artifact = self._start_model_trainer(transformation_artifact)
            evaluation_artifact = self._start_model_evaluation(
                transformation_artifact, trainer_artifact
            )

            if not evaluation_artifact.is_model_accepted:
                logging.warning("Model not accepted — pipeline stopping.")
                return

            logging.info("TrainPipeline completed successfully.")

        except Exception as e:
            logging.error("TrainPipeline failed.")
            raise MyException(e, sys) from e