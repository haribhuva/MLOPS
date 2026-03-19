from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    """Stores paths to the ingested train and test CSV files."""
    trained_data_path: str
    test_data_path: str


@dataclass
class DataValidationArtifact:
    """Stores the result of data validation checks."""
    validation_status: bool
    message: str
    validation_report_file_path: str

@dataclass
class DataTransformationArtifact:
    """Stores paths to the transformed training and testing data, and the preprocessing object."""
    transformed_train_file_path: str
    transformed_test_file_path: str
    transformed_object_file_path: str

@dataclass
class ModelTrainerArtifact:
    """Stores the trained model path and evaluation metrics."""
    trained_model_file_path: str
    metric_artifact: dict

@dataclass
class ModelEvaluationArtifact:
    """Stores evaluation results and model acceptance decision."""
    is_model_accepted: bool
    changed_accuracy: float
    trained_model_path: str
    best_model_path: str