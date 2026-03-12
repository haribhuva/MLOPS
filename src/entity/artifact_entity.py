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