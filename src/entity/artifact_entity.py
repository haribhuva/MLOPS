from dataclasses import dataclass

@dataclass
class ArtifactEntity:
    """
    A dataclass to represent an artifact entity in the MLOps pipeline.
    """
    name: str
    type: str
    description: str
    uri: str

@dataclass
class DataIngestionArtifact(ArtifactEntity):
    """
    A dataclass to represent a data ingestion artifact.
    """
    trained_data_path: str
    test_data_path: str