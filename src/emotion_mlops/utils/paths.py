from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRACKING_URI = f"sqlite:///{PROJECT_ROOT}/mlflow.db"
REGISTRY = "s3://emotion-mlops/mlflow-artifacts"
