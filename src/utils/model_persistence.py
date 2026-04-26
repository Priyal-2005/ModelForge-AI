import joblib
import json
from pathlib import Path
from src.utils.logger import logger

def save_model(model, path: str):
    """Saves a model to the specified path."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    logger.info(f"Model saved to {p}")

def load_model(path: str):
    """Loads a model from the specified path."""
    p = Path(path).resolve()
    if not p.exists():
        logger.error(f"Model file not found at {p}")
        raise FileNotFoundError(f"Model file not found at {p}")
    logger.info(f"Loaded model from {p}")
    return joblib.load(p)

def save_metadata(model_name: str, metrics: dict, features: list, path: str):
    """Saves model metadata and feature names."""
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_type": model_name,
        "accuracy": metrics.get("accuracy", 0.0),
        "f1_score": metrics.get("f1", 0.0),
        "features": features
    }
    with p.open("w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved to {p}")

def load_metadata(path: str):
    """Loads model metadata."""
    p = Path(path).resolve()
    if not p.exists():
        logger.error(f"Metadata file not found at {p}")
        raise FileNotFoundError(f"Metadata file not found at {p}")
    with p.open("r") as f:
        return json.load(f)
