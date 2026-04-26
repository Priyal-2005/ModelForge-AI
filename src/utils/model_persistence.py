import joblib
import json
import os

def save_model(model, path: str):
    """Saves a model to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str):
    """Loads a model from the specified path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found at {path}")
    return joblib.load(path)

def save_metadata(model_name: str, metrics: dict, features: list, path: str):
    """Saves model metadata and feature names."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    metadata = {
        "model_type": model_name,
        "accuracy": metrics.get("accuracy", 0.0),
        "f1_score": metrics.get("f1", 0.0),
        "features": features
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)

def load_metadata(path: str):
    """Loads model metadata."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found at {path}")
    with open(path, "r") as f:
        return json.load(f)
