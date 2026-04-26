import json
from pathlib import Path
from datetime import datetime
from src.utils.logger import logger

class ExperimentTracker:
    def __init__(self, filepath="outputs/experiments.json"):
        self.filepath = Path(filepath).resolve()
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        
    def log_experiment(self, model_name: str, metrics: dict, params: dict, training_time: float):
        """Logs experiment results to a JSON file."""
        record = {
            "model": model_name,
            "f1_score": metrics.get("f1", 0.0),
            "accuracy": metrics.get("accuracy", 0.0),
            "params": params,
            "training_time": training_time,
            "timestamp": datetime.now().isoformat()
        }
        
        history = []
        if self.filepath.exists():
            with self.filepath.open("r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        history.append(record)
        
        with self.filepath.open("w") as f:
            json.dump(history, f, indent=4)
        logger.info(f"Experiment logged for {model_name} in {self.filepath}")
