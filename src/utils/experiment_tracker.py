import json
import os
from datetime import datetime

class ExperimentTracker:
    def __init__(self, filepath="outputs/experiments.json"):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def log_experiment(self, model_name, metrics, params, training_time):
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
        if os.path.exists(self.filepath):
            with open(self.filepath, "r") as f:
                try:
                    history = json.load(f)
                except json.JSONDecodeError:
                    pass
                    
        history.append(record)
        
        with open(self.filepath, "w") as f:
            json.dump(history, f, indent=4)
