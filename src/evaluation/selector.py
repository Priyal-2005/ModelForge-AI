import pandas as pd
from pathlib import Path
from src.utils.logger import logger

def select_best_model(evaluations: dict, save_path="outputs/model_comparison.csv"):
    """Selects best model based on F1-score and saves comparison table."""
    records = []
    best_model = None
    best_f1 = -1
    
    for name, metrics in evaluations.items():
        records.append({
            "Model": name,
            "F1-Score": metrics.get("f1", 0),
            "Accuracy": metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall": metrics.get("recall", 0),
            "ROC-AUC": metrics.get("roc_auc", 0)
        })
        
        if metrics.get("f1", 0) > best_f1:
            best_f1 = metrics.get("f1", 0)
            best_model = name
            
    df = pd.DataFrame(records)
    df = df.sort_values(by="F1-Score", ascending=False)
    
    p = Path(save_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    logger.info(f"Comparison results saved to {p}")
    
    return best_model, best_f1, df
