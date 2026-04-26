import pandas as pd
import os

def select_best_model(evaluations, save_path="outputs/model_comparison.csv"):
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
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    return best_model, best_f1, df
