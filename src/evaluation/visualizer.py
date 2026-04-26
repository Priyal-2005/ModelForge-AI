import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from pathlib import Path
from src.utils.logger import logger

def generate_confusion_matrices(y_true, predictions_dict: dict, save_dir="outputs/plots"):
    """Generates and saves confusion matrices for all models."""
    p_dir = Path(save_dir).resolve()
    p_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, y_pred in predictions_dict.items():
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(p_dir / f'cm_{model_name}.png')
        plt.close()
    logger.info(f"Confusion matrices saved to {p_dir}")

def generate_roc_curves(y_true, probabilities_dict: dict, save_path="outputs/plots/roc_curves.png"):
    """Generates and saves combined ROC curves."""
    p = Path(save_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    
    for model_name, y_prob in probabilities_dict.items():
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')
            
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(p)
    plt.close()
    logger.info(f"ROC curves saved to {p}")
