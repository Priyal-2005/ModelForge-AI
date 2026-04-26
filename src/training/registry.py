from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_models():
    """Returns a dictionary of models and their hyperparameter grids."""
    return {
        "LogisticRegression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0]
            }
        },
        "RandomForest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_depth": [None, 5, 10]
            }
        },
        "DecisionTree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [None, 5, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1.0, 10.0],
                "kernel": ["linear", "rbf"]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        }
    }
