import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import load_titanic_data
from src.data.validator import validate_data
from src.data.preprocessor import DataPreprocessor
from src.training.trainer import train_all_models
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import generate_confusion_matrices, generate_roc_curves
from src.evaluation.selector import select_best_model
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.model_persistence import save_model, save_metadata

def main():
    print("=== MODELFORGE AI: TRAINING PIPELINE ===")
    
    # 1. Load data
    df = load_titanic_data()
    print(f"Data loaded successfully. Shape: {df.shape}")
    
    # 2. Validate data
    print("Validating data...")
    is_valid, errors = validate_data(df)
    if not is_valid:
        print("Data validation failed!")
        for error in errors:
            print(f"- {error}")
        sys.exit(1)
    print("Data validation passed.")
    
    # 3 & 4. Split and Preprocess
    print("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    print(f"Preprocessing complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Train models
    print("\n--- Training Phase ---")
    trained_models = train_all_models(X_train, y_train)
    
    # 6. Evaluate models
    print("\n--- Evaluation Phase ---")
    evaluations = {}
    predictions_dict = {}
    probabilities_dict = {}
    tracker = ExperimentTracker()
    
    for name, info in trained_models.items():
        model = info["best_estimator"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        
        predictions_dict[name] = y_pred
        probabilities_dict[name] = y_prob
        
        metrics = compute_metrics(y_test, y_pred, y_prob)
        evaluations[name] = metrics
        
        # Log to experiment tracker
        tracker.log_experiment(
            model_name=name,
            metrics=metrics,
            params=info["best_params"],
            training_time=info["training_time"]
        )
        print(f"Evaluated {name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
    # Visualizations
    print("Generating visualizations...")
    generate_confusion_matrices(y_test, predictions_dict)
    generate_roc_curves(y_test, probabilities_dict)
    
    # 7 & 8. Select best model & Save results
    print("\n--- Selection Phase ---")
    best_model, best_f1, comp_df = select_best_model(evaluations)
    print(f"Best model: {best_model} (F1: {best_f1:.4f})")
    print("Comparison results saved to outputs/model_comparison.csv")
    print("Experiment tracking updated in outputs/experiments.json")
    
    print("\n--- Model Persistence ---")
    print("Saving best model and preprocessor...")
    feature_names = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    best_model_obj = trained_models[best_model]["best_estimator"]
    
    save_model(best_model_obj, "models/best_model.pkl")
    save_model(preprocessor.preprocessor, "models/preprocessor.pkl")
    save_metadata(best_model, evaluations[best_model], feature_names, "models/metadata.json")
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
