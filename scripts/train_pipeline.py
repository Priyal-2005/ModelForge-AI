import sys
import os
from pathlib import Path

# Ensure absolute import path works
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_titanic_data
from src.data.validator import validate_data
from src.data.preprocessor import DataPreprocessor
from src.training.trainer import train_all_models
from src.evaluation.metrics import compute_metrics
from src.evaluation.visualizer import generate_confusion_matrices, generate_roc_curves
from src.evaluation.selector import select_best_model
from src.utils.experiment_tracker import ExperimentTracker
from src.utils.model_persistence import save_model, save_metadata
from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH
from src.utils.logger import logger

def main():
    logger.info("=== MODELFORGE AI: TRAINING PIPELINE ===")
    
    # 1. Load data
    df = load_titanic_data()
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    
    # 2. Validate data
    logger.info("Validating data...")
    is_valid, errors = validate_data(df)
    if not is_valid:
        logger.error("Data validation failed!")
        for error in errors:
            logger.error(f"- {error}")
        sys.exit(1)
    logger.info("Data validation passed.")
    
    # 3 & 4. Split and Preprocess
    logger.info("Preprocessing data...")
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df)
    logger.info(f"Preprocessing complete. Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
    
    # 5. Train models
    logger.info("\n--- Training Phase ---")
    trained_models = train_all_models(X_train, y_train)
    
    # 6. Evaluate models
    logger.info("\n--- Evaluation Phase ---")
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
        logger.info(f"Evaluated {name} - F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        
    # Visualizations
    logger.info("Generating visualizations...")
    generate_confusion_matrices(y_test, predictions_dict)
    generate_roc_curves(y_test, probabilities_dict)
    
    # 7 & 8. Select best model & Save results
    logger.info("\n--- Selection Phase ---")
    best_model, best_f1, comp_df = select_best_model(evaluations)
    logger.info(f"Best model: {best_model} (F1: {best_f1:.4f})")
    
    logger.info("\n--- Model Persistence ---")
    logger.info("Saving best model and preprocessor...")
    feature_names = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    best_model_obj = trained_models[best_model]["best_estimator"]
    
    save_model(best_model_obj, MODEL_PATH)
    save_model(preprocessor.preprocessor, PREPROCESSOR_PATH)
    save_metadata(best_model, evaluations[best_model], feature_names, METADATA_PATH)
    
    logger.info("Pipeline completed successfully.")

if __name__ == "__main__":
    main()
