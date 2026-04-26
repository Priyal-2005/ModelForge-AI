from .registry import get_models
from .tuner import tune_model
from src.utils.logger import logger

def train_all_models(X_train, y_train):
    """Trains all models sequentially and returns a dict of trained model info."""
    models_registry = get_models()
    trained_models = {}
    
    for name, config in models_registry.items():
        logger.info(f"Training {name}...")
        results = tune_model(config['model'], config['params'], X_train, y_train)
        trained_models[name] = results
        logger.info(f"Completed {name}. Best CV Score: {results['cv_score']:.4f}")
        
    return trained_models
