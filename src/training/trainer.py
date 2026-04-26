from .registry import get_models
from .tuner import tune_model

def train_all_models(X_train, y_train):
    """Trains all models sequentially and returns a dict of trained model info."""
    models_registry = get_models()
    trained_models = {}
    
    for name, config in models_registry.items():
        print(f"Training {name}...")
        results = tune_model(config['model'], config['params'], X_train, y_train)
        trained_models[name] = results
        
    return trained_models
