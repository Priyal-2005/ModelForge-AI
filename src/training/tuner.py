from sklearn.model_selection import GridSearchCV
import time
from src.utils.logger import logger

def tune_model(model, param_grid, X_train, y_train):
    """Tunes a model using GridSearchCV and returns the best estimator, params, CV score, and time."""
    start_time = time.time()
    
    logger.info(f"Starting GridSearchCV with {len(param_grid)} parameter grids...")
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    training_time = time.time() - start_time
    logger.info(f"GridSearchCV completed in {training_time:.2f}s")
    
    return {
        "best_estimator": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "cv_score": grid_search.best_score_,
        "training_time": training_time
    }
