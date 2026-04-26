# ModelForge AI

An End-to-End Machine Learning Pipeline & Deployment System.

## Project Overview
ModelForge AI is a robust, modular, production-ready ML system built to automatically load data, preprocess features, train multiple classification models, evaluate their performance, select the best model, and persist it for real-time inference via a highly scalable FastAPI layer.

## Architecture
- **Phase 1: Data Pipeline**: Dynamic dataset loading (`seaborn`), schema validation, and structured preprocessing (scaling, imputation, one-hot encoding).
- **Phase 2: Training Pipeline**: Multi-model training orchestration using a configuration registry and hyperparameter tuning with `GridSearchCV`.
- **Phase 3: Evaluation Pipeline**: Comprehensive metrics collection, visualizations (ROC & Confusion Matrices), and an automated best-model selector based on F1-score.
- **Phase 4: Model Persistence**: Saving binary artifacts (`joblib`) and metadata safely into persistent directories, preserving feature consistency for incoming inference requests.
- **Phase 5: FastAPI Deployment**: Production API offering a `/predict` endpoint wrapped in Pydantic schema validation.

## Features
- **Multi-model training**: Automatically evaluates LogisticRegression, RandomForest, DecisionTree, SVM, and XGBoost.
- **Hyperparameter tuning**: Utilizes parallelized `GridSearchCV`.
- **Evaluation + comparison**: Saves ROC curves, confusion matrices, and detailed metrics logs (`outputs/experiments.json`).
- **Model persistence**: Keeps artifacts aligned with training features.
- **FastAPI deployment**: Real-time prediction endpoint with confidence thresholds.

## How to run

### 1. Training the Pipeline
To extract data, build the models, run evaluations, and save artifacts:
```bash
python scripts/train_pipeline.py
```

### 2. Serving the API
To launch the FastAPI service and begin accepting traffic:
```bash
python scripts/serve_api.py
```

*Docs will be available at: http://localhost:8000/docs*

## API Example

You can hit the live `/predict` endpoint with a cURL request:

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "pclass": 3,
           "sex": "male",
           "age": 22.0,
           "sibsp": 1,
           "parch": 0,
           "fare": 7.25,
           "embarked": "S"
         }'
```

**Expected Response**:
```json
{
  "prediction": 0,
  "probability": 0.12,
  "model": "XGBoost",
  "confidence": "high"
}
```

## Deployment Note
**Deployed using Docker + Render.** The system ships with a multi-stage `Dockerfile` which natively consumes standard `.env` variables like `$PORT`, ready for 1-click cloud deployments.