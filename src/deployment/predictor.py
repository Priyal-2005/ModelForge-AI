import pandas as pd
from src.utils.model_persistence import load_model, load_metadata
from src.utils.config import MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH
from src.utils.logger import logger

class Predictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = None
        
    def load_artifacts(self):
        """Loads model artifacts from disk."""
        self.model = load_model(MODEL_PATH)
        self.preprocessor = load_model(PREPROCESSOR_PATH)
        self.metadata = load_metadata(METADATA_PATH)
        logger.info("Predictor loaded all artifacts successfully.")
        
    def predict(self, input_data: dict) -> dict:
        """Runs preprocessing and prediction on the input data."""
        if self.model is None or self.preprocessor is None:
            logger.error("Model artifacts not loaded during prediction.")
            raise RuntimeError("Model artifacts not loaded.")
            
        # Convert JSON to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure features match exactly
        expected_features = self.metadata.get("features", [])
        if expected_features:
            missing_features = [f for f in expected_features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            df = df[expected_features]
            
        # Apply preprocessor
        X_processed = self.preprocessor.transform(df)
        
        # Predict
        prediction = int(self.model.predict(X_processed)[0])
        
        # Get probability
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(X_processed)[0][1])
        else:
            probability = float(prediction)
            
        # Confidence logic
        if probability >= 0.8 or probability <= 0.2:
            confidence = "high"
        elif probability >= 0.6 or probability <= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
            
        return {
            "prediction": prediction,
            "probability": probability,
            "model": self.metadata.get("model_type", "unknown"),
            "confidence": confidence
        }

predictor_service = Predictor()
