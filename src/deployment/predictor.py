import pandas as pd
from src.utils.model_persistence import load_model, load_metadata

class Predictor:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.metadata = None
        
    def load_artifacts(self):
        """Loads model artifacts from disk."""
        self.model = load_model("models/best_model.pkl")
        self.preprocessor = load_model("models/preprocessor.pkl")
        self.metadata = load_metadata("models/metadata.json")
        
    def predict(self, input_data: dict) -> dict:
        """Runs preprocessing and prediction on the input data."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model artifacts not loaded.")
            
        # Convert JSON to DataFrame
        df = pd.DataFrame([input_data])
        
        # Ensure features match
        expected_features = self.metadata.get("features", [])
        if expected_features:
            # Reorder columns to match training and fill missing with appropriate defaults if needed
            # For simplicity, assuming incoming data matches expected_features perfectly
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
