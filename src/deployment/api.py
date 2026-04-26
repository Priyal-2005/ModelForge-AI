from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from .schemas import PredictionInput, PredictionOutput
from .predictor import predictor_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    print("Loading model artifacts...")
    try:
        predictor_service.load_artifacts()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
    yield
    # Shutdown logic

app = FastAPI(title="ModelForge AI API", lifespan=lifespan)

@app.get("/health")
def health_check():
    is_loaded = predictor_service.model is not None
    return {
        "status": "healthy",
        "model_loaded": is_loaded
    }

@app.get("/models")
def get_models():
    if not predictor_service.metadata:
        raise HTTPException(status_code=500, detail="Metadata not loaded.")
    return predictor_service.metadata

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if predictor_service.model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")
        
    try:
        result = predictor_service.predict(input_data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")
