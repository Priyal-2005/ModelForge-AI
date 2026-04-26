import time
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from src.deployment.schemas import PredictionInput, PredictionOutput
from src.deployment.predictor import predictor_service
from src.utils.logger import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("Starting API, loading model artifacts...")
    try:
        predictor_service.load_artifacts()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
    yield
    # Shutdown logic
    logger.info("API shut down")

app = FastAPI(title="ModelForge AI API", version="1.0.0", lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred."}
    )

@app.get("/")
def read_root():
    """Root endpoint returning basic API metadata."""
    return {"name": "ModelForge AI API", "version": "1.0.0", "status": "running"}

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
        logger.warning("Metadata requested but not loaded.")
        raise HTTPException(status_code=500, detail="Metadata not loaded.")
    return predictor_service.metadata

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if predictor_service.model is None:
        logger.error("Prediction requested but model is not loaded.")
        raise HTTPException(status_code=500, detail="Model is not loaded. Check server logs.")
        
    try:
        logger.info(f"Processing prediction request: {input_data.model_dump()}")
        result = predictor_service.predict(input_data.model_dump())
        logger.info(f"Prediction successful: {result}")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Prediction failed: {str(e)}")
