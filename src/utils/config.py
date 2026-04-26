import os
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH", "models/preprocessor.pkl")
METADATA_PATH = os.getenv("METADATA_PATH", "models/metadata.json")
PORT = int(os.getenv("PORT", 8000))
