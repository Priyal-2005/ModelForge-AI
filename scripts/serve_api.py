import uvicorn
import os
import sys
from pathlib import Path

# Ensure absolute import path works
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.config import PORT
from src.utils.logger import logger

if __name__ == "__main__":
    logger.info("Starting ModelForge AI API...")
    logger.info(f"API will be running at: http://0.0.0.0:{PORT}")
    logger.info(f"Docs available at: http://0.0.0.0:{PORT}/docs")
    # Timeout is handled implicitly by uvicorn's defaults, but can be customized
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=PORT, reload=False, timeout_keep_alive=10)
