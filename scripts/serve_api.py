import uvicorn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.config import PORT

if __name__ == "__main__":
    print("Starting ModelForge AI API...")
    print(f"API will be running at: http://0.0.0.0:{PORT}")
    print(f"Docs available at: http://0.0.0.0:{PORT}/docs")
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=PORT, reload=False)
