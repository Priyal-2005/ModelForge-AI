import uvicorn
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

if __name__ == "__main__":
    print("Starting ModelForge AI API...")
    print("API will be running at: http://0.0.0.0:8000")
    print("Docs available at: http://0.0.0.0:8000/docs")
    uvicorn.run("src.deployment.api:app", host="0.0.0.0", port=8000, reload=True)
