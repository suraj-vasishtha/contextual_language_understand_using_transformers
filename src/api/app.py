from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
from pathlib import Path
from ..service.model_service import ModelService

app = FastAPI(title="Text Classification API")

# Initialize model service
MODEL_PATH = Path("model_checkpoints/best_model.pth")
CONFIG_PATH = Path("configs/config.yaml")

# Check if model exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please train the model first.")

model_service = None

@app.on_event("startup")
async def startup_event():
    global model_service
    model_service = ModelService(
        model_path=str(MODEL_PATH),
        config_path=str(CONFIG_PATH)
    )

class PredictionRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predictions: List[Dict[str, float]]

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_service:
        raise HTTPException(status_code=500, detail="Model service not initialized")
    try:
        predictions = model_service.predict(request.texts)
        return PredictionResponse(predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    start_server() 