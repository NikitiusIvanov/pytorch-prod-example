from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import json

from models import SimpleRegressionNN

with open('inference_config.json', 'r') as f:
    inference_config = json.load(f)

# inference_config = {
#     "best_model_path": "./models_2024-07-04/model_epoch_14.pth",
#     "forecasting_horizon_days": 7,
#     "window_size_hours": 336,
#     "hidden_size": 256,
#     "dropout_rate": 0.01,
# }

best_model_path = inference_config['best_model_path']
window_size_hours = inference_config['window_size_hours']
hidden_size = inference_config['hidden_size']
dropout_rate = inference_config['dropout_rate']

# Init model
model = SimpleRegressionNN(
    input_size=2 * window_size_hours,
    hidden_size=hidden_size, 
    dropout_rate=dropout_rate
)
# Load weights
model.load_state_dict(torch.load(best_model_path))
model.eval()

@torch.inference_mode()
def run_inference(
    model: SimpleRegressionNN, 
    heartrate: list, 
    steps: list, 
    window_size_hours: int
) -> float:
    if len(heartrate) != len(steps):
        raise ValueError("Heart rate and steps lists must have the same length")
    
    window_size = window_size_hours
    if len(heartrate) < window_size:
        raise ValueError("Input lists must be at least the size of the window")

    # Prepare the input tensor
    input_slice = np.stack((heartrate[-window_size:], steps[-window_size:]), axis=1)
    input_tensor = torch.tensor(input_slice, dtype=torch.float32).unsqueeze(0)

    # Get the predict
    output = model(input_tensor)
    output = output.view(-1).item()
    
    return output


# Define the request schema
class InferenceRequest(BaseModel):
    heart_rate: list[float]
    steps: list[float]

# Define the FastAPI app
app = FastAPI()

@app.post("/predict/")
def predict(request: InferenceRequest):
    try:
        prediction = run_inference(
            model=model,
            steps=request.steps,
            heartrate=request.heart_rate, 
            window_size_hours=window_size_hours
        )
        return {"prediction": prediction}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
