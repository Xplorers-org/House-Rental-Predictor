# routers/predict.py
from fastapi import APIRouter, HTTPException
from schema.prediction import HouseFeatures
from model.model import get_model
from utils.preprocess import preprocess_input
import numpy as np

router = APIRouter(prefix="/api", tags=["Prediction"])

# Update this lambda if you used Box-Cox! (most people used log → lambda ≈ 0)
LAMBDA = 0.105  # ← Change this to your actual lambda from training (or 0 if pure log)

@router.post("/predict")
async def predict_rent(features: HouseFeatures):
    model = get_model()
    
    try:
        X = preprocess_input(features)
        boxcox_pred = model.predict(X)[0]

        # Inverse transform
        if abs(LAMBDA) < 1e-6:  # ≈ log transform
            predicted_rent = int(np.exp(boxcox_pred))
        else:
            predicted_rent = int((LAMBDA * boxcox_pred + 1) ** (1 / LAMBDA))

        predicted_rent = max(3000, min(predicted_rent, 500000))

        confidence = "High" if 8000 <= predicted_rent <= 150000 else "Check inputs"

        return {
            "predicted_rent_inr": predicted_rent,
            "predicted_rent": f"₹{predicted_rent:,}",
            "confidence": confidence,
            "input": features.dict()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@router.get("/health")
async def health():
    return {"status": "healthy", "model_loaded": get_model() is not None}