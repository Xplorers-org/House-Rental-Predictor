import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "rent_prediction_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Failed to load model: {e}")
    model = None

def get_model():
    if model is None:
        raise RuntimeError("Model not loaded!")
    return model