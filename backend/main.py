from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import re
from typing import Optional
from datetime import datetime

app = FastAPI(title="House Rent Prediction API", version="1.0.0", description="Predict house rent using Random Forest model")

# Load model once at startup
try:
    model = joblib.load("rent_prediction_model.pkl")
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

class HouseFeatures(BaseModel):
    """Input features for rent prediction (matching Kaggle notebook preprocessing)"""
    BHK: int  # e.g., 2
    Size: int  # Sqft, e.g., 800
    Bathroom: int  # e.g., 2
    Floor: str = "1 out of 3"  # e.g., "Ground out of 2", "5 out of 10"
    Area_Type: str = "Super Area"  # "Super Area", "Carpet Area", "Built Area"
    City: str = "Kolkata"  # e.g., "Mumbai", "Delhi"
    Furnishing_Status: str = "Semi-Furnished"  # "Furnished", "Semi-Furnished", "Unfurnished"
    Tenant_Preferred: str = "Bachelors/Family"  # "Bachelors", "Family"
    Point_of_Contact: str = "Contact Owner"  # "Contact Owner", "Contact Agent"
    Posted_Month: Optional[int] = None  # 1-12, defaults to current month if None

def extract_floor_info(floor_str: str) -> tuple[int, int]:
    """Parse Floor string like in Kaggle notebook (e.g., 'Ground out of 2' -> (0, 2))"""
    try:
        parts = str(floor_str).split(" out of ")
        if len(parts) == 2:
            current = parts[0].strip()
            total = int(parts[1].strip())
            # Handle text floors
            floor_map = {'Ground': 0, 'Basement': -1, 'Lower Basement': -2, 'Upper Basement': -1}
            if current in floor_map:
                current_num = floor_map[current]
            else:
                # Extract number if present
                num_match = re.findall(r'\d+', current)
                current_num = int(num_match[0]) if num_match else 0
            return current_num, total
        return 0, 1  # Default
    except Exception:
        return 0, 1

def preprocess_input(features: HouseFeatures) -> np.ndarray:
    """Preprocess input to match training data (one-hot encoded + scaled)"""
    # Get current month if not provided
    post_month = features.Posted_Month or datetime.now().month
    
    # Parse floor
    floor_current, floor_total = extract_floor_info(features.Floor)
    
    # Create base numerical features (matching training notebook order)
    numerical_features = [
        features.BHK,           # 0
        features.Size,          # 1  
        features.Bathroom,      # 2
        floor_current,          # 3 - Floor Level
        floor_total,            # 4 - Total Floors
        post_month,             # 5 - month posted
        post_month,             # 6 - day posted (simplified - using month)
        post_month % 7,         # 7 - day of week posted (simplified)
        (post_month - 1) // 3 + 1  # 8 - quarter posted
    ]
    
    # One-hot encoding for categorical variables (with drop_first=True)
    # Based on training notebook: ['Area Type', 'City', 'Furnishing Status', 'Tenant Preferred', 'Point of Contact']
    
    # Area Type: ['Built Area', 'Carpet Area', 'Super Area'] -> drop 'Built Area'
    area_carpet = 1 if features.Area_Type == "Carpet Area" else 0
    area_super = 1 if features.Area_Type == "Super Area" else 0
    
    # City: Common cities in dataset (you may need to adjust based on your training data)
    # Assuming: ['Bangalore', 'Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai'] -> drop first
    city_chennai = 1 if features.City == "Chennai" else 0
    city_delhi = 1 if features.City == "Delhi" else 0
    city_hyderabad = 1 if features.City == "Hyderabad" else 0  
    city_kolkata = 1 if features.City == "Kolkata" else 0
    city_mumbai = 1 if features.City == "Mumbai" else 0
    
    # Furnishing Status: ['Furnished', 'Semi-Furnished', 'Unfurnished'] -> drop 'Furnished'
    furnish_semi = 1 if features.Furnishing_Status == "Semi-Furnished" else 0
    furnish_unfurnished = 1 if features.Furnishing_Status == "Unfurnished" else 0
    
    # Tenant Preferred: ['Bachelors', 'Bachelors/Family', 'Family'] -> drop 'Bachelors'
    tenant_bachelor_family = 1 if features.Tenant_Preferred == "Bachelors/Family" else 0
    tenant_family = 1 if features.Tenant_Preferred == "Family" else 0
    
    # Point of Contact: ['Contact Agent', 'Contact Owner'] -> drop 'Contact Agent'  
    contact_owner = 1 if features.Point_of_Contact == "Contact Owner" else 0
    
    # Combine all features (should total 21 features to match model)
    all_features = numerical_features + [
        area_carpet, area_super,  # 9, 10
        city_chennai, city_delhi, city_hyderabad, city_kolkata, city_mumbai,  # 11-15
        furnish_semi, furnish_unfurnished,  # 16, 17
        tenant_bachelor_family, tenant_family,  # 18, 19
        contact_owner  # 20
    ]
    
    # Convert to numpy array and reshape for single prediction
    return np.array(all_features).reshape(1, -1)

@app.post("/predict", summary="Predict House Rent")
async def predict_rent(features: HouseFeatures):
    """Predict monthly rent based on house features."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Check server logs.")
    
    try:
        # Preprocess
        input_array = preprocess_input(features)
        
        # Predict (model was trained on Box-Cox transformed target, need to inverse transform)
        boxcox_pred = model.predict(input_array)[0]
        
        # Inverse Box-Cox transformation (lambda from training notebook)
        # Note: You may need to save lambda value from training. For now using approximate value
        lambda_value = 0.0  # This should be saved from training; 0 means log transform
        if lambda_value == 0:
            predicted_rent = int(np.exp(boxcox_pred))  # Inverse of log
        else:
            # General Box-Cox inverse: (lambda * y + 1)^(1/lambda)
            predicted_rent = int(np.power(lambda_value * boxcox_pred + 1, 1/lambda_value))
        
        # Ensure reasonable bounds
        predicted_rent = max(1000, min(predicted_rent, 500000))  # Clamp between 1k-500k
        
        confidence = "High" if 5000 <= predicted_rent <= 100000 else "Medium"
        
        return {
            "predicted_rent": f"₹{predicted_rent:,}",
            "confidence": confidence,
            "features_processed": input_array.shape[1],
            "input_summary": features.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/")
async def root():
    """Health check."""
    return {"message": "House Rent Prediction API is running!", "model_loaded": model is not None}

@app.get("/health")
async def health():
    """API health endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}