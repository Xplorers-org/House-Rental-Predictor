from pydantic import BaseModel
from typing import Optional

class HouseFeatures(BaseModel):
    
    """
    Input features for rent prediction (matching Kaggle notebook preprocessing)
    """

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