# utils/preprocessing.py
import numpy as np
import re
from datetime import datetime

def extract_floor_info(floor_str: str):
    try:
        parts = str(floor_str).split(" out of ")
        if len(parts) != 2:
            return 0, 1
        current = parts[0].strip()
        total = int(parts[1].strip())

        floor_map = {'Ground': 0, 'Basement': -1, 'Lower Basement': -2, 'Upper Basement': -1}
        if current in floor_map:
            current_num = floor_map[current]
        else:
            match = re.findall(r'\d+', current)
            current_num = int(match[0]) if match else 0
        return current_num, total
    except:
        return 0, 1

def preprocess_input(features) -> np.ndarray:
    post_month = features.Posted_Month or datetime.now().month
    floor_current, floor_total = extract_floor_info(features.Floor)

    # Numerical features (adjust if your model used different ones)
    numerical = [
        features.BHK,
        features.Size,
        features.Bathroom,
        floor_current,
        floor_total,
        post_month,
        post_month,              # dummy day
        post_month % 7,          # dummy weekday
        (post_month - 1) // 3 + 1 # quarter
    ]

    # Categorical one-hot (drop_first = True style)
    area_carpet = 1 if features.Area_Type == "Carpet Area" else 0
    area_super = 1 if features.Area_Type == "Super Area" else 0

    city_chennai = 1 if features.City == "Chennai" else 0
    city_delhi = 1 if features.City == "Delhi" else 0
    city_hyderabad = 1 if features.City == "Hyderabad" else 0
    city_kolkata = 1 if features.City == "Kolkata" else 0
    city_mumbai = 1 if features.City == "Mumbai" else 0

    furnish_semi = 1 if features.Furnishing_Status == "Semi-Furnished" else 0
    furnish_unfurnished = 1 if features.Furnishing_Status == "Unfurnished" else 0

    tenant_bf = 1 if features.Tenant_Preferred == "Bachelors/Family" else 0
    tenant_family = 1 if features.Tenant_Preferred == "Family" else 0

    contact_owner = 1 if features.Point_of_Contact == "Contact Owner" else 0

    features_list = numerical + [
        area_carpet, area_super,
        city_chennai, city_delhi, city_hyderabad, city_kolkata, city_mumbai,
        furnish_semi, furnish_unfurnished,
        tenant_bf, tenant_family,
        contact_owner
    ]

    return np.array(features_list).reshape(1, -1)