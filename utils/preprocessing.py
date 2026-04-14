import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

def preprocess_data(df, is_training=True, scaler_path='models/scaler.joblib'):
    """
    Handle missing values, encode categorical variables, and scale features.
    """
    # 1. Handle Missing Values (if any)
    df = df.copy()
    df = df.fillna(df.median(numeric_only=True))
    
    # 2. Encode Categorical Variables
    if 'Diet_Quality' in df.columns:
        diet_map = {'Poor': 0, 'Average': 1, 'Good': 2}
        df['Diet_Quality'] = df['Diet_Quality'].map(diet_map)
    
    # Stress_Level encoding is handled separately during training for target
    
    # 3. Scaling
    features = [col for col in df.columns if col not in ['Stress_Level', 'Stress_Level_Encoded']]
    
    if is_training:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
        # Ensure models directory exists
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(scaler, scaler_path)
    else:
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            df[features] = scaler.transform(df[features])
        else:
            raise FileNotFoundError(f"Scaler not found at {scaler_path}. Train the model first.")
            
    return df, features

def encode_target(y):
    """
    Encode target labels to numeric values.
    """
    le = LabelEncoder()
    # Ensure consistent order: Low=0, Medium=1, High=2
    le.classes_ = np.array(['Low', 'Medium', 'High'])
    return le.transform(y), le

if __name__ == "__main__":
    import numpy as np # Needed for le.classes_
