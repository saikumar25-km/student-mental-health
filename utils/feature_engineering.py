import pandas as pd

def apply_feature_engineering(df):
    """
    Apply feature engineering to the student dataset.
    """
    # 1. Sleep Deficit: 8 hours is ideal
    df['Sleep_Deficit'] = (8 - df['Sleep_Hours']).clip(lower=0)
    
    # 2. Assignment Density: Assignments per study hour
    # Avoid division by zero
    df['Assignment_Density'] = df['Assignment_Load'] / (df['Study_Hours'] + 0.5)
    
    # 3. Lifestyle Score: Combined metric (higher is better)
    # Weights: Physical Activity (0.4), Diet (0.3), Sleep (0.2), Social (0.1)
    
    diet_map = {'Poor': 1, 'Average': 2, 'Good': 3}
    diet_score = df['Diet_Quality'].map(diet_map)
    
    # Normalize components to 0-1 for scoring
    phys_norm = (df['Physical_Activity'] - df['Physical_Activity'].min()) / (df['Physical_Activity'].max() - df['Physical_Activity'].min())
    diet_norm = (diet_score - 1) / 2
    sleep_norm = df['Sleep_Hours'] / 10
    social_norm = df['Social_Activity'] / 8
    
    df['Lifestyle_Score'] = (
        phys_norm * 0.4 + 
        diet_norm * 0.3 + 
        sleep_norm * 0.2 + 
        social_norm * 0.1
    ) * 100 # Scale to 0-100
    
    return df
