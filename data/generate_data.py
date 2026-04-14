import pandas as pd
import numpy as np
import os

def generate_synthetic_data(n_samples=2500, seed=42):
    np.random.seed(seed)
    
    # Base features
    sleep_hours = np.random.normal(6.5, 1.2, n_samples).clip(3, 10)
    study_hours = np.random.normal(5, 2, n_samples).clip(1, 12)
    assignment_load = np.random.randint(1, 10, n_samples)
    social_activity = np.random.normal(3, 1.5, n_samples).clip(0, 8)
    screen_time = np.random.normal(4, 2, n_samples).clip(1, 12)
    physical_activity = np.random.normal(1, 0.8, n_samples).clip(0, 3)
    diet_quality = np.random.choice(['Poor', 'Average', 'Good'], n_samples, p=[0.2, 0.5, 0.3])
    gpa = np.random.uniform(2.0, 4.0, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Sleep_Hours': sleep_hours,
        'Study_Hours': study_hours,
        'Assignment_Load': assignment_load,
        'Social_Activity': social_activity,
        'Screen_Time': screen_time,
        'Physical_Activity': physical_activity,
        'Diet_Quality': diet_quality,
        'GPA': gpa
    })
    
    # Custom stress logic (simplified for generation, model will learn this)
    # Higher study load, higher assignments, lower sleep, lower physical activity -> Higher Stress
    stress_score = (
        (8 - df['Sleep_Hours']) * 1.5 + 
        df['Study_Hours'] * 0.8 + 
        df['Assignment_Load'] * 1.2 + 
        df['Screen_Time'] * 0.5 - 
        df['Physical_Activity'] * 2.0 - 
        df['Social_Activity'] * 0.5
    )
    
    # Factor in Diet
    diet_map = {'Poor': 2, 'Average': 0, 'Good': -2}
    stress_score += df['Diet_Quality'].map(diet_map)
    
    # Add noise
    stress_score += np.random.normal(0, 2, n_samples)
    
    # Categorize stress
    # Using quantiles to ensure balanced classes
    q1 = np.percentile(stress_score, 33)
    q2 = np.percentile(stress_score, 66)
    
    def categorize_stress(s):
        if s < q1: return 'Low'
        elif s < q2: return 'Medium'
        else: return 'High'
        
    df['Stress_Level'] = stress_score.apply(categorize_stress)
    
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_synthetic_data()
    
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    output_path = os.path.join('data', 'synthetic_dataset.csv')
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print("\nDataset Preview:")
    print(df.head())
    print("\nClass Distribution:")
    print(df['Stress_Level'].value_counts())
