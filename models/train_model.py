import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import shap

# Import custom utilities
import sys
sys.path.append(os.getcwd())
from utils.feature_engineering import apply_feature_engineering
from utils.preprocessing import preprocess_data

def train_and_evaluate():
    print("Loading data...")
    df = pd.read_csv('data/synthetic_dataset.csv')
    
    print("Applying feature engineering...")
    df = apply_feature_engineering(df)
    
    # Encode target
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['Stress_Level_Encoded'] = df['Stress_Level'].map(stress_map)
    y = df['Stress_Level_Encoded']
    
    print("Preprocessing data...")
    X_processed, feature_names = preprocess_data(df.drop(['Stress_Level', 'Stress_Level_Encoded'], axis=1), is_training=True)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTraining on {len(X_train)} samples, testing on {len(X_test)} samples.")
    
    # 1. Logistic Regression (Baseline)
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
    
    # 2. XGBoost (Primary Model)
    print("\nTraining XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    print("XGBoost Accuracy:", xgb_acc)
    print("\nXGBoost Classification Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=['Low', 'Medium', 'High']))
    
    # Save the best model
    print("\nSaving best model (XGBoost)...")
    joblib.dump(xgb_model, 'models/saved_model.joblib')
    
    # SHAP Explainability
    print("\nCalculating SHAP values...")
    explainer = shap.TreeExplainer(xgb_model)
    # Convert numpy array back to pandas for feature names in SHAP
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    shap_values = explainer.shap_values(X_test_df)
    
    # Save explainer for later use in app
    joblib.dump(explainer, 'models/shap_explainer.joblib')
    
    print("\nModel training and evaluation complete!")
    return xgb_model, feature_names

if __name__ == "__main__":
    train_and_evaluate()
