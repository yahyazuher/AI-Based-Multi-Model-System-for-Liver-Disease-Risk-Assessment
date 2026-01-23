"""
Project: AI-Based Multi-Model System for Liver Disease Risk Assessment
Developer: Yahya Zuher
Description: This script fetches training data directly from GitHub and trains 
             an optimized XGBoost model for status prediction.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys

# --- Configuration (GitHub Integration) ---
RAW_DATA_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/data/processed/hepatitisC_status.csv'
MODEL_FILENAME = 'hepatitisC_status_model.pkl'

def get_live_dataset():
    """Downloads the dataset from GitHub."""
    try:
        print(f"⬇️ Downloading dataset from GitHub...")
        df = pd.read_csv(RAW_DATA_URL)
        print(f"✅ Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        sys.exit(f"❌ Critical Error: Could not fetch data. {e}")

def add_medical_features(df):
    """Adds ALBI and APRI features for high-precision mortality prediction."""
    df_eng = df.copy()
    
    # 1. APRI Score
    df_eng['APRI'] = ((df_eng['SGOT'] / 40.0) / (df_eng['Platelets'] + 0.1)) * 100
    
    # 2. ALBI Score (Log-based liver function assessment)
    df_eng['ALBI_Score'] = (np.log10(df_eng['Bilirubin'].clip(lower=0.1) * 17.1) * 0.66) + \
                            (df_eng['Albumin'] * 10 * -0.085)
    
    # 3. Bilirubin to Albumin Ratio
    df_eng['Bili_Alb_Ratio'] = df_eng['Bilirubin'] / (df_eng['Albumin'] + 0.1)
    
    return df_eng

def run_pipeline():
    print("🚀 Starting Automated Training Pipeline...")

    # 1. Get Data from GitHub
    df = get_live_dataset()
    
    # 2. Add Engineered Features
    df = add_medical_features(df)

    # 3. Define Features & Target
    # We exclude 'Status' (Target) and 'Stage' (due to its low accuracy)
    X = df.drop(columns=['Status', 'Stage'], errors='ignore')
    y = df['Status'].astype(int)

    # 4. Preprocessing Layers
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # 5. XGBoost Model Configuration (Tuned for 125/187 ratio)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=250,
            learning_rate=0.03,
            max_depth=4,
            scale_pos_weight=1.5, # Balance for your specific dataset
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42
        ))
    ])

    # 6. Stratified Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 7. Model Training
    print("⚙️ Training model on remote data...")
    model.fit(X_train, y_train)

    # 8. Evaluation
    y_pred = model.predict(X_test)
    print("\n📈 Final Validation Results:")
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print("-" * 30)
    print(classification_report(y_test, y_pred))

    # 9. Global Save
    print(f"💾 Saving model locally: {MODEL_FILENAME}")
    model.fit(X, y) # Retrain on full dataset
    joblib.dump(model, MODEL_FILENAME)
    print("✅ Done! The model is now updated and ready for deployment.")

if __name__ == "__main__":
    run_pipeline()
