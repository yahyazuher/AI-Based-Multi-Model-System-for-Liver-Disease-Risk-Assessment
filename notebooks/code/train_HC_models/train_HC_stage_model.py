"""
Project: AI-Based Multi-Model System for Liver Disease Risk Assessment
Description: Trains an XGBoost classifier to predict liver disease stages (1, 2, 3)
             using blood biomarkers and engineered features (APRI,Bilirubin × Albumin,Copper ÷ Platelets.).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import sys

# --- Configuration ---
DATASET_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/data/processed/hepatitisC_Stage.csv'
LOCAL_FILENAME = 'hepatitisC_Stage.csv'
MODEL_FILENAME = 'hepatitisC_stage_model.pkl'

COLUMN_NAMES = [
    'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos',
    'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin',
    'Stage', 'Status', 'Age', 'Sex', 'Ascites',
    'Hepatomegaly', 'Spiders', 'Edema'
]

def get_dataset():
    """Downloads dataset if not available locally."""
    if not os.path.exists(LOCAL_FILENAME):
        try:
            print("⬇Downloading dataset...")
            df = pd.read_csv(DATASET_URL)
            if len(df.columns) == len(COLUMN_NAMES):
                df.columns = COLUMN_NAMES
            df.to_csv(LOCAL_FILENAME, index=False)
        except Exception as e:
            sys.exit(f"Error downloading data: {e}")
    return pd.read_csv(LOCAL_FILENAME)

def add_medical_features(df):
    """Calculates APRI score and other medical interaction terms."""
    df_eng = df.copy()
    # APRI: AST to Platelet Ratio Index
    df_eng['APRI'] = ((df_eng['SGOT'] / 40.0) / (df_eng['Platelets'] + 0.1)) * 100
    # Liver Function Synthesis
    df_eng['Bilirubin_Albumin'] = df_eng['Bilirubin'] * df_eng['Albumin']
    # Copper/Platelet Ratio
    df_eng['Copper_Platelets'] = df_eng['Copper'] / (df_eng['Platelets'] + 1)
    return df_eng

def train():
    print("🚀 Starting Training Pipeline...")

    # 1. Load & Engineer Features
    df = get_dataset()
    df = add_medical_features(df)

    # 2. Prepare Data
    X = df.drop(columns=['Stage'])
    y = df['Stage']

    # Map Targets: 1->0 (Early), 2->1 (Fibrosis), 3->2 (Cirrhosis)
    y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
    y = y.map({1: 0, 2: 1, 3: 2}).astype(int)

    # 3. Preprocessing
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    # 4. XGBoost Model (Optimized Parameters)
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier(
            n_estimators=200,
            learning_rate=0.01,
            max_depth=3,
            min_child_weight=5,
            gamma=0.1,
            subsample=0.7,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42
        ))
    ])

    # 5. Train & Evaluate
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("⚙️ Training model...")
    model.fit(X_train, y_train)

    print("\n📈 Evaluation Results:")
    y_pred = model.predict(X_test)
    print(f"🎯 Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Stage 1', 'Stage 2', 'Stage 3']))

    # 6. Save Final Model
    print("💾 Retraining on full data and saving...")
    model.fit(X, y)
    joblib.dump(model, MODEL_FILENAME)
    print(f"✅ Saved: {MODEL_FILENAME}")

if __name__ == "__main__":
    train()
