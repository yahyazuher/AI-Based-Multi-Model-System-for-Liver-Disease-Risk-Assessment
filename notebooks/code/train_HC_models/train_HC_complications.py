"""
    - Source: HepatitisC.csv (Processed)
    - Target: Ascites (0 = No, 1 = Yes)

Author: Yahya Zuher
Project: AI-Based Multi-Model System for Liver Disease Risk Assessment
"""

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import requests
import io
import sys

# --- Configuration ---
DATA_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/data/processed/HepatitisC.csv'
MODEL_FILENAME = 'hepatitis_complications.pkl'

def load_data():
    """Fetches the dataset directly from the GitHub repository."""
    print(f" Downloading dataset from GitHub...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        print(f" Dataset loaded successfully: {len(df)} records.")
        return df
    except Exception as e:
        sys.exit(f" Error downloading data: {e}")

def train_model():
    # 1. Load Data
    df = load_data()

    print("\n Starting Complications Model Training...")

    # 2. Feature Selection
    # Dropping 'Stage' and 'Status' to ensure the model relies only on raw biomarkers
    features_to_drop = ['Ascites', 'Status', 'Stage', 'ID', 'N_Days']
    
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['Ascites']

    print(f" Features used ({len(X.columns)}): {list(X.columns)}")

    # 3. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Handle Class Imbalance (Calculate Ratio)
    # This ensures the model pays attention to the minority class (Ascites cases)
    ratio = float(len(y_train[y_train == 0])) / len(y_train[y_train == 1])

    # 5. Initialize XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=150,       # Number of trees
        learning_rate=0.05,     # Step size shrinkage
        max_depth=4,            # Tree depth (prevent overfitting)
        subsample=0.8,          # Data sampling ratio per tree
        colsample_bytree=0.8,   # Feature sampling ratio
        scale_pos_weight=ratio, # Automatically balance classes
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )

    # 6. Train
    print(" Training XGBoost model...")
    model.fit(X_train, y_train)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    
    print("-" * 40)
    print(f" Accuracy: {acc:.2f}%")
    print("-" * 40)
    print(classification_report(y_test, y_pred, target_names=['No Ascites', 'Ascites']))

    # 8. Save
    print(f" Saving model to {MODEL_FILENAME}...")
    joblib.dump(model, MODEL_FILENAME)
    print(f" Model saved successfully.")

if __name__ == "__main__":
    train_model()
