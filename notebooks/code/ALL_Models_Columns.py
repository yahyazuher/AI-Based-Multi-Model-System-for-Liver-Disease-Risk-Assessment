"""
Description:
    This script automatically downloads ALL serialized machine learning models 
    from the AiLDS GitHub repository (including Cancer, Fatty Liver, Gate, and Hepatitis modules).
    
    It extracts and displays the exact feature signature (input columns) required 
    for each model to ensure strict alignment between the web interface and the AI backend.

Author: Yahya Zuher
Project: AI-Based Multi-Model System for Liver Disease Risk Assessment
"""

import joblib
import os
import requests
import sys

# ==========================================
# CONFIGURATION
# ==========================================
REPO_BASE_URL = "https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/models/"

# Registry of all models found in the 'models/' directory
MODEL_REGISTRY = {
    "1. Gate Model (Dispatcher)": "gate_model.pkl",
    "2. Liver Cancer Model": "cancer_model.pkl",
    "3. Fatty Liver Model": "fatty_liver_model.pkl",
    "4. Hepatitis C Stage Model": "hepatitisC_stage_model.pkl",
    "5. Hepatitis C Status Model": "hepatitisC_status_model.pkl",
    "6. Hepatitis Complications": "hepatitis_complications.pkl"
}

LOCAL_DIR = "models/"

# ==========================================
# CORE FUNCTIONS
# ==========================================

def fetch_model(filename):
    """Downloads the model from GitHub if it does not exist locally."""
    local_path = os.path.join(LOCAL_DIR, filename)
    url = REPO_BASE_URL + filename
    
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            print(f"  Downloading {filename}...")
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            return True
        except requests.exceptions.RequestException as e:
            print(f" Error downloading {filename}: {e}")
            return False
    return True

def get_feature_names(file_path):
    """
    Intelligently extracts feature names from various model types 
    (Sklearn Pipeline, XGBoost, RandomForest, etc.).
    """
    try:
        model = joblib.load(file_path)
        features = []

        # Case A: Scikit-Learn Estimators (Standard)
        if hasattr(model, 'feature_names_in_'):
            features = list(model.feature_names_in_)
        
        # Case B: Scikit-Learn Pipelines
        elif hasattr(model, 'named_steps'):
            # Check the last step (usually the classifier)
            step_name = 'classifier' if 'classifier' in model.named_steps else 'model'
            if step_name in model.named_steps:
                clf = model.named_steps[step_name]
                if hasattr(clf, 'get_booster'): # XGBoost inside Pipeline
                    features = clf.get_booster().feature_names
                elif hasattr(clf, 'feature_names_in_'): # Standard Sklearn inside Pipeline
                    features = list(clf.feature_names_in_)
        
        # Case C: Raw XGBoost Booster
        elif hasattr(model, 'get_booster'):
            features = model.get_booster().feature_names
            
        return features
    except Exception as e:
        return f"Error reading file: {str(e)}"

# ==========================================
# EXECUTION
# ==========================================

def run_full_audit():
    print(f"\n{'='*70}")
    print(f" AiLDS Full System Model Inspector")
    print(f"   Target Repository: {REPO_BASE_URL}")
    print(f"{'='*70}\n")

    for display_name, filename in MODEL_REGISTRY.items():
        local_path = os.path.join(LOCAL_DIR, filename)

        # 1. Download
        if not fetch_model(filename):
            continue

        # 2. Extract
        features = get_feature_names(local_path)

        # 3. Report
        print(f"   {display_name}")
        print(f"   File: {filename}")
        
        if isinstance(features, list) and features:
            print(f"   Input Features Required: {len(features)}")
            print(f"   Feature Order:")
            # Print features in a neat grid or list
            for i, feat in enumerate(features, 1):
                print(f"      {i:02d}. {feat}")
        else:
            print(f"     Warning: Could not extract features automatically.")
            print(f"       Debug Info: {features}")
        
        print(f"\n{'-'*70}\n")

if __name__ == "__main__":
    run_full_audit()
