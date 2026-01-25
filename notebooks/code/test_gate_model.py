import pandas as pd
import joblib
import os
import sys
import requests

# Configuration
MODEL_FILENAME = 'gate_model.pkl'

# Direct link to the raw binary model file
GITHUB_MODEL_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/models/gate_model.pkl'

def download_model_if_missing():
    """
    Checks if the model exists locally. If not, downloads the pre-trained .pkl file
    directly from the GitHub repository.
    """
    if not os.path.exists(MODEL_FILENAME):
        print(f"Model '{MODEL_FILENAME}' not found locally.")
        print(f"Downloading pre-trained model from GitHub...")
        
        try:
            response = requests.get(GITHUB_MODEL_URL)
            response.raise_for_status() # Check for HTTP errors
            
            with open(MODEL_FILENAME, 'wb') as f:
                f.write(response.content)
            
            print("Download successful. Model saved locally.")
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            sys.exit(1)
    else:
        print(f"Loading local model: {MODEL_FILENAME}")

def run_prediction_tests():
    # 1. Ensure Model is Available
    download_model_if_missing()
    
    try:
        model = joblib.load(MODEL_FILENAME)
    except Exception as e:
        print(f"Error loading model file: {e}")
        sys.exit(1)

    # 2. Define Test Data (10 Cases)
    # Feature Order: [Age, Gender, TB, DB, Alkphos, Sgpt, Sgot, TP, ALB, A/G Ratio]
    # Gender Encoding: Male=1, Female=0
    new_patients_data = [
        [45, 1, 8.5, 4.5, 400, 200, 180, 6.8, 3.0, 0.80], # Case 1: Sick (High enzymes/bilirubin)
        [60, 1, 2.5, 1.2, 200, 45,  60,  5.0, 1.8, 0.50], # Case 2: Sick (Low albumin)
        [30, 0, 15.0, 8.0, 550, 120, 110, 7.0, 3.2, 0.80],# Case 3: Sick (Acute hepatitis signs)
        [50, 1, 3.0, 1.5, 290, 80,  250, 6.0, 2.8, 0.85], # Case 4: Sick (Alcoholic liver damage pattern)
        [40, 1, 1.5, 0.6, 190, 150, 140, 7.2, 3.8, 1.10], # Case 5: Sick (Fatty liver signs)
        [75, 0, 5.2, 2.8, 600, 40,  80,  4.5, 1.5, 0.50], # Case 6: Sick (Critical condition)
        [25, 1, 0.7, 0.1, 150, 20,  22,  7.5, 4.0, 1.10], # Case 7: Healthy
        [35, 0, 0.9, 0.2, 180, 25,  19,  6.8, 3.5, 1.00], # Case 8: Borderline (Slightly elevated Alkphos)
        [65, 1, 1.0, 0.3, 195, 35,  40,  7.0, 3.2, 0.90], # Case 9: Borderline (Elevated Alkphos for age)
        [18, 0, 0.6, 0.1, 140, 15,  18,  7.8, 4.2, 1.20]  # Case 10: Healthy
    ]

    # 3. DataFrame Creation
    # Uses feature_names_in_ to ensure compatibility with the trained model
    try:
        correct_columns = model.feature_names_in_
        patients_df = pd.DataFrame(new_patients_data, columns=correct_columns)
    except AttributeError:
        # Fallback if attribute is missing
        cols = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 
                'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
                'Albumin', 'Albumin_and_Globulin_Ratio']
        patients_df = pd.DataFrame(new_patients_data, columns=cols)

    # 4. Prediction
    print("\nRunning diagnostics on test cases...")
    predictions = model.predict(patients_df)

    # 5. Result Display
    print("-" * 75)
    print(f"{'Case':<5} | {'Expected':<12} | {'Prediction':<22} | {'Result':<10}")
    print("-" * 75)

    for i, result in enumerate(predictions):
        # Determine Label: 0 = Sick, 1 = Healthy (based on LabelEncoder logic)
        if result == 0:
            status_label = "Liver Patient (Sick)"
            icon = "🔴"
        else:
            status_label = "Healthy"
            icon = "🟢"
        
        # Validation Logic
        if i < 6:
            # Cases 1-6 are clearly Sick
            expected = "Sick"
            is_correct = "PASS" if result == 0 else "FAIL"
            
        elif i == 7 or i == 8: 
            # Cases 8 and 9 (Indices 7 & 8) are medically borderline
            expected = "Very Close"
            is_correct = "BORDERLINE" 
            
        else:
            # Cases 7 and 10 are clearly Healthy
            expected = "Healthy"
            is_correct = "PASS" if result == 1 else "FAIL"
        
        print(f"{i+1:<5} | {expected:<12} | {icon} {status_label:<20} | {is_correct}")
    print("-" * 75)

if __name__ == "__main__":
    run_prediction_tests()
