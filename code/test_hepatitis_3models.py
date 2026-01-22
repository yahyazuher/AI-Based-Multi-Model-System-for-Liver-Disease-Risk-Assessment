import pandas as pd
import pickle
import numpy as np

# --- 1. Load Pre-trained Models ---
try:
    model_stage = pickle.load(open("hepatitis_stage.pkl", "rb"))
    model_comp = pickle.load(open("hepatitis_complications.pkl", "rb"))
    model_status = pickle.load(open("hepatitis_status.pkl", "rb"))
    print("✅ Models loaded successfully.\n")
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found. {e}")
    exit()

# --- 2. Dataset Structure Definitions ---
# Order must match Hepatitis.csv (17 columns)
all_columns = [
    'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 
    'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage', 
    'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'Status'
]

# --- 3. Test Cases Configuration ---
# Format: [Bili, Chol, Alb, Cop, Alk, SGOT, Tryg, Plat, Proth, Stage, Age, Sex, Asci, Hepato, Spid, Edem, Stat]
test_data = [
    [0.8, 175, 4.5, 35, 650, 45, 80, 400, 10.1, 1, 30, 0, 0, 0, 0, 0, 0],       # Case 1: Healthy/Early
    [2.8, 260, 3.2, 90, 1300, 105, 180, 190, 12.1, 3, 50, 1, 0, 1, 1, 0.5, 0],  # Case 2: Intermediate
    [14.5, 420, 2.1, 260, 2600, 190, 290, 85, 16.2, 4, 65, 1, 1, 1, 1, 1.0, 1]  # Case 3: Critical
]

df_test = pd.DataFrame(test_data, columns=all_columns)

def run_inference(case_idx):
    """Executes prediction across the triple-model suite."""
    row = df_test.iloc[[case_idx]]
    print(f"--- Inference Results: Case {case_idx + 1} ---")

    # Model 1: Stage Prediction (Features exclude Target & Status)
    features_stage = row.drop(columns=['Stage', 'Status'])
    pred_stage = model_stage.predict(features_stage)[0] + 1
    print(f"Prediction [Stage]: {int(pred_stage)}")

    # Model 2: Complications (Ascites) Prediction (Features exclude Ascites, Status, & Stage)
    features_comp = row.drop(columns=['Ascites', 'Status', 'Stage'])
    prob_comp = model_comp.predict_proba(features_comp)[0][1]
    label_comp = "Positive" if prob_comp > 0.5 else "Negative"
    print(f"Prediction [Ascites]: {label_comp} ({prob_comp:.1%})")

    # Model 3: Mortality Risk (Status) (Features exclude Status)
    features_status = row.drop(columns=['Status'])
    prob_status = model_status.predict_proba(features_status)[0][1]
    label_status = "High Risk" if prob_status > 0.5 else "Low Risk"
    print(f"Prediction [Status]: {label_status} ({prob_status:.1%})")
    print("-" * 40)

# --- 4. Main Execution ---
if __name__ == "__main__":
    for i in range(len(df_test)):
        run_inference(i)
