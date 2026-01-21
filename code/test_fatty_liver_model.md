import pandas as pd
import joblib
import os

def load_model():
    """
    Locates and loads the pre-trained XGBoost model.
    """
    model_filename = 'FattyLiver_Model.pkl'
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    else:
        # Search for any available .pkl file if the specific filename is missing
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if pkl_files:
            return joblib.load(pkl_files[0])
        else:
            raise FileNotFoundError(f"Error: Model file '{model_filename}' not found.")

if __name__ == "__main__":
    try:
        model = load_model()
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit()

    # Strict Positional Logic: The model expects exactly 13 features in this specific order.
    # Disruption of this sequence will lead to diagnostic failure.
    columns = [
        'Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol',
        'Creatinine', 'Glucose', 'GGT', 'Bilirubin',
        'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL'
    ]

    print("\n" + "="*75)
    print("VIRTUAL CLINIC: FATTY LIVER (NAFLD) MODEL VALIDATION")
    print("="*75)

    # Defining 7 clinical scenarios to validate biological synergy logic.
    cases = [
        {'Case': '1. Healthy Baseline (Athletic)', 'Data': [4.5, 60, 20, 18, 170, 0.9, 85, 25, 0.6, 90, 4.5, 250, 55]},
        {'Case': '2. Isolated Hyperlipidemia', 'Data': [4.2, 70, 22, 20, 240, 1.0, 95, 30, 0.7, 300, 5.2, 230, 40]},
        {'Case': '3. Active NAFLD (Early)', 'Data': [3.8, 40, 45, 55, 210, 1.1, 110, 65, 0.8, 220, 6.5, 210, 35]},
        {'Case': '4. Metabolic Syndrome', 'Data': [3.5, 110, 65, 75, 280, 1.2, 145, 90, 1.1, 450, 8.2, 185, 28]},
        {'Case': '5. Advanced Stress', 'Data': [3.1, 140, 85, 80, 200, 1.3, 130, 110, 1.4, 190, 7.5, 130, 31]},
        {'Case': '6. Non-Fatty Injury', 'Data': [4.0, 65, 130, 160, 165, 0.8, 88, 40, 1.3, 105, 4.2, 245, 52]},
        {'Case': '7. Moderate/Borderline Risk', 'Data': [4.0, 40, 35, 41, 190, 1.0, 105, 39, 0.8, 152, 5.8, 210, 42]},
    ]

    print(f"{'Clinical Scenario':<45} | {'Final Diagnosis'}")
    print("-" * 75)

    for case in cases:
        # Transforming raw data into the required DataFrame structure for prediction
        df_test = pd.DataFrame([case['Data']], columns=columns)
        prediction = model.predict(df_test)[0]

        # Display outcome status (Patient vs. Healthy)
        result_text = "🔴 PATIENT (NAFLD)" if prediction == 1 else "🟢 HEALTHY"

        print(f"{case['Case']:<45} | {result_text}")

    print("-" * 75)
    print("Scientific Logic: Thresholds applied at ALT: 40, AST: 40, Triglycerides: 150.")
    print("="*75)
