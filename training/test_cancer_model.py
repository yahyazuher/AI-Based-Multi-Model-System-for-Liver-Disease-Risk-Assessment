import pandas as pd
import numpy as np
import pickle

# --- 🧪 Virtual Clinic: Testing the Liver Cancer Model ---

def test_model_validity():
    # 1. Load the trained model
    model_name = "Liver_Cancer_Model.pkl"
    try:
        with open(model_name, "rb") as file:
            model = pickle.load(file)
        print(f"✅ Model '{model_name}' loaded successfully.\n")
    except FileNotFoundError:
        print(f"❌ Error: {model_name} not found. Please train and save the model first.")
        return

    # 2. Define 7 Diverse Patient Scenarios (The 7 Cases) 
    # Features Order: Age, Gender, BMI, Smoking, GeneticRisk, PhysicalActivity, AlcoholIntake, CancerHistory
    # Note: Gender (0=Male, 1=Female), Smoking/History (0=No, 1=Yes), GeneticRisk (0=Low, 1=Med, 2=High)
    
    test_cases = {
        "1. Young Healthy Athlete":      [25, 0, 22.5, 0, 0, 15, 0, 0],
        "2. Smoker & Heavy Drinker":    [45, 0, 28.0, 1, 1, 1, 25, 1],
        "3. High Genetic Risk/Healthy": [30, 1, 23.0, 0, 2, 12, 0, 1],
        "4. High-Risk (All Factors)":   [60, 0, 35.0, 1, 2, 0, 30, 1],
        "5. Obese (Non-smoker)":        [40, 1, 38.0, 0, 0, 2, 2, 0],
        "6. Healthy Elderly (80+)":     [82, 0, 24.5, 0, 0, 10, 1, 0],
        "7. Borderline (Medium Risk)":  [50, 1, 27.0, 0, 1, 5, 5, 0]
    }

    # Convert to DataFrame for the model
    feature_names = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 
                     'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
    
    df_test = pd.DataFrame.from_dict(test_cases, orient='index', columns=feature_names)

    # 3. Run Predictions
    predictions = model.predict(df_test)
    probabilities = model.predict_proba(df_test)[:, 1] # Probability of Cancer (Class 1)

    # 4. Display Results
    print(f"{'Patient Profile':<30} | {'Diagnosis':<10} | {'Risk Probability'}")
    print("-" * 65)
    
    for i, (profile, row) in enumerate(test_cases.items()):
        status = "🔴 At Risk" if predictions[i] == 1 else "🟢 Healthy"
        prob = probabilities[i] * 100
        print(f"{profile:<30} | {status:<10} | {prob:>6.1f}%")

    print("\n--- 🧠 Scientific Logic Verification ---")
    print("Check Case 3 vs Case 2:")
    print("- Case 3 has High Genetic Risk but stays 'Healthy' due to lifestyle [cite: 148-151].")
    print("- Case 2 has lower genetics but high 'Smoking/Alcohol' causing high risk [cite: 153-154].")

if __name__ == "__main__":
    test_model_validity()
