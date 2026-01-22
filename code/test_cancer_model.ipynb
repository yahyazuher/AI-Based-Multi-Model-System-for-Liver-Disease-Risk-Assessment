import pandas as pd
import joblib
import os

def load_model():
    """
    Locates and loads the trained XGBoost model file.
    Default search: 'cancer_model.pkl'.
    """
    # The filename must match the output from the training script
    model_filename = 'cancer_model.pkl' 
    
    if os.path.exists(model_filename):
        return joblib.load(model_filename)
    else:
        # Fallback: search for any available .pkl file in the root directory
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if pkl_files:
            print(f"Loading detected model: {pkl_files[0]}")
            return joblib.load(pkl_files[0])
        else:
            raise FileNotFoundError(f"Error: Model file '{model_filename}' not found.")

if __name__ == "__main__":
    # Initialize model
    try:
        model = load_model()
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit()

    # Feature columns defined during the training phase (XGBoost requirement)
    # Order: ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']
    columns = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

    print("\n" + "="*85)
    print(" VIRTUAL CLINIC: MODEL VALIDATION THROUGH CLINICAL SCENARIOS")
    print("="*85)

    # 7 diverse cases to test model logic and weights
    cases = [
        {'Case': '1. Healthy Athletic Young Male', 'Data': [25, 0, 22, 0, 0, 9, 0, 0]},
        {'Case': '2. Heavy Smoker (Chronic Exposure)', 'Data': [55, 1, 29, 1, 0, 2, 5, 0]},
        {'Case': '3. Genetic Risk vs. Ideal Lifestyle', 'Data': [30, 0, 24, 0, 2, 5, 1, 0]},
        {'Case': '4. High-Risk Multi-Factor Case', 'Data': [68, 1, 35, 1, 2, 0, 5, 1]},
        {'Case': '5. Severe Obesity (No Smoking/Alcohol)', 'Data': [45, 1, 40, 0, 0, 1, 0, 0]},
        {'Case': '6. Healthy Elderly (Age-Bias Check)', 'Data': [80, 0, 23, 0, 0, 6, 0, 0]},
        {'Case': '7. Borderline / Moderate Risk Profile', 'Data': [50, 1, 27, 0, 1, 3, 2, 0]}
    ]

    # CLI Table Header
    print(f"{'Clinical Scenario':<45} | {'Diagnosis':<15} | {'Risk Probability'}")
    print("-" * 85)

    for case in cases:
        # Prepare input data as a DataFrame for prediction
        df_test = pd.DataFrame([case['Data']], columns=columns)

        # Generate prediction (0 = Healthy, 1 = High Risk)
        prediction = model.predict(df_test)[0]
        
        # Calculate probability percentage for the positive class (index 1)
        probability = model.predict_proba(df_test)[0][1]

        # UI Formatting
        result_text = "🔴 HIGH RISK" if prediction == 1 else "🟢 HEALTHY"
        prob_text = f"{probability*100:.2f}%"

        print(f"{case['Case']:<45} | {result_text:<15} | {prob_text}")

    print("-" * 85)
    print("Technical Note: GeneticRisk levels are encoded as (0=Low, 1=Medium, 2=High).")
    print("="*85)
