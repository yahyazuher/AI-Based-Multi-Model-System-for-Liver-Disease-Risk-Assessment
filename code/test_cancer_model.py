import pandas as pd
import joblib
import os
import requests
import io

def load_model():
    """
    Downloads and loads the trained XGBoost model from GitHub.
    """
    # Direct RAW link to the cancer model on GitHub
    model_url = 'https://github.com/yahyazuher/AI-Liver-Diseases-Diagnosis-System/raw/main/models/cancer_model.pkl'
    
    try:
        print(f"Connecting to GitHub to fetch: {model_url.split('/')[-1]}...")
        response = requests.get(model_url)
        
        # Check if the download was successful
        if response.status_code == 200:
            # Use BytesIO to let joblib read the binary content directly from RAM
            return joblib.load(io.BytesIO(response.content))
        else:
            raise Exception(f"Download failed. Status code: {response.status_code}")
            
    except Exception as e:
        print(f"Cloud Load Failed: {e}")
        # Fallback: Check if the file exists locally
        model_filename = 'cancer_model.pkl'
        if os.path.exists(model_filename):
            print("Local backup found. Loading from disk...")
            return joblib.load(model_filename)
        else:
            raise FileNotFoundError("Critical Error: Cancer model not found online or locally.")

if __name__ == "__main__":
    # Initialize model
    try:
        model = load_model()
        print("Cancer Model loaded successfully!")
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit()

    # Feature columns defined during the training phase
    columns = ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

    print("\n" + "="*85)
    print(" VIRTUAL CLINIC: CANCER MODEL VALIDATION THROUGH CLINICAL SCENARIOS")
    print("="*85)

    # 7 diverse cases to test model logic
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
        df_test = pd.DataFrame([case['Data']], columns=columns)

        # Generate prediction and probability
        prediction = model.predict(df_test)[0]
        probability = model.predict_proba(df_test)[0][1]

        # UI Formatting
        result_text = "🔴 HIGH RISK" if prediction == 1 else "🟢 HEALTHY"
        prob_text = f"{probability*100:.2f}%"

        print(f"{case['Case']:<45} | {result_text:<15} | {prob_text}")

    print("-" * 85)
    print("Technical Note: GeneticRisk levels are encoded as (0=Low, 1=Medium, 2=High).")
    print("="*85)
