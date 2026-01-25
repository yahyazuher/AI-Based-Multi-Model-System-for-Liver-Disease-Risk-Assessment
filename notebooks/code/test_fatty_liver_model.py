"""
[IMPORTANT NOTE / ملاحظة هامة]
--------------------------------------------------
English: This script is specifically designed and optimized to run in the GOOGLE COLAB environment.
- It is configured to automatically download models and training files directly from GitHub.
- Copy-pasting this code to other environments (local IDEs) may require adjustments 
  to file paths and library configurations.

Arabic: Google Colab هذا الكود مخصص ومجهز للعمل مباشرة داخل بيئة 
- GitHub لضمان التشغيل الفوري تم إعداد الكود ليقوم بتحميل النماذج وملفات التدريب تلقائياً من 
- نسخ هذا الكود وتشغيله في تطبيقات أو بيئات أخرى قد يتطلب تعديلات في مسارات الملفات وإعدادات المكتبات.
--------------------------------------------------
Created by: Yahya Zuher
Project: AI-Liver-Diseases-Diagnosis-System
"""
import pandas as pd
import joblib
import os
import requests
import io

def load_model():
    model_url = 'https://github.com/yahyazuher/AI-Liver-Diseases-Diagnosis-System/raw/main/models/fatty_liver_model.pkl'

    try:
        print(f"Connecting to GitHub to fetch model...")
        response = requests.get(model_url)

        # Check if the download was successful (HTTP 200)
        if response.status_code == 200:
            # Wrap the raw bytes in a BytesIO object so joblib can read it like a file
            return joblib.load(io.BytesIO(response.content))
        else:
            raise Exception(f"Failed to download model. Status code: {response.status_code}")

    except Exception as e:
        print(f"Cloud Load Failed: {e}")
        # Fallback: Try to find the file locally if the internet is down
        model_filename = 'fatty_liver_model.pkl'
        if os.path.exists(model_filename):
            print("Local backup found. Loading from disk...")
            return joblib.load(model_filename)
        else:
            raise FileNotFoundError("Critical Error: Model not found online or locally.")

if __name__ == "__main__":
    try:
        model = load_model()
        print("Model loaded successfully from GitHub!")
    except Exception as e:
        print(f"Initialization Error: {e}")
        exit()

    # Feature labels (13 mandatory clinical inputs)
    columns = [
        'Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol',
        'Creatinine', 'Glucose', 'GGT', 'Bilirubin',
        'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL'
    ]

    print("\n" + "="*75)
    print("VIRTUAL CLINIC: FATTY LIVER (NAFLD) MODEL VALIDATION")
    print("="*75)

    # Clinical scenarios for testing
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
        # Create DataFrame for prediction
        df_test = pd.DataFrame([case['Data']], columns=columns)

        # Get result (0 = Healthy, 1 = NAFLD)
        prediction = model.predict(df_test)[0]
        result_text = "🔴 PATIENT (NAFLD)" if prediction == 1 else "🟢 HEALTHY"

        print(f"{case['Case']:<45} | {result_text}")

    print("-" * 75)
    print("Scientific Logic: Thresholds applied at ALT: 40, AST: 40, Triglycerides: 150.")
    print("="*75)
