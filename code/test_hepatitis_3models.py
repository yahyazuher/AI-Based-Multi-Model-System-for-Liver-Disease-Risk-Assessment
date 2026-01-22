"""
AI Liver Disease Diagnosis System (Inference Script)
Created by: Yahya
Description: This script loads trained XGBoost models to perform a full 
             diagnostic sweep for liver patients, including clinical scores 
             (APRI/ALBI), AI staging, and survival risk.
"""

import pandas as pd
import pickle
import numpy as np
import math
import os

class LiverDiseasePredictor:
    def __init__(self, model_path='/content'):
        self.model_path = model_path
        self.models = {}
        # Define the exact feature order expected by the models
        self.input_cols = [
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
            'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex',
            'Ascites', 'Hepatomegaly', 'Spiders', 'Edema'
        ]
        self.status_order = [
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
            'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage', 'Age', 'Sex',
            'Ascites', 'Hepatomegaly', 'Spiders', 'Edema'
        ]

    def load_models(self):
        """Loads the three saved .pkl models from the specified path."""
        filenames = {
            'stage': 'hepatitis_stage.pkl',
            'comp': 'hepatitis_complications.pkl',
            'status': 'hepatiti_status.pkl'
        }
        try:
            for key, name in filenames.items():
                with open(os.path.join(self.model_path, name), 'rb') as f:
                    self.models[key] = pickle.load(f)
            print("✅ All models loaded successfully from environment.\n")
            return True
        except FileNotFoundError as e:
            print(f"❌ Error: Model files not found. Please ensure .pkl files are in {self.model_path}")
            return False

    @staticmethod
    def calculate_clinical_scores(row):
        """Calculates traditional medical scores: APRI and ALBI."""
        # APRI Calculation
        ast = row['SGOT']
        plat = row['Platelets'] if row['Platelets'] > 0 else 1
        apri = ((ast / 40) / plat) * 100

        # ALBI Calculation
        bili = row['Bilirubin'] if row['Bilirubin'] > 0 else 0.1
        alb = row['Albumin']
        albi = (math.log10(bili * 17.1) * 0.66) + ((alb * 10) * -0.085)

        return apri, albi

    def run_diagnosis(self, patients_list):
        """Processes a list of patients and prints detailed AI diagnostic reports."""
        if not self.models and not self.load_models():
            return

        df_test = pd.DataFrame(patients_list, columns=self.input_cols)

        for i in range(len(df_test)):
            patient_row = df_test.iloc[[i]]
            patient_data = df_test.iloc[i]

            print(f"🔷 Patient Case #{i+1}")
            print("-" * 40)

            # 1. Clinical Scores
            apri, albi = self.calculate_clinical_scores(patient_data)
            apri_status = "Healthy" if apri < 0.5 else "Likely Cirrhosis" if apri > 1.5 else "Inconclusive"
            
            if albi <= -2.60: albi_grade = "Grade 1 (Excellent)"
            elif albi <= -1.39: albi_grade = "Grade 2 (Moderate)"
            else: albi_grade = "Grade 3 (Severe Failure)"

            print(f"1️⃣ Clinical Scores:")
            print(f"   • APRI: {apri:.2f} ({apri_status})")
            print(f"   • ALBI: {albi:.2f} ({albi_grade})")

            # 2. AI Stage Diagnosis
            stage_pred = self.models['stage'].predict(patient_row)[0]
            stage = stage_pred + 1
            print(f"2️⃣ AI Diagnosis: Stage {stage}")

            # 3. Complications Risk (Ascites)
            row_no_ascites = patient_row.drop(columns=['Ascites'], errors='ignore')
            ascites_risk = self.models['comp'].predict_proba(row_no_ascites)[:, 1][0]
            print(f"3️⃣ Ascites Risk: {ascites_risk*100:.1f}%")

            # 4. Survival Probability
            row_status = patient_row.copy()
            row_status['Stage'] = stage
            row_status = row_status[self.status_order] # Reorder to match training
            
            death_risk = self.models['status'].predict_proba(row_status)[:, 1][0]
            print(f"4️⃣ Survival Risk Analysis: {death_risk*100:.1f}% Mortality Probability")

            if death_risk > 0.5:
                print("   🔴 SUMMARY: CRITICAL CASE - Immediate intervention required.")
            else:
                print("   🟢 SUMMARY: STABLE - Routine monitoring recommended.")

            print("=" * 40 + "\n")

if __name__ == "__main__":
    # Sample Dataset: 7 Patients for testing
    test_data = [
        [0.7, 242.0, 4.08, 73.0, 5890.0, 56.76, 118.0, 300.0, 10.6, 53.0, 1, 0, 0, 0, 0],
        [3.2, 562.0, 3.08, 79.0, 2276.0, 144.15, 88.0, 251.0, 11.0, 53.0, 0, 0, 0, 1, 0],
        [1.1, 302.0, 4.14, 54.0, 7394.8, 113.52, 88.0, 221.0, 10.6, 58.0, 0, 0, 1, 1, 0],
        [0.6, 252.0, 3.83, 41.0, 843.0, 65.1, 83.0, 336.0, 11.4, 59.0, 1, 0, 1, 1, 0],
        [14.5, 261.0, 2.6, 156.0, 1718.0, 137.95, 172.0, 190.0, 12.2, 58.0, 0, 1, 1, 1, 1],
        [3.6, 236.0, 3.52, 94.0, 591.0, 82.15, 95.0, 71.0, 13.6, 53.0, 0, 0, 0, 1, 0],
        [1.8, 244.0, 2.54, 64.0, 6121.8, 60.63, 92.0, 183.0, 10.3, 70.0, 0, 1, 1, 1, 0.5]
    ]

    predictor = LiverDiseasePredictor()
    predictor.run_diagnosis(test_data)
