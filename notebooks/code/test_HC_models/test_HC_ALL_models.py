"""
AI Liver Disease Diagnosis System (Final Inference Script)
--------------------------------------------------------
Created by: Yahya Zuher
Project: AiLDS (AI-Based Multi-Model System)

Description: 
    Synchronized diagnostic script for AiLDS models.
    - FEATURES: Auto-downloads missing models from GitHub.
    - FIXES: Corrected filename 'hepatitisC_complications.pkl'.
    - RUNS: Inference on standard test cases.
"""

import pandas as pd
import joblib
import numpy as np
import math
import os
import sys
import requests

class LiverDiseasePredictor:
    def __init__(self, model_path='models'):
        self.model_path = model_path
        self.models = {}
        # Base URL for Raw GitHub Files (Used for auto-download)
        self.repo_url = "https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/models/"
        
        # Standard 15-column input structure
        self.raw_input_cols = [
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
            'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex',
            'Ascites', 'Hepatomegaly', 'Spiders', 'Edema'
        ]

    def _download_file(self, filename):
        """Helper to download a missing model file directly from GitHub."""
        url = self.repo_url + filename
        local_path = os.path.join(self.model_path, filename)
        
        print(f"  Model '{filename}' missing locally. Downloading from GitHub...")
        try:
            os.makedirs(self.model_path, exist_ok=True)
            response = requests.get(url)
            response.raise_for_status()
            with open(local_path, 'wb') as f:
                f.write(response.content)
            print(f" Downloaded: {local_path}")
            return True
        except Exception as e:
            print(f" Failed to download {filename}: {e}")
            return False

    def load_models(self):
        """Loads models. If a file is missing, it attempts to download it."""
        # UPDATED: Exact filenames as per your GitHub repo
        filenames = {
            'stage': 'hepatitisC_stage_model.pkl',
            'status': 'hepatitisC_status_model.pkl',
            'comp': 'hepatitisC_complications.pkl'  # Corrected Name
        }
        
        print(f"Initializing AiLDS Models...")
        all_loaded = True
        
        for key, name in filenames.items():
            path = os.path.join(self.model_path, name)
            
            # 1. Check existence, if not -> Download
            if not os.path.exists(path):
                success = self._download_file(name)
                if not success:
                    all_loaded = False
                    continue
            
            # 2. Load Model
            try:
                self.models[key] = joblib.load(path)
            except Exception as e:
                print(f"Error loading {name}: {e}")
                all_loaded = False

        if all_loaded:
            print("All AiLDS models loaded and synchronized.\n")
            return True
        else:
            print("Critical Error: One or more models could not be loaded.\n")
            return False

    def _prepare_dataframes(self, row):
        """
        Calculates medical indices and constructs specific DataFrames.
        """
        # 1. Feature Engineering
        apri = ((row['SGOT'] / 40.0) / (row['Platelets'] + 0.1)) * 100
        bili_adj = max(row['Bilirubin'], 0.1)
        albi = (math.log10(bili_adj * 17.1) * 0.66) + (row['Albumin'] * 10 * -0.085)
        
        bili_alb = row['Bilirubin'] * row['Albumin']
        copper_plat = row['Copper'] / (row['Platelets'] + 1)
        bili_alb_ratio = row['Bilirubin'] / (row['Albumin'] + 0.1)

        data = row.to_dict()
        data['APRI'] = apri
        data['ALBI_Score'] = albi
        data['Bilirubin_Albumin'] = bili_alb
        data['Copper_Platelets'] = copper_plat
        data['Bili_Alb_Ratio'] = bili_alb_ratio
        data['Status'] = 0 

        # 2. Construct Model-Specific DataFrames
        
        # A. Stage Model (19 Features)
        stage_cols = [
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
            'Tryglicerides', 'Platelets', 'Prothrombin', 'Status', 'Age', 'Sex',
            'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI',
            'Bilirubin_Albumin', 'Copper_Platelets'
        ]
        df_stage = pd.DataFrame([data], columns=stage_cols)

        # B. Status Model (18 Features)
        status_cols = [
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
            'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex',
            'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI',
            'ALBI_Score', 'Bili_Alb_Ratio'
        ]
        df_status = pd.DataFrame([data], columns=status_cols)

        # C. Complications Model (14 Features)
        comp_cols = [
            'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT',
            'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex',
            'Hepatomegaly', 'Spiders', 'Edema'
        ]
        df_comp = pd.DataFrame([data], columns=comp_cols)

        return df_stage, df_status, df_comp, apri, albi

    def run_diagnosis(self, patients_list):
        if not self.models and not self.load_models():
            return

        df_input = pd.DataFrame(patients_list, columns=self.raw_input_cols)

        for i in range(len(df_input)):
            patient_row = df_input.iloc[i]
            df_stage, df_status, df_comp, apri, albi = self._prepare_dataframes(patient_row)

            # --- INFERENCE ---
            # 1. Stage
            stage_pred = self.models['stage'].predict(df_stage)[0]
            if stage_pred == 0: stage_pred = 1 # Correction map
            
            # 2. Ascites Risk
            ascites_risk = self.models['comp'].predict_proba(df_comp)[:, 1][0]

            # 3. Mortality Risk
            death_risk = self.models['status'].predict_proba(df_status)[:, 1][0]

            # --- REPORT ---
            print(f"Case #{i+1} | AI Clinical Report")
            print("-" * 50)
            print(f"Indices:      APRI: {apri:.2f} | ALBI: {albi:.2f}")
            print(f"AI Stage:     Stage {stage_pred} (Histological)")
            print(f"Ascites Risk: {ascites_risk*100:.1f}%")
            print(f"Survival Risk:{death_risk*100:.1f}%")

            if death_risk > 0.5:
                print(" ASSESSMENT:   CRITICAL - Immediate intervention required.")
            elif ascites_risk > 0.5:
                print(" ASSESSMENT:   WARNING - High risk of decompensation.")
            else:
                print(" ASSESSMENT:   STABLE - Continue routine monitoring.")
            print("=" * 50 + "\n")

if __name__ == "__main__":
    # Test Data (7 Cases)
    test_data = [
        [0.7, 242.0, 4.08, 73.0, 5890.0, 56.76, 118.0, 300.0, 10.6, 53.0, 1, 0, 0, 0, 0],   # Healthy
        [3.2, 562.0, 3.08, 79.0, 2276.0, 144.15, 88.0, 251.0, 11.0, 53.0, 0, 0, 0, 1, 0],   # Acute
        [1.1, 302.0, 4.14, 54.0, 7394.8, 113.52, 88.0, 221.0, 10.6, 58.0, 0, 0, 1, 1, 0],   # Fibrosis
        [0.6, 252.0, 3.83, 41.0, 843.0, 65.1, 83.0, 336.0, 11.4, 59.0, 1, 0, 1, 1, 0],      # Compensated
        [14.5, 261.0, 2.6, 156.0, 1718.0, 137.95, 172.0, 190.0, 12.2, 58.0, 0, 1, 1, 1, 1], # Decompensated
        [3.6, 236.0, 3.52, 94.0, 591.0, 82.15, 95.0, 71.0, 13.6, 53.0, 0, 0, 0, 1, 0],      # Active
        [1.8, 244.0, 2.54, 64.0, 6121.8, 60.63, 92.0, 183.0, 10.3, 70.0, 0, 1, 1, 1, 0.5]   # Geriatric
    ]

    predictor = LiverDiseasePredictor(model_path='models')
    predictor.run_diagnosis(test_data)
