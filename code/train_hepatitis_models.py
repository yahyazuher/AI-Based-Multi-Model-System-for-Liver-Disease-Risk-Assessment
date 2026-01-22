"""
AI Liver Disease Diagnosis System
Created by: Yahya
Description: This script trains three XGBoost models with clean output logs.
"""

import os
import pickle
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from google.colab import files

class LiverDiseaseTrainer:
    def __init__(self, data_path='/content/Hepatitis.csv'):
        self.data_path = data_path
        self.base_path = '/content'
        self.df = None

    def upload_data(self):
        if not os.path.exists(self.data_path):
            print("Please upload 'Hepatitis.csv':")
            uploaded = files.upload()

        try:
            self.df = pd.read_csv(self.data_path, na_values=['', 'NA', 'nan'])
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f" Error loading data: {e}")
            raise

    def train_stage_model(self):
        print("\n--- Training: Hepatitis Stage Model ---")
        X = self.df.drop(columns=['Stage', 'Status', 'ID', 'N_Days'], errors='ignore')
        y = self.df['Stage']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Removed: use_label_encoder=False (Deprecated)
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=len(le.classes_),
            eval_metric='mlogloss'
        )
        model.fit(X, y_encoded)
        self._save_model(model, 'hepatitis_stage.pkl')

    def train_complications_model(self):
        print("\n--- Training: Hepatitis Complications Model ---")
        X = self.df.drop(columns=['Ascites', 'Status', 'ID', 'N_Days', 'Stage'], errors='ignore')
        y = self.df['Ascites']

        # Removed: use_label_encoder=False
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X, y)
        self._save_model(model, 'hepatitis_complications.pkl')

    def train_status_model(self):
        print("\n--- Training: Hepatitis Status Model ---")
        X = self.df.drop(columns=['Status', 'ID', 'N_Days'], errors='ignore')
        y = self.df['Status']

        # Removed: use_label_encoder=False
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X, y)
        self._save_model(model, 'hepatiti_status.pkl')

    def _save_model(self, model, filename):
        path = os.path.join(self.base_path, filename)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        print(f" Model saved as: {filename}")

if __name__ == "__main__":
    trainer = LiverDiseaseTrainer()
    trainer.upload_data()
    trainer.train_stage_model()
    trainer.train_complications_model()
    trainer.train_status_model()
    print("\nAll models trained and saved successfully with clean logs!")
