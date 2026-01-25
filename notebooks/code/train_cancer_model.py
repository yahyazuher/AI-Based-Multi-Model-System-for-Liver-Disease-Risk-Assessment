import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# =================================================================
# AiLDS: Liver Cancer Risk Assessment Module (XGBoost)
# Project: AI-Based Multi-Model System for Liver Disease Risk Assessment
# Author: Yahya Zuher
# =================================================================

print("--- Initializing Liver Cancer Risk Assessment System ---")


DATA_URL = "https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/data/processed/The_Cancer_data_1500.csv"

try:
    print(f"Fetching dataset from: {DATA_URL}")
    df = pd.read_csv(DATA_URL)
    print("Dataset loaded successfully.")
except Exception as e:
    print(f"Error: Failed to load dataset. {e}")
    raise SystemExit

# Data Cleaning: Ensure no missing values before processing
df = df.dropna()
print(f"Total records available for processing: {len(df)}")

# ---------------------------------------------------------
# 2. Feature Engineering & Preprocessing
# ---------------------------------------------------------
# X: Feature matrix (Age, Smoking, Genetics, Alcohol, etc.)
# y: Target vector (Diagnosis: 0 = Healthy, 1 = Cancer)
X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

print("\nIdentified Features for Model Input:")
print(list(X.columns))

# Split data: 80% Training - 20% Testing for validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. Model Training (XGBoost Classifier)
# ---------------------------------------------------------
print("\nTraining XGBoost model on diagnostic patterns...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.8,
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Evaluation & Performance Metrics
# ---------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Performance Metrics:")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
print("-" * 40)
print("Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Visualizing results via Confusion Matrix

plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Confusion Matrix - Cancer Prediction Model')
plt.xlabel('Predicted Diagnosis')
plt.ylabel('Actual Diagnosis')
plt.show()

# ---------------------------------------------------------
# 5. Model Export for Integration
# ---------------------------------------------------------
# Exporting as a pickle file for use in the AiLDS web application
MODEL_FILENAME = "cancer_model.pkl"
with open(MODEL_FILENAME, "wb") as file:
    pickle.dump(model, file)

print(f"\n✔ Model successfully serialized as: {MODEL_FILENAME}")
