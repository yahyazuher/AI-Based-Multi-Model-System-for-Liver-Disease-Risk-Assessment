import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# =================================================================
# AiLDS: Fatty Liver (NAFLD) Diagnostic Module
# Project: AI-Based Multi-Model System for Liver Disease Risk Assessment
# Author: Yahya Zuher
# =================================================================

print("--- Initializing Fatty Liver (NAFLD) Diagnostic System ---")


RAW_DATA_URL = "https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/data/processed/FattyLiver.csv"

try:
    print(f"Accessing remote repository: {RAW_DATA_URL}")
    df = pd.read_csv(RAW_DATA_URL)
    print(f"✔ Dataset synchronized successfully. Total records: {len(df)}")
except Exception as e:
    print(f"✘ Error: Connection to repository failed. {e}")
    raise SystemExit

# ---------------------------------------------------------
# 2. Data Preprocessing & Cleaning
# ---------------------------------------------------------
# Sanitizing column headers and ensuring numeric integrity
df.columns = df.columns.str.strip()
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Removing incomplete clinical records
df = df.dropna()
print(f"Cleaned dataset ready for training: {len(df)} samples")

# ---------------------------------------------------------
# 3. Clinical Target Engineering (NAFLD Logic)
# ---------------------------------------------------------
def clinical_diagnosis_logic(row):
    """
    Logic: Positive diagnosis if (Triglycerides > 150 mg/dL AND (ALT > 40 U/L OR GGT > 40 U/L))
    or if both liver enzymes (ALT & GGT) are significantly elevated.
    """
    trig_high = row['Triglycerides'] > 150
    alt_high = row['ALT'] > 40
    ggt_high = row['GGT'] > 40
    
    if (trig_high and (alt_high or ggt_high)) or (alt_high and ggt_high):
        return 1  # NAFLD Detected
    return 0      # Healthy

df['Diagnosis'] = df.apply(clinical_diagnosis_logic, axis=1)

# ---------------------------------------------------------
# 4. Feature Selection & Data Splitting
# ---------------------------------------------------------
# Removing target and identification columns (SEQN is a sequence ID)
X = df.drop(['Diagnosis'], axis=1)
if 'SEQN' in X.columns:
    X = X.drop(['SEQN'], axis=1)
y = df['Diagnosis']

# 80/20 Train-Test split for robust validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 5. Model Training (Optimized XGBoost)
# ---------------------------------------------------------
print("\nTraining diagnostic model on clinical biomarkers...")
model = xgb.XGBClassifier(
    n_estimators=100, 
    learning_rate=0.1, 
    max_depth=4, 
    subsample=0.8, 
    eval_metric='logloss',
    use_label_encoder=False
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 6. Evaluation & Statistical Metrics
# ---------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nModel Validation Accuracy: {accuracy * 100:.2f}%")
print("-" * 45)
print("Detailed Classification Performance:")
print(classification_report(y_test, y_pred))

# Visualizing Confusion Matrix

plt.figure(figsize=(7, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Fatty Liver Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.show()

# ---------------------------------------------------------
# 7. Model Serialization (Pickle Export)
# ---------------------------------------------------------
MODEL_EXPORT_NAME = "fatty_liver_model.pkl"
with open(MODEL_EXPORT_NAME, "wb") as f:
    pickle.dump(model, f)

print(f"\n✔ Module finalized and saved as: {MODEL_EXPORT_NAME}")
