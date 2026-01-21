import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from google.colab import files  # Library to handle file uploads in Colab

print("--- Liver Cancer Risk Assessment System (XGBoost) ---")

# =========================================================
#  NOTE: REQUIRED FILES
# Please ensure you have the file "The_Cancer_data_1500.csv"
# ready on your computer before running this step.
# =========================================================

print("\n Please upload the dataset file: 'The_Cancer_data_1500.csv'")
# This command opens the file upload widget in Colab
uploaded = files.upload()

# 1. Load Data
try:
    # Check if the file exists in the uploaded dictionary or current directory
    if "The_Cancer_data_1500.csv" in uploaded.keys() or "The_Cancer_data_1500.csv" in pd.io.common.os.listdir():
        df = pd.read_csv("The_Cancer_data_1500.csv")
        print(" Cancer dataset loaded successfully.")
    else:
        raise FileNotFoundError
except FileNotFoundError:
    print(" Error: File 'The_Cancer_data_1500.csv' not found.")
    print("Please re-run the cell and upload the correct file.")
    # Stop execution if file is missing
    raise SystemExit

# Precautionary cleaning (Drop any rows with missing values)
df = df.dropna()
print(f" Total records ready for training: {len(df)}")

# ---------------------------------------------------------
# 2. Data Preprocessing for AI
# ---------------------------------------------------------
# X = Features (Age, Smoking, Genetics, Alcohol, etc.)
# We drop the 'Diagnosis' column because that is the Target
X = df.drop(['Diagnosis'], axis=1)

# y = Target (Does the patient have cancer? 0 or 1)
y = df['Diagnosis']

# Print column names to verify features (Useful for web integration later)
print("\n Features used by this model:")
print(list(X.columns))

# Split data: 80% Training - 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# 3. Model Training
# ---------------------------------------------------------
print("\n Training XGBoost model on cancer patterns...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. Results & Evaluation
# ---------------------------------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Cancer Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 30)

# Detailed Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False) # Red colormap for Cancer context
plt.title('Confusion Matrix (Cancer Model)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ---------------------------------------------------------
# 5. Saving the Model
# ---------------------------------------------------------
# Saving with a unique name to distinguish it from the Fatty Liver model
pickle.dump(model, open("Liver_Cancer_Model.pkl", "wb"))
print("\n Cancer model saved as: Liver_Cancer_Model.pkl")
