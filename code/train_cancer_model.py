import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os

# ---  Liver Cancer Risk Assessment System ---

#  Instructions for Google Colab:
# 1. Upload the training file "The_Cancer_data_1500_V2.csv" to your Colab storage.
# 2. Ensure the filename matches exactly as written in the code below.
# 3. Once execution is complete, "Liver_Cancer_Model.pkl" will be ready for download.

FILE_NAME = "The_Cancer_data_1500_V2.csv"

# 1. Data Loading and Preparation
try:
    df = pd.read_csv(FILE_NAME)
    print(f"✅ Dataset loaded successfully: {FILE_NAME}")
except FileNotFoundError:
    print(f"❌ Error: File '{FILE_NAME}' not found. Please upload it to your environment.")
    raise

# Clean data by removing any null values to ensure model stability
df = df.dropna()
print(f"📊 Total records ready for analysis: {len(df)}")

# 2. Feature and Target Separation
# Remove the 'Diagnosis' column from inputs as it is the target variable
X = df.drop(['Diagnosis'], axis=1)
y = df['Diagnosis']

# Display column order to ensure compatibility with the inference system later
print("\n📋 Input Features Order:")
for i, col in enumerate(X.columns, 1):
    print(f"{i}. {col}")

# Split data: 80% for training and 20% for testing to evaluate accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Machine Learning Model Construction (XGBoost)
print("\n🧠 Training the model to detect cancer risk patterns...")
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

# 4. Model Performance Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n🎉 Final Model Accuracy: {accuracy * 100:.2f}%")
print("-" * 40)
print("📄 Detailed Classification Report:")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix to visualize prediction quality
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=False)
plt.title('Confusion Matrix: Liver Cancer Risk')
plt.xlabel('Predicted Diagnosis')
plt.ylabel('Actual Diagnosis')
plt.show()

# 5. Save the Final Model
MODEL_NAME = "Liver_Cancer_Model.pkl"
pickle.dump(model, open(MODEL_NAME, "wb"))
print(f"\n💾 Model successfully saved as: {MODEL_NAME}")
