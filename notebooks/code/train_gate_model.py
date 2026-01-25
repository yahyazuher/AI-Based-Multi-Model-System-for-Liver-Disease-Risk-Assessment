import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import joblib
import os
import sys

# Configuration
DATASET_FILENAME = 'Liver_Patient_Dataset_Cleaned_19k.csv'
MODEL_FILENAME = 'gate_model.pkl'

# Direct link to the raw CSV file on GitHub
GITHUB_RAW_URL = 'https://raw.githubusercontent.com/yahyazuher/AI-Based-Multi-Model-System-for-Liver-Disease-Risk-Assessment/main/data/processed/Liver_Patient_Dataset_Cleaned_19k.csv'

def download_dataset_if_missing():
    """
    Checks if the dataset exists locally. If not, downloads it directly from the repository.
    """
    if not os.path.exists(DATASET_FILENAME):
        try:
            print(f"Dataset not found locally. Downloading from {GITHUB_RAW_URL}...")
            # Read CSV directly from the raw GitHub URL
            df = pd.read_csv(GITHUB_RAW_URL)
            # Save to local disk for future runs
            df.to_csv(DATASET_FILENAME, index=False)
            print("Download successful. Dataset saved locally.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            sys.exit(1)

def train_liver_prediction_model():
    print("Starting Liver Disease Prediction Pipeline...")
    
    # 1. Data Acquisition
    download_dataset_if_missing()

    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILENAME)
    
    # Remove any inadvertent missing values
    df = df.dropna()

    # Separate features (X) and target variable (y)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # 2. Label Encoding
    # Transforms target labels: 1 (Patient) -> 0, 2 (Healthy) -> 1
    le = LabelEncoder()
    y = le.fit_transform(y)

    # 3. Train-Test Split
    # Stratify ensures the training and test sets have the same proportion of class labels
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Model Initialization
    print("Training XGBoost Classifier...")
    # Hyperparameters are set to prevent overfitting on the cleaned dataset
    model = xgb.XGBClassifier(
        n_estimators=200,        
        learning_rate=0.05,      
        max_depth=4,             
        subsample=0.8,           
        colsample_bytree=0.8,    
        eval_metric='logloss',   
        random_state=42
    )

    # 5. Model Training
    model.fit(X_train, y_train)

    # 6. Evaluation
    print("Evaluating Model Performance...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("-" * 40)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("-" * 40)
    print(classification_report(y_test, y_pred))

    # 7. Serialization
    # Save the trained model to a file
    joblib.dump(model, MODEL_FILENAME)
    print(f"Model saved successfully: {MODEL_FILENAME}")

if __name__ == "__main__":
    train_liver_prediction_model()
