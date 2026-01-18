# AI-Based Multi-Model System for Liver Disease Risk Assessment

An integrated AI ecosystem designed to assess liver health through a pipeline of machine learning models and clinical rule-based logic. The system diagnoses **Hepatitis C**, screens for **Blood Donor Eligibility**, detects **Fatty Liver Disease (NAFLD)**, and predicts **Liver Cancer Risk**.

## 📂 Repository Structure

| Directory | Description |
|-----------|-------------|
| `data/` | Contains dataset placeholders. **Note:** Raw medical data is not included for privacy/ethical reasons. |
| `models/` | Serialized models organized by disease type (Fatty Liver, Fibrosis, Donor, Cancer). |
| `training/` | Scripts used to train and validate the models (`.py` files). |
| `inference/` | The production logic, including the **Veto System** and prediction pipelines. |
| `notebooks/` | Jupyter notebooks for initial data exploration (EDA) and prototyping. |
| `docs/` | Detailed documentation on methodology, medical logic, and ethical standards. |

## 🧠 System Components

### 1. Fatty Liver Model (Rule-Based + ML)
Uses a hybrid approach where clinical rules (ALT/AST thresholds) generate ground-truth labels for training a robust classifier.

### 2. Blood Donor & Veto System
A safety-critical module. It ensures that a patient classified as "Eligible" by the Donor Model is **blocked** if the Fibrosis Model detects high-risk stages.

### 3. Liver Cancer Risk
Evaluates 7 pathological states to assess the risk of hepatocellular carcinoma using feature importance analysis.

## 🛠️ Installation
```bash
pip install -r requirements.txt
