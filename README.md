# AI-Based Multi-Model System for Liver Disease Risk Assessment

An integrated AI ecosystem designed to assess liver health through a pipeline of machine learning models and clinical rule-based logic. The system leverages **Hepatitis C** clinical data to diagnose liver damage progression, specifically identifying **Fibrosis** and its critical end-stage, **Cirrhosis**, using a **Multi-Stage Classification System** (Stage 1 to 4). Additionally, the pipeline screens for **Blood Donor Eligibility**, detects **Fatty Liver Disease (NAFLD)**, and predicts **Liver Cancer Risk**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

>  **Disclaimer:** This system is for **research and educational purposes only**.  
> It does **not** replace professional medical diagnosis.

---
##  Repository Structure

| Directory | Description |
|-----------|-------------|
| `models/` | Serialized models organized by disease type (Fatty Liver, Fibrosis, Donor, Cancer). |
| `docs/` | Detailed documentation on methodology, medical logic, and ethical standards. |
| `data/` | Contains dataset placeholders. **Note:** Raw medical data is not included for privacy/ethical reasons. |
| `code/` | Training and testing scripts (`.py`). **Fully optimized and ready for immediate execution on Google Colab.** |

---
### Project Overview

This project presents a **multi-model AI system** designed to assess liver health. The system works as follows:

* It first analyzes lab results to determine whether the user needs further assessment by subsequent models. Healthy users are excluded to save resources and avoid unnecessary processing.
* Diagnoses **liver fibrosis (Fibrosis)** and **cirrhosis (Cirrhosis)** at different stages.
* Detects **Non-Alcoholic Fatty Liver Disease (NAFLD)**.
* Predicts the **risk of liver cancer (Hepatocellular Carcinoma)**.

**System Features:**

* Specialized models for each liver-related condition.
* Sequential logic between models with a safety protocol to prevent incorrect decisions.
* Relies on routine blood markers to minimize invasive testing.
* **Clinical Basis:** The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and standard medical guidelines.

---

## Implemented Models
| # | Model | Training Data | Original Training Data | Original Data Source |
|:-:|:---|:---|:---|:---:|
| 1 | Gate&nbsp;Model | `data/processed/Liver_Patient_Dataset_Cleaned_19k.csv` | `data/raw/Liver Patient Dataset (LPD)_train.csv` | [Dataset](https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset) |
| 1 | Fatty&nbsp;Liver&nbsp;Model | `data/processed/FattyLiver.csv` | `data/raw/BIOPRO_H.xpt` `CBC_H.xpt` `HDL_H.xpt`  | [Dataset](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2013) |
| 3 |     HepatitisC&nbsp;Models <br> 3 Specialized Models | `data/processed/HepatitisC_Stage_model.csv` `data/processed/HepatitisC_status_model.csv` `data/processed/HepatitisC_complications.csv` | `data/raw/cirrhosis.csv` `data/processed/master8323pationt.csv` | [Dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset) |
| 4 | Cancer&nbsp;Model | `data/processed/The_Cancer_data_1500.csv` | `data/raw/The_Cancer_data_1500_V2.csv` | [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) |

---
## 1. The Gate Model: First Line 

The Gate Model serves as the **first line of defense** in the system, performing binary classification to separate healthy users from potential liver patients. It relies on an **XGBoost-trained model** (`gate_model.pkl`) that interprets biochemical input values using learned weights from a rigorously cleaned dataset, ensuring resource-efficient pre-screening before activating complex sub-models.

**(Trained Model):** `gate_model.pkl`

**Core Logic:** Evaluates user biochemical profiles and identifies potential risk patterns to filter cases needing further analysis.

**Critical Requirement (Positional Logic):** Inputs must follow the exact order used in training:
`['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'ALP', 'ALT', 'AST', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']`

For more information on dataset preparation, model training, and testing methodology, please visit: ➔ `docs/Gate_Model.md`

---

## 2. Fatty Liver Diagnosis Model (NAFLD)
Non-Alcoholic Fatty Liver Disease Diagnosis Model

This model analyzes the interplay between triglyceride levels and liver enzymes to identify inflammatory lipid accumulation. The system features a safety Veto protocol based on "Platelet counts" for the early detection of liver fibrosis markers associated with fatty liver.

(Trained Model): fatty_liver_model.pkl

Core Logic: Connects the "Raw Material" (Triglycerides) with the "Alarm Signal" (ALT/GGT) to distinguish NAFLD from viral hepatitis.

Critical Requirement (Positional Logic): The model processes data as an ordered mathematical matrix; therefore, inputs must be entered in the exact following order: `['Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol', 'Creatinine', 'Glucose', 'GGT', 'Bilirubin', 'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL']`.

For detailed technical and medical information: regarding NHANES Data Integration, cleaning strategies, and clinical scenario analysis, please visit: ➔ `docs/FattyLiver_Model.md`

---
## 3. Hepatitis C (HCV) Diagnostic & Prognostic Framework

This ensemble framework assesses liver fibrosis progression and calculates survival risk for **Hepatitis C (HCV)** patients. It integrates structural liver damage with functional outcomes using weighted **XGBoost** architectures to provide a multi-dimensional health assessment.

**(Trained Models):** `hepatitisC_stage_model.pkl`, `hepatitisC_complications.pkl`, `hepatitisC_status_model.pkl` (all in `models/` directory).

The framework employs a **hierarchical decision-making process** where the predicted histological stage acts as a high-weight input for the final survival probability. This mirrors clinical reality: physical scarring (detected by the Stage model) is a primary driver of functional failure and mortality risk.

### Serialized Models & Input Matrix

The system relies on three specialized models, each requiring a strict mathematical input order to function correctly.

#### **A. Complications Prediction Model**

* **Target File:** `models/hepatitisC_complications.pkl`
* **Clinical Goal:** Predicts risk of Ascites (Fluid Retention).
* **Performance:** **95.24% Accuracy**.
* **Input Dimension:** 14 Features.
* **Required Feature Order:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex', 'Hepatomegaly', 'Spiders', 'Edema']`

#### **B. Stage Prediction Model (Structural)**

* **Target File:** `models/hepatitisC_stage_model.pkl`
* **Clinical Goal:** Classifies Histological Fibrosis Stage (1, 2, or 3).
* **Input Dimension:** 19 Features (Includes calculated indices).
* **Required Feature Order:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Status', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI', 'Bilirubin_Albumin', 'Copper_Platelets']`

#### **C. Status Prediction Model (Prognostic)**

* **Target File:** `models/hepatitisC_status_model.pkl`
* **Clinical Goal:** Calculates Mortality Risk Probability.
* **Input Dimension:** 18 Features (Includes ALBI Score).
* **Required Feature Order:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI', 'ALBI_Score', 'Bili_Alb_Ratio']`


> **Developer Note & Reproducibility:**
> For the exact mathematical implementations of derived features (such as APRI and ALBI scores) and the complete training pipeline, please refer to the source code:
> * **Training Logic:** `code/train_HC_models.py` (Contains feature engineering & model serialization).
> * **Testing & Validation:** `code/test_HC_models.py` (Contains the inference engine and the 7-case validation suite).

**For comprehensive technical details, performance metrics, and clinical validation analysis, please refer to:** ➔ `docs/HepatitisC_Models.md`

---

## 4. Liver Cancer Risk Assessment Model

This model evaluates the probability of developing Hepatocellular Carcinoma (HCC) by analyzing the complex interplay between genetic predisposition and environmental triggers. It utilizes XGBoost weights to determine the impact of each analytical factor.

(Trained Model): cancer_model.pkl

Core Logic: The model demonstrates that a healthy lifestyle can effectively "neutralize" genetic predisposition; hereditary risk remains a "potential" rather than an "inevitable fate" without environmental catalysts (e.g., smoking and alcohol).

Critical Requirement (Positional Logic): The model processes data as an ordered mathematical matrix; therefore, inputs must be entered in the exact following order: `['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']`.

For detailed technical and medical information: regarding feature importance analysis, virtual clinic scenarios, and preventive prediction logic, please visit: ➔ `docs/Cancer_Risk_Model.md`

---
*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

---
