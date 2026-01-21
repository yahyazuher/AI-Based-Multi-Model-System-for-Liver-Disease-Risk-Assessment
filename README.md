# AI-Based Multi-Model System for Liver Disease Risk Assessment

An integrated AI ecosystem designed to assess liver health through a pipeline of machine learning models and clinical rule-based logic. The system leverages **Hepatitis C** clinical data to diagnose liver damage progression, specifically identifying **Fibrosis** and its critical end-stage, **Cirrhosis**, using a **Multi-Stage Classification System** (Stage 1 to 4). Additionally, the pipeline screens for **Blood Donor Eligibility**, detects **Fatty Liver Disease (NAFLD)**, and predicts **Liver Cancer Risk**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

>  **Disclaimer:** This system is for **research and educational purposes only**.  
> It does **not** replace professional medical diagnosis.

---

## Project Overview

This repository implements a **multi-model architecture** powered by **XGBoost (Extreme Gradient Boosting)** algorithms to ensure high-performance classification and risk assessment.

**Key System Features:**
- **Specialized Modeling:** Each model focuses on a **specific liver-related condition** rather than a generic output.
- **Interconnected Logic:** Models **do not act independently**; outputs act as flags for subsequent logic layers.
- **Safety-First Veto System:** A built-in logic layer prevents unsafe decisions (e.g., flagging a high-risk patient as a donor).
- **Clinical Grounding:** All predictions are strictly grounded in **clinical guidelines**.
- **Non-Invasive Focus:** The core diagnostic models are designed to minimize reliance on physical measurements, focusing primarily on **routine blood analysis** biomarkers.
---

##  Repository Structure

| Directory | Description |
|-----------|-------------|
| `data/` | Contains dataset placeholders. **Note:** Raw medical data is not included for privacy/ethical reasons. |
| `models/` | Serialized models organized by disease type (Fatty Liver, Fibrosis, Donor, Cancer). |
| `training/` | Training scripts (`.ipynb`). **Fully optimized and ready for immediate execution on Google Colab.** |
| `docs/` | Detailed documentation on methodology, medical logic, and ethical standards. |



---

## Implemented Models
| # | Model | Training Data | Original Training Data | Original Data Source | 80/20 Accuracy | Scientific Reliability |
|:---:|:---|:---|:---|:---:|:---:|:---:|
| 1 | Fatty&nbsp;Liver&nbsp;Model | `data/processed/FattyLiver.csv` | `data/raw/BIOPRO_H.xpt` `CBC_H.xpt` `HDL_H.xpt`  | [Dataset](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2013) | 100% | 99% |
| 2 | Fibrosis&nbsp;Model | `data/processed/Hepatitis.csv` | `data/raw/Fibrosis.csv` | [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) | 100% | 99% |
| 3 | Donor&nbsp;Eligibility&nbsp;Model | `data/processed/Hepatitis_DonorTyper.csv` | `data/raw/Donor.csv` | [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) | 100% | 99% |
| 4 | Hepatitis&nbsp;Model&nbsp;(C&nbsp;only) | `data/processed/Hepatitis.csv` | `data/raw/HepatitisC.csv` | [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) | 100% | 99% |
| 5 | Cancer&nbsp;Model | `data/processed/The_Cancer_data_1500.csv` | `data/raw/The_Cancer_data_1500_V2.csv` | [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) | 100% | 99% |

 Detailed documentation for each model is available under `docs/`.
---

# Fatty Liver Diagnosis Model (NAFLD)
Non-Alcoholic Fatty Liver Disease Diagnosis Model

This model analyzes the interplay between triglyceride levels and liver enzymes to identify inflammatory lipid accumulation. The system features a safety Veto protocol based on "Platelet counts" for the early detection of liver fibrosis markers associated with fatty liver.

Core Logic: Connects the "Raw Material" (Triglycerides) with the "Alarm Signal" (ALT/GGT) to distinguish NAFLD from viral hepatitis.

Critical Requirement (Positional Logic): The model processes data as an ordered mathematical matrix; therefore, inputs must be entered in the exact following order: ['Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol', 'Creatinine', 'Glucose', 'GGT', 'Bilirubin', 'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL']

For detailed technical and medical information: regarding NHANES Data Integration, cleaning strategies, and clinical scenario analysis, please visit: ➔ `docs/FattyLiver_Model.md`

---

---

---

---

# 5. Liver Cancer Risk Assessment Model

This model evaluates the probability of developing Hepatocellular Carcinoma (HCC) by analyzing the complex interplay between genetic predisposition and environmental triggers. It utilizes XGBoost weights to determine the impact of each analytical factor.

Core Logic: The model demonstrates that a healthy lifestyle can effectively "neutralize" genetic predisposition; hereditary risk remains a "potential" rather than an "inevitable fate" without environmental catalysts (e.g., smoking and alcohol).

Critical Requirement (Positional Logic): The model processes data as an ordered mathematical matrix; therefore, inputs must be entered in the exact following order: ['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']

For detailed technical and medical information: regarding feature importance analysis, virtual clinic scenarios, and preventive prediction logic, please visit: ➔ `docs/Cancer_Risk_Model.md`

---

