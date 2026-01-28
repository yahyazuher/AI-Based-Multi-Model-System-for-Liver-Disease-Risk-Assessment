# AI Liver Disease Diagnosis System

An integrated AI ecosystem designed to assess liver health through a pipeline of machine learning models and clinical rule-based logic. The system leverages **Hepatitis C** clinical data to diagnose liver damage progression, specifically identifying **Fibrosis** and its critical end-stage, **Cirrhosis**, using a **Multi-Stage Classification System** (Stage 1 to 4). Additionally, the pipeline screens for **Blood Donor Eligibility**, detects **Fatty Liver Disease (NAFLD)**, and predicts **Liver Cancer Risk**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

>  **Disclaimer:** This system is for **research and educational purposes only**.  
> It does **not** replace professional medical diagnosis.




---

| Directory / File | Description |
| --- | --- |
| **[`notebooks/`](./notebooks/)** | The core functional hub. It contains the **Main [.ipynb](https://github.com/yahyazuher/AI-Liver-Disease-Diagnosis-System/blob/main/notebooks/AI_Liver_Diseases_Diagnosis_System.ipynb)** file**, which integrates the entire diagnostic pipeline—from data processing to final prediction. |
| **[`models/`](./models/)** | Contains the finalized **`.pkl`** serialized models (Fatty Liver, Fibrosis, Cancer, etc.), ready for instant deployment. |
| **[`docs/`](./docs/)** | **Detailed documentation** regarding medical logic, ethical guidelines, and the **`XGBoost.md`** file for technical deep-dives. |
| **[`data/`](./data/)** | The storage hub for the **`.csv `,`.pxt`** processed and raw datasets. |
| **[`notebooks/code/`](./notebooks/code/)** | A dedicated sub-directory for **modular `.py` scripts**. Source Code for the project's backend logic |

---

### Project Overview

This project presents a **multi-model AI system** designed to assess liver health. The system works as follows:

* It first analyzes lab results using Gate Model to determine whether the user needs further assessment by subsequent models. Healthy users are excluded to save resources and avoid unnecessary processing.
* Diagnoses Stage Model **liver fibrosis** and **cirrhosis** at different stages.
* Detects **Non-Alcoholic Fatty Liver Disease (NAFLD)**.
* Predicts the **risk of liver cancer (Hepatocellular Carcinoma)**.
* The Status Model estimates the probability of mortality risk.

**System Features:**

* Specialized models for each liver-related condition.
* Sequential logic between models with a safety protocol to prevent incorrect decisions.
* Relies on routine blood markers to minimize invasive testing.
* **Clinical Basis:** The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and standard medical guidelines.

**Performance & Analysis:**
To view the comprehensive performance visualizations, including **Confusion Matrices** and detailed evaluation metrics for all models, please visit the main analysis notebook:
**[notebooks/AI_Liver_Disease_Diagnosis_System.ipynb](https://github.com/yahyazuher/AI-Liver-Disease-Diagnosis-System/blob/main/notebooks/AI_Liver_Diseases_Diagnosis_System.ipynb)**

---

## Implemented Models
| # | Model | Training Data | Original Training Data | Original Data Source |
|:-:|:---|:---|:---|:---:|
| 1 | [Gate Model](./docs/Gate_Model.md) | `data/processed/Liver_Patient_Dataset_Cleaned_19k.csv` | `data/raw/Liver Patient Dataset (LPD)_train.csv` | [Dataset](https://www.kaggle.com/datasets/abhi8923shriv/liver-disease-patient-dataset) |
| 2 | [Fatty Liver Model](./docs/FattyLiver_Model.md)| `data/processed/FattyLiver.csv` | `data/raw/BIOPRO_H.xpt` `CBC_H.xpt` `HDL_H.xpt`  | [Dataset](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2013) |
| 3 |    [HepatitisC&nbsp;Models <br> 3 Specialized Models](./docs/HepatitisC_Models.md) | `data/processed/HepatitisC_Stage_model.csv` `data/processed/HepatitisC_status_model.csv` `data/processed/HepatitisC_complications.csv` | `data/raw/cirrhosis.csv` `data/processed/master8323pationt.csv` | [Dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset) |
| 4 | [Cancer Model](./docs/Cancer_Risk_Model.md) | `data/processed/The_Cancer_data_1500.csv` | `data/raw/The_Cancer_data_1500_V2.csv` | [Dataset](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset) |

<p align="center"> <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="1000"> </p>

## 1. The Gate Model: First Line 

The Gate Model serves as the **first line of defense** in the system, performing binary classification to separate healthy users from potential liver patients. It relies on an **XGBoost-trained model** that interprets biochemical input values using learned weights from a rigorously cleaned dataset, ensuring resource-efficient pre-screening before activating complex sub-models.

**Trained Model:** [gate_model.pkl](./models/gate_model.pkl)

**Core Logic:** Evaluates user biochemical profiles and identifies potential risk patterns to filter cases needing further analysis.

**Critical Requirement (Positional Logic):** Inputs must follow the exact order used in training:
`['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'ALP', 'ALT', 'AST', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']`

For more information on dataset preparation, model training, and testing methodology, please visit: ➔ [docs/Gate_Model.md](./docs/Gate_Model.md)

<p align="center"> <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="1000"> </p>

## 2. Fatty Liver Diagnosis Model (NAFLD)
Non-Alcoholic Fatty Liver Disease Diagnosis Model

This model analyzes the interplay between triglyceride levels and liver enzymes to identify inflammatory lipid accumulation. 

**Trained Model:** [fatty_liver_model.pkl](./models/fatty_liver_model.pkl)

Core Logic: Connects the "Raw Material" (Triglycerides) with the "Alarm Signal" (ALT/GGT) to distinguish NAFLD from viral hepatitis.
* **Performance:** **97.41% Accuracy**.
* * **Required Feature Order:**  `['Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol', 'Creatinine', 'Glucose', 'GGT', 'Bilirubin', 'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL']`.

For detailed technical and medical information: regarding NHANES Data Integration, cleaning strategies, and clinical scenario analysis, please visit: ➔  [docs/FattyLiver_Model.md](./docs/FattyLiver_Model.md) 

<p align="center"> <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="1000"> </p>

## 3. Hepatitis C (HCV) Diagnostic & Prognostic Framework

This ensemble framework assesses liver fibrosis progression and calculates survival risk for **Hepatitis C (HCV)** patients. It integrates structural liver damage with functional outcomes using weighted **XGBoost** architectures to provide a multi-dimensional health assessment.

**(Trained Models):** [hepatitisC_stage_model.pkl](./models/hepatitisC_stage_model.pkl) , [hepatitisC_complications.pkl](./models/hepatitisC_complications.pkl), [hepatitisC_status_model.pkl](./models/hepatitisC_status_model.pkl) (all in `models/` directory).

The framework employs a **hierarchical decision-making process** where the predicted histological stage acts as a high-weight input for the final survival probability. This mirrors clinical reality: physical scarring (detected by the Stage model) is a primary driver of functional failure and mortality risk.

### Serialized Models & Input Matrix

The system relies on three specialized models, each requiring a strict mathematical input order to function correctly.

#### **A. Complications Prediction Model**

* **Target File:** [hepatitisC_complications.pkl](./models/hepatitisC_complications.pkl)
* **Clinical Goal:** Predicts risk of Ascites (Fluid Retention).
* **Performance:** **95.24% Accuracy**.
* **Input Dimension:** 14 Features.
* **Required Feature Order:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex', 'Hepatomegaly', 'Spiders', 'Edema']`

#### **B. Stage Prediction Model (Structural)**

* **Target File:** [hepatitisC_stage_model.pkl](./models/hepatitisC_stage_model.pkl)
* **Clinical Goal:** Classifies Histological Fibrosis Stage (1, 2, or 3).
* * **Performance:** **62.50% Accuracy**.
* **Input Dimension:** 19 Features (Includes calculated indices).
* **Required Feature Order:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Status', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI', 'Bilirubin_Albumin', 'Copper_Platelets']`

#### **C. Status Prediction Model (Prognostic)**

* **Target File:** [hepatitisC_status_model.pkl](./models/hepatitisC_status_model.pkl)
* **Clinical Goal:** Calculates Mortality Risk Probability.
* * **Performance:** **71.43% Accuracy**.
* **Input Dimension:** 18 Features (Includes ALBI Score).
* **Required Feature Order:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI', 'ALBI_Score', 'Bili_Alb_Ratio']`


> **Note:**
> For the exact mathematical implementations of derived features (such as APRI and ALBI scores) and the complete training pipeline, please refer to the source code:
> * **Training Logic:** `notebooks/code/train_HC_models.py` (Contains feature engineering & model serialization).
> * **Testing & Validation:** `notebooks/code/test_HC_models.py` (Contains the inference engine and the 7-case validation suite).

**For comprehensive technical details, performance metrics, and clinical validation analysis, please refer to:** ➔ [docs/HepatitisC_Models.md](./docs/HepatitisC_Models.md)

<p align="center"> <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="1000"> </p>

## 4. Liver Cancer Risk Assessment Model

This model evaluates the probability of developing Hepatocellular Carcinoma (HCC) by analyzing the complex interplay between genetic predisposition and environmental triggers. It utilizes XGBoost weights to determine the impact of each analytical factor.

**Trained Model:** [cancer_model.pkl](./models/cancer_model.pkl)

Core Logic: The model demonstrates that a healthy lifestyle can effectively "neutralize" genetic predisposition; hereditary risk remains a "potential" rather than an "inevitable fate" without environmental catalysts (e.g., smoking and alcohol).

* **Performance:** **94.00% Accuracy**.

Critical Requirement (Positional Logic): The model processes data as an ordered mathematical matrix; therefore, inputs must be entered in the exact following order: `['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']`.

For detailed technical and medical information: regarding feature importance analysis, virtual clinic scenarios, and preventive prediction logic, please visit: ➔ [docs/Cancer_Risk_Model.md](./docs/Cancer_Risk_Model.md)

<p align="center"> <img src="https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/aqua.png" width="1000"> </p>

### **Project Comprehensive Feature Dictionary**

The following 36 clinical markers form the foundation for all feature engineering and model inputs.


| #  | Feature Code Name | Scientific Name             | Unit / Value       | Input Type   | Formula                     | Clinical Importance & Diagnostic Role                     | Model Scope            |
| -- | ----------------- | --------------------------- | ------------------ | ------------ | --------------------------- | --------------------------------------------------------- | ---------------------- |
| 01 | Age               | Patient Age                 | Years              | Numerical    | N/A                         | Core demographic factor influencing fibrosis progression. | All Models             |
| 02 | Gender            | Biological Sex              | 0: Female, 1: Male | Binary       | N/A                         | Adjusts for biological enzyme variability.                | All Models             |
| 03 | BMI               | Body Mass Index             | kg/m²              | Numerical    | Weight / Height²            | Primary indicator of fatty liver risk.                    | Fatty Liver & Cancer   |
| 04 | Smoking           | Smoking Status              | 0: No, 1: Yes      | Binary       | N/A                         | Increases oxidative stress and accelerates fibrosis.      | Cancer Risk Model      |
| 05 | GeneticRisk       | Genetic Predisposition      | 0–2 (Categories)   | Categorical  | N/A                         | Classifies hereditary and familial cancer risk.           | Cancer Risk Model      |
| 06 | PhysicalActivity  | Physical Activity           | Hours/week         | Numerical    | N/A                         | Improves insulin sensitivity and reduces hepatic fat.     | Fatty Liver & Cancer   |
| 07 | AlcoholIntake     | Alcohol Consumption         | Units/week         | Numerical    | N/A                         | Differentiates alcoholic vs non-alcoholic liver injury.   | Fatty Liver & Cancer   |
| 08 | CancerHistory     | Cancer History              | 0: No, 1: Yes      | Binary       | N/A                         | Raises probabilistic weight for HCC development.          | Cancer Risk Model      |
| 09 | Ascites           | Ascites                     | 0: No, 1: Yes      | Binary       | N/A                         | Marker of advanced hepatic decompensation.                | Hep C (Stage & Status) |
| 10 | Hepatomegaly      | Hepatomegaly                | 0: No, 1: Yes      | Binary       | N/A                         | Clinical enlargement of the liver.                        | Hep C (All Models)     |
| 11 | Spiders           | Spider Angiomas             | 0: No, 1: Yes      | Binary       | N/A                         | Cutaneous vascular lesions from hormonal imbalance.       | Hep C (All Models)     |
| 12 | Edema             | Peripheral Edema            | 0: No, 1: Yes      | Binary       | N/A                         | Fluid retention due to hypoalbuminemia.                   | Hep C (All Models)     |
| 13 | Total_Bilirubin   | Total Bilirubin             | mg/dL              | Numerical    | Direct + Indirect           | Reflects hepatic excretory function.                      | All Models             |
| 14 | Direct_Bilirubin  | Direct Bilirubin            | mg/dL              | Numerical    | N/A                         | Indicates mechanical or cholestatic obstruction.          | Gate Model         |
| 15 | Albumin           | Serum Albumin               | g/dL               | Numerical    | N/A                         | Indicator of liver synthetic capacity.                    | All Models             |
| 16 | Total_Proteins    | Total Proteins              | g/dL               | Numerical    | Albumin + Globulin          | Assesses nutritional and immune status.                   | Gate Model         |
| 17 | A/G_Ratio         | Albumin/Globulin Ratio      | Ratio              | Calculated   | Albumin / (Total − Albumin) | Marker of chronic inflammation and tissue damage.         | Gate Model         |
| 18 | ALT               | Alanine Aminotransferase    | U/L                | Numerical    | N/A                         | Most specific enzyme for hepatocellular injury.           | All Models             |
| 19 | AST               | Aspartate Aminotransferase  | U/L                | Numerical    | N/A                         | Enzyme reflecting multi-tissue injury.                    | All Models             |
| 20 | ALP               | Alkaline Phosphatase        | U/L                | Numerical    | N/A                         | Elevated in biliary obstruction.                          | All Models             |
| 21 | GGT               | Gamma-Glutamyl Transferase  | U/L                | Numerical    | N/A                         | Highly sensitive to alcohol toxicity and steatosis.       | Fatty Liver Model      |
| 22 | Triglycerides     | Triglycerides               | mg/dL              | Numerical    | N/A                         | Core biomarker of fatty liver disease.                    | Fatty Liver & Hep C    |
| 23 | Cholesterol       | Total Cholesterol           | mg/dL              | Numerical    | N/A                         | Reflects metabolic and hepatic synthetic status.          | Fatty Liver & Hep C    |
| 24 | HDL               | High-Density Lipoprotein    | mg/dL              | Numerical    | N/A                         | Inversely associated with metabolic syndrome.             | Fatty Liver Model      |
| 25 | Glucose           | Fasting Blood Glucose       | mg/dL              | Numerical    | N/A                         | Diabetes is a major driver of fibrosis.                   | Fatty Liver Model      |
| 26 | Creatinine        | Serum Creatinine            | mg/dL              | Numerical    | N/A                         | Assesses renal function (hepatorenal risk).               | Fatty Liver Model      |
| 27 | Uric_Acid         | Uric Acid                   | mg/dL              | Numerical    | N/A                         | Oxidative stress marker linked to steatosis.              | Fatty Liver Model      |
| 28 | Copper            | Urinary Copper              | µg/day             | Numerical    | N/A                         | Detects copper accumulation disorders.                    | Hep C (All Models)     |
| 29 | Platelets         | Platelet Count              | 10⁹/L              | Numerical    | N/A                         | Thrombocytopenia indicates portal hypertension.           | All Models             |
| 30 | Prothrombin       | Prothrombin Time            | Seconds            | Numerical    | N/A                         | Reflects clotting factor synthesis efficiency.            | Hep C (All Models)     |
| 31 | Status            | Clinical Status             | Categories         | Target/Input | N/A                         | Clinical outcome (Alive / Death ).            | Input for Stage Model  |
| 32 | Stage             | Fibrosis Stage              | Categories         | Target       | N/A                         | fibrosis severity (F1–F3).                   | Target (Output)        |
| 33 | APRI              | AST to Platelet Ratio Index | Score              | Calculated   | ((AST/40)/Platelets)*100    | fibrosis predictor.                          | Hep C (Stage & Status) |
| 34 | ALBI_Score        | Albumin-Bilirubin Score     | Score              | Calculated   | Formula-based               | Assesses liver function and mortality risk.               | Hep C (Status Model)   |
| 35 | Bili_Alb_Ratio    | Bilirubin/Albumin Ratio     | Ratio              | Calculated   | Total_Bilirubin / Albumin   | Reflects excretory–synthetic imbalance.                   | Hep C (Stage & Status) |
| 36 | Copper_Platelets  | Copper–Platelet Interaction | Ratio              | Calculated   | Copper / Platelets          | Marker of toxicity and portal hypertension.               | Hep C (Stage Model)    |

---


*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

