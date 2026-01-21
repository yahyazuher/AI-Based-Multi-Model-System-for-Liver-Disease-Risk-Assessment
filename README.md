# AI-Based Multi-Model System for Liver Disease Risk Assessment

An integrated AI ecosystem designed to assess liver health through a pipeline of machine learning models and clinical rule-based logic. 

The system leverages **Hepatitis C** clinical data to diagnose liver damage progression, specifically identifying **Fibrosis** and its critical end-stage, **Cirrhosis**, using a **Multi-Stage Classification System** (Stage 1 to 4). Additionally, the pipeline screens for **Blood Donor Eligibility**, detects **Fatty Liver Disease (NAFLD)**, and predicts **Liver Cancer Risk**.


>  This system is for **research and educational purposes only**.  
> It does **not** replace professional medical diagnosis.

##  Repository Structure

| Directory | Description |
|-----------|-------------|
| `data/` | Contains dataset placeholders. **Note:** Raw medical data is not included for privacy/ethical reasons. |
| `models/` | Serialized models organized by disease type (Fatty Liver, Fibrosis, Donor, Cancer). |
| `training/` | Scripts used to train and validate the models (`.py` files). |
| `docs/` | Detailed documentation on methodology, medical logic, and ethical standards. |



---

##  Project Overview

This repository implements a **multi-model architecture** for liver disease analysis, where:

- Each model focuses on a **specific liver-related condition**
- Models **do not act independently**
- A safety-first **Veto System** prevents unsafe decisions
- All predictions are grounded in **clinical guidelines**

The system is intentionally designed to work **without physical measurements**
(e.g., weight, BMI), relying instead on **routine blood analysis**.

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

Shutterstock
استكشاف
Donor Eligibility Model: Acts as a biosafety gatekeeper for blood donation screening. The model evaluates a donor's general health parameters to ensure that donation is safe for the donor and that the blood is free from indicators that might harm the recipient.

Hepatitis Model (Category C): Focuses on assessing the risk of Hepatitis C Virus (HCV) infection. By studying viral patterns and immune responses within the data, the model provides a rapid and highly accurate preliminary diagnosis.

Cancer Model (Risk Assessment): A comprehensive assessment tool that bridges the gap between Lifestyle factors and Genetic predisposition. It predicts the probability of Hepatocellular Carcinoma (HCC), demonstrating how behavioral changes can mitigate hereditary risks.

Supervisory Logic: The system's "central intelligence" unit. It reconciles outputs from the five previous models to ensure there are no conflicting results. Its core function is to enforce final safety protocols before issuing a clinical report to the user.
---

##  Ethics & Patient Safety (Core Design)

This project follows a **safety-first AI philosophy**:

-  False Negatives are treated as **critical failures**
-  Conservative decisions are preferred over optimistic ones
-  Patient privacy is enforced by design
-  No single model is allowed to make high-impact decisions alone

 Full ethical framework:
- `docs/ETHICS_AND_PATIENT_SAFETY.md`

---

##  The Veto System (Fail-Safe Mechanism)

Some models may operate with:
- Missing inputs
- Default or imputed values

To prevent unsafe outcomes, the system applies a **Veto System**:

- A permissive decision from one model can be overridden
- A supervisory model detects high-risk indicators
- Final decisions always prioritize **patient and recipient safety**

 Related documentation:
- `docs/Fibrosis_Model.md`
- `docs/Donor_Eligibility_Model.md`

---

##  Clinical Ground Truth (Rule-Based Labeling)

Instead of heuristic or inferred labels, all models use
**guideline-based rule labeling** derived from medical literature.

Examples:
- ALT > 40 IU/L
- Triglycerides > 150 mg/dL
- GGT > 40 IU/L

This ensures:
- Transparency
- Explainability
- Clinical defensibility

 Labeling methodology:
- `docs/FattyLiver_Model.md`

---

##  Data Privacy & De-Identification

Patient confidentiality is treated as a **hard constraint**:

- Engineering identifiers (e.g., `SEQN`) are used only for dataset merging
- All identifiers are dropped **before training**
- Models operate purely on numerical data
- Re-identification is not possible

 Data engineering details:
- `docs/FattyLiver_DataEngineering.md`

---

##  Repository Structure

