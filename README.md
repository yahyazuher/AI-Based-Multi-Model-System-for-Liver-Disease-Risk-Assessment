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

##  Implemented Models

| Model | Purpose | Training Data |
|------|--------|---------------|
| Fatty Liver Model | Detects active NAFLD using biochemical markers | `data/processed/FattyLiver.csv` |
|  | Detects liver scarring stages (1–4) | `data/processed/` |
|  | Evaluates blood donation safety | `data/processed/` |
|  | Viral hepatitis risk analysis (only C type) | `data/processed/` |
| Cancer Model | Lifestyle + genetic risk assessment | `data/processed/The_Cancer_data_1500.csv` |
| Supervisory Logic | Cross-model safety enforcement | _Rule-based (no training data)_ |

 Detailed documentation for each model is available under `docs/`.

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

