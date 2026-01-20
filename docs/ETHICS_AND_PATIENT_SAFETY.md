# Ethics & Patient Safety

This document explains the **ethical principles and safety mechanisms** used in the  
**Multi-Model AI Liver Disease System**.

The goal is to clarify **why** specific design decisions were made to protect patients,
reduce medical risk, and prevent unsafe AI behavior.

>  Technical details, training logic, and datasets are documented separately under `docs/`
> and `data/processed/` to keep this file focused on ethics and safety only.

---

## Core Ethical Principles

- **Patient safety comes first**
- **False Negatives are treated as critical failures**
- **No model is allowed to act alone**
- **Uncertainty leads to conservative decisions**
- **AI supports clinicians — it does not replace them**

---

## 1. False Negatives as a High-Risk Ethical Failure

In medical AI, not all errors are equal.

A **False Negative** (classifying a sick or high-risk patient as healthy) can lead to:
- Delayed diagnosis
- Unsafe blood donation
- Direct harm to other patients

For this reason, the system is intentionally designed to be **risk-averse**, prioritizing
safety over optimistic predictions.

---

## 2. The Veto System (Fail-Safe Architecture)

### Why a Veto System?
Some models rely on biochemical markers that may be:
- Missing
- Incomplete
- Replaced with default values

Allowing such models to make final medical decisions **without supervision** was considered
ethically unsafe.

### How It Works
The system applies **cross-model oversight**:

- A *primary model* may issue a favorable decision.
- A *supervisory model* independently evaluates structural or high-risk indicators.
- If a conflict is detected, the permissive decision is **automatically overridden**.

This guarantees that **safety always dominates model autonomy**.

Related documentation:
- `Hepatitis_Model.md`

---

## 3. Example: Fatty Liver & Blood Donation Safety

This project contains multiple models.  
The following example illustrates the ethical logic using **one model only**.

### Scenario
- The donor model predicts **Eligible**
- Some biochemical inputs were missing or defaulted
- The fibrosis model detects **Stage ≥ 2** using Platelets / Prothrombin

### Ethical Outcome
- The system triggers the **Veto**
- The donor status is changed to **Rejected**
- Potential harm is prevented

Model details:
- `docs/FattyLiver_Model.md`  
- Training data: `data/processed/FattyLiver_Learning_db.csv`

---

## 4. Unified Input & Accountability

### Design Choice
The frontend sends **one unified JSON payload** containing all user data.

### Ethical Benefit
- Prevents selective input manipulation
- Ensures all models analyze the same clinical snapshot
- Improves traceability and accountability

---

## 5. Data Privacy & De-Identification

Patient privacy is treated as a **hard constraint**, not an optional feature.

### Applied Measures
- Engineering identifiers (e.g. `SEQN`) are used **only for data merging**
- All identifiers are dropped **before training**
- Models operate on numerical arrays only
- No personal identity can be reconstructed

Model details:
- `docs/FattyLiver_Model.md`  
- Training data: `data/processed/FattyLiver_Learning_db.csv`
  
---

## 6. Guideline-Based Labeling (No Heuristic Diagnosis)

All target labels are created using **rule-based labeling** grounded in
established clinical guidelines.

### Why This Matters
- Prevents speculative AI decisions
- Makes predictions explainable
- Ensures clinical defensibility

Labeling logic:
- `docs/FattyLiver_Model.md`

---

## 7. Avoiding Genetic Determinism

The system is explicitly designed to avoid **fatalistic predictions**.

- Genetic risk increases probability — not certainty
- Lifestyle factors can amplify or reduce risk
- Protective behavior is treated as an ethical modifier

This ensures **non-discrimination** and preserves patient agency.

Related model:
- `docs/Cancer_Risk_Model.md`

---

## 8. Scope & Responsibility

This system is a **clinical decision-support tool** only.

- It does **not** replace medical professionals
- The Veto System is a computational safeguard, not a guarantee
- Final responsibility always lies with qualified clinicians

---

## Final Statement

> When uncertainty exists, the system chooses **refusal over reassurance**.
>  
> Safety is enforced by design — not left to model confidence.

This document outlines the ethical framework governing the **Multi-Model AI Liver Disease Diagnostic System**, emphasizing patient safety, data privacy, and the fail-safe mechanisms implemented to prevent medical errors.

---

## 1. The "Veto System": An Ethical Necessity for Blood Safety

In medical AI, a "False Negative" (classifying a sick patient as healthy) is not just a statistical error; it is a life-threatening risk. This project implements a **Veto System** specifically for the Blood Donor Eligibility module to mitigate this risk.

### The Ethical Dilemma
The **Donor Model** relies on biochemical markers. In scenarios where a user lacks specific input values (e.g., ALT, GGT), the system might use default "normal" values to process the request. This creates a risk where the Donor Model incorrectly classifies a candidate as "Eligible".

### The Solution: Cross-Model Validation
We enforce a strict rule: **"The models must not act in isolation."**.
The system integrates the **Fibrosis Model** as a supervisor (Veto Authority) over the Donor Model based on the following logic:

1.  **Independent Analysis:** The Fibrosis Model analyzes different features, specifically **Platelets** and **Prothrombin**, which are critical indicators of liver scarring.
2.  **Conflict Resolution:** If the Donor Model predicts "Eligible" (0), but the Fibrosis Model detects advanced scarring (Stage 2, 3, or 4), the system detects a conflict.
3.  **The Veto Action:** The system automatically overrides the Donor Model's decision, changing the status to **"Rejected"**. This ensures that blood from a patient with hidden liver fibrosis never reaches a recipient.


---

## 2. Data Privacy & Anonymization Strategy

Respecting patient confidentiality is paramount. This project adheres to strict de-identification protocols to ensure the AI learns biological patterns, not personal identities.

### Handling of Personally Identifiable Information (PII)
* **Removal of Explicit Identifiers:** The raw dataset contained a sequence number (`SEQN`) for each patient (docs/FattyLiver_DataEngineering.md and data/processed/FattyLiver_Learning_db.csv). While this was used initially as a **Primary Key** to merge disparate datasets (Biochemistry, CBC, Cholesterol), it was strictly treated as an engineering utility.
* **The "Drop" Protocol:** Before the data enters the training phase (`.fit`), the `SEQN` column is programmatically dropped. The model acts purely on mathematical arrays of biological data, ensuring it is impossible to reverse-engineer the data to identify specific individuals.
* **Attribute Encoding:** Demographic attributes like Gender (Sex) were converted to numerical binary formats (Male=1, Female=0) purely for statistical correlation, stripping away text-based identifiers .

---

## 3. Responsible Labeling: Guideline-Based Ground Truth

To avoid "AI Hallucinations," the target variables for the **Fatty Liver Model** were not inferred loosely. Instead, we applied a **Rule-Based Labeling** approach grounded in clinical guidelines.

* **Clinical Thresholds:** Diagnoses were established using globally recognized thresholds derived from medical literature:
    * **ALT > 40 IU/L:** Indicator of hepatocellular injury (ACG Guidelines).
    * **Triglycerides > 150 mg/dL:** Indicator of metabolic syndrome (NCEP ATP III).
    * **GGT > 40 IU/L:** A sensitive marker for oxidative stress and fatty liver detection.
* **Justification:** The diagnosis is assigned only when biochemical markers confirm both lipid elevation and liver stress, creating a model that is clinically defensible.
* 
---
 
**This system is designed as a decision-support tool and does not replace professional medical diagnosis. The "Veto System" acts as a computational safeguard, not a guarantee of clinical safety.** 
