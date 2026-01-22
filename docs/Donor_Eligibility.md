
# Blood Donor Eligibility Model (HCV-Screening)

This model is a specialized component of the **AiLDS** framework designed to determine whether a person is eligible for blood donation based on biochemical markers. It specifically screens for **Hepatitis C** indicators and "suspect" patterns to ensure the safety of the blood supply.

---

### 1. Dataset & Model Overview

The model logic is trained on a refined version of the **Hepatitis C** dataset, specifically focused on distinguishing between healthy donors and individuals with underlying conditions.

| File Type | Path / Name | Description |
| --- | --- | --- |
| **Trained Model** | `models/donor_elgibility.pkl` | The serialized XGBoost classifier for donor screening. |
| **Processed Data** | `data/processed/donor_elgibility.csv` | The final dataset used for training, derived from `Hepatitis_DonorTyper.csv`. |
| **Original Source** | `data/raw/cirrhosis.csv` | Original source refined into `Hepatitis_DonorTyper.csv`. |

---

### 2. Feature Engineering & Encoding

To ensure mathematical compatibility with the **XGBoost** algorithm, clinical features were transformed as follows:

#### **A. Target Column (Category)**

The original labels were re-mapped to prioritize blood donation safety:

* **`0` (Blood Donor):** Eligible for donation.
* **`6` (Suspect Blood Donor):** Flagged for further screening or rejection.
* *(Other categories like 1=Hepatitis, 2=Fibrosis, 3=Cirrhosis are treated as automatic rejections in the workflow).*

#### **B. Clinical Biomarkers (Input Matrix)**

The model analyzes 13 distinct variables:
`[Category, Age, Sex, ALB, ALP, ALT, AST, BIL, CHE, CHOL, CREA, GGT, PROT]`

* **Sex Encoding:** `Male = 1`, `Female = 0`.
* **Missing Data Strategy:** If critical columns like **ALT**, **GGT**, or **Creatinine** are missing, the system utilizes **Default Healthy Values** or prompts the user for a manual entry. If the user chooses to proceed without these, the "Veto System" becomes the primary safeguard.

---

### 3. The Veto System (The Safety Valve)

This is the core security logic of the **Donor Eligibility Framework**. It prevents "False Negatives" by ensuring that the Donor Model and the Fibrosis (Stage) Model work in tandem.

#### **The Problem:**

If a user lacks certain tests and the system uses "default normal" values, the **Donor Model** might incorrectly flag a patient as "Sane/Healthy".

#### **The Solution: Integrated Veto**

The system implements a mandatory cross-check with the **Fibrosis Model (`hepatitis_stage.pkl`)**:

1. **Stage Detection:** Regardless of the Donor Model's output, if the Fibrosis Model detects that the patient is in **Stage 2, 3, or 4**, a "Veto" is triggered.
2. **Automatic Rejection:** The donation result is immediately overwritten and changed to **"Rejected"**.

> **Logic:** "If a patient has advanced scarring (Stage 2-4), they are medically unfit for donation, even if their acute enzymes (ALT/GGT) appear artificially normal due to missing data."

---

### 4. Technical Workflow Scenario

1. **Input:** User provides available blood tests.
2. **Missing Data:** If **GGT** is missing, the system uses a default "safe" value.
3. **Donor Output:** `donor_elgibility.pkl` sees safe GGT and says: **"Eligible"**.
4. **Fibrosis Check:** `hepatitis_stage.pkl` sees low **Platelets** and high **Prothrombin** (which are available) and says: **"Stage 3"**.
5. **Veto Trigger:** The system detects **Stage 3**  **VETO ENABLED**  Final Result: **REJECTED**.

---

### 5. Mathematical Justification

The Veto system ensures that structural liver damage (Fibrosis) is prioritized over acute functional markers when they are incomplete. This ensures a **Zero-Tolerance** safety policy for blood recipients.

---

بهذا الشكل، الملف جاهز للرفع على **GitHub**. هل تود مني الآن صياغة ملف الـ **`requirements.txt`** النهائي الذي يضمن تنصيب المكتبات البرمجية لتشغيل كافة هذه النماذج ونظام الفيتو بنجاح؟
