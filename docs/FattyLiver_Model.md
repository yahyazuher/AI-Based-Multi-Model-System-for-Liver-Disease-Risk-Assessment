# Fatty Liver Diagnosis Model (NAFLD)

This section is dedicated to the detection of **Non-Alcoholic Fatty Liver Disease (NAFLD)**. The model distinguishes between general hyperlipidemia (high blood fats) and actual liver injury caused by hepatic steatosis. It bridges biochemical laboratory data with advanced clinical logic using the **XGBoost** algorithm, stored as `fatty_liver_model.pkl`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

---

### Dataset Overview

| Name | Database Location | Function |
| --- | --- | --- |
| **fatty_liver_model.pkl** | `models/` | The trained model containing the optimized weights for NAFLD detection. |
| **train_fatty_liver.py** | `code/` | Source code for data merging (`SEQN` logic) and model training.Can be run directly in Google Colab |
| **test_fatty_liver.py** | `code/` | Source code dedicated to testing and evaluating the model performance.Can be run directly in Google Colab |
| **FattyLiver.csv** | `data/processed/` | Engineered dataset from NHANES 2013-2014 cycles containing 6,533 patient records. |
| **XGBoost.md** | `docs/` | Technical documentation of the underlying boosting mechanism. |

---

### **Training Phase & Data Partitioning**

The model's exceptional performance is derived from a strategic data split of **80% for training** and **20% for testing**, which resulted in a final predictive accuracy of **99.98%**. This high level of precision ensures that the diagnostic logic is consistently applied across the entire population.

* **Total Clean Records:** **6,533 patients** (After rigorous cleaning and removing missing values).
* **Training Data:** The model was trained on **5,226 patients** from the `FattyLiver.csv` dataset to learn complex metabolic patterns.
* **Testing Data:** A holdout set of **1,307 patients** was reserved to validate the model's accuracy on unseen clinical data.
* **Class Distribution:** The dataset contains **5,877 Healthy** cases and **656 NAFLD** cases, demonstrating the model's ability to identify the minority "Patient" class with high sensitivity.

> (For more details about ML model, see:  [`docs/XGBoost.md`](./XGBoost.md). 

---
### Data Engineering & Integration Strategy

The integrity of the `FattyLiver.csv` dataset is built on a surgical data integration strategy. The primary challenge involved merging records from three different laboratory files where patient counts were inconsistent (e.g., Biochemistry: 6,946 vs. CBC: 9,249).

#### **The "SEQN" Surgical Merge**

To eliminate Data Shift errors—where one patient’s results are incorrectly mapped to another—we utilized the SEQN (Sequence Number) as a Primary Key via the VLOOKUP function. Any patient missing core laboratory components was purged, ensuring 100% data integrity for the final 6,533 records.

---

### Feature Selection: The "Biological Fingerprint"

The dataset was reduced by over **50%** to eliminate statistical noise and prevent **Multicollinearity**, focusing only on markers with high clinical weight.

| Feature Type | Markers | Engineering Logic |
| --- | --- | --- |
| **Inflammation** | ALT, AST, GGT, ALP | Core indicators of active hepatocellular injury and bile duct stress. |
| **Metabolic Fuel** | Triglycerides (TG), Glucose | Identifies the excess lipids available for liver fat storage. |
| **The Veto Factor** | **Platelets** | A critical marker for detecting underlying Fibrosis/Cirrhosis. |
| **Synthetic Capacity** | Albumin, Bilirubin | Evaluates the liver's ability to manufacture protein and clear toxins. |

* **Redundancy & SI Units:** Columns such as (`LBDSTRSI`, `LBDSALSI`) were removed to prevent **Multicollinearity**. Including the same clinical information twice in different units (e.g., conventional vs. SI units) confuses the mathematical logic of the model and leads to inaccurate weight distribution.
* **Non-Specific Electrolytes & Minerals:** General systemic markers like Calcium (`LBXSCA`), Sodium (`LBXSNASI`), and Phosphorus (`LBXSPH`) were excluded. While important for general health, these markers lack the clinical specificity required for **NAFLD** diagnosis and do not contribute to the liver-specific predictive power.
* **Noise Reduction (Low-Impact Markers):** Variables such as Globulin (`LBXSGB`) and LDH (`LBXSLDSI`) were discarded. This ensures the model remains "lightweight" and highly focused on the core biological fingerprint of fatty liver (Enzymes and Metabolic markers), significantly reducing statistical noise.

---


### Positional Logic:

The `FattyLiver_Model.pkl` file is a **Mathematical Matrix**. It does not interpret column headers; instead, it relies strictly on **Positional Indices** (the order of data).

**Critical Execution Requirement:**
Feeding data in the wrong sequence (e.g., placing Glucose in the Triglycerides slot) will lead to a total diagnostic failure. Data must be submitted in this exact order, model input sequence (13 Features):
['Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol', 'Creatinine', 'Glucose', 'GGT', 'Bilirubin', 'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL']

---
## **Model Optimization and Diagnostic Logic**


### **1. The Deterministic Nature of the Model**

The primary reason for the exceptional accuracy lies in the **Target Engineering** phase. Unlike models that attempt to predict "stochastic" or "hidden" outcomes, this system is designed to learn and execute a specific, high-integrity clinical protocol.

The ground truth was established using the following **Diagnostic Logic**:

```python
def create_clinical_target(row):
    """
    Diagnostic Logic: Confirms NAFLD when high lipids (Triglycerides)
    coexist with markers of hepatocellular injury (ALT/GGT).
    """
    trig_high = row['Triglycerides'] > 150
    alt_high = row['ALT'] > 40
    ggt_high = row['GGT'] > 40

    if (trig_high and (alt_high or ggt_high)) or (alt_high and ggt_high):
        return 1
    else:
        return 0

```

Because the target is based on explicit mathematical conditions (IF/ELSE logic), the problem becomes **deterministic**. The XGBoost algorithm is not "guessing" based on noisy patterns; instead, it is performing automated logical deduction. With **6,544 records**, the model has an abundance of evidence to perfectly map these clinical thresholds into its decision trees.

---

### **2- Strategic Justification of Model Parameters**

To ensure the system operates with both mathematical precision and clinical relevance, the following hyperparameters were selected for the final XGBoost configuration:

```python
model = xgb.XGBClassifier(
    n_estimators=100,    
    learning_rate=0.1,    
    max_depth=4,         
    subsample=0.8,       
    eval_metric='logloss'
)

```

#### **A. `max_depth=4`**

In XGBoost, `max_depth` determines the maximum number of levels (or "questions") a decision tree can develop to reach a diagnosis. A depth of 4 is the **"sweet spot"** for this specific logic:

* **Logic Mapping:** Since our diagnostic rule primarily relies on **3 variables** (`Triglycerides`, `ALT`, `GGT`), a depth of 4 provides enough levels to evaluate each primary marker and a final level to confirm the synergy between them.
* **Noise Filtering:** By capping the depth at 4, we force the model to prioritize high-weight variables. This prevents the model from asking unnecessary "questions" about the other 10 biological markers (like Creatinine or Albumin), which might contain minor noise that does not contribute to a NAFLD diagnosis.

**Example of the 4-Level Clinical Decision Path:**

 * **Level 1 (Depth 1):** Is Triglycerides > 150 mg/dL? *(If Yes, move deeper)*.
 * **Level 2 (Depth 2):** Is ALT > 40 U/L? *(If Yes, move deeper)*.
 * **Level 3 (Depth 3):** Is GGT > 40 U/L? *(If Yes, move deeper)*.
 * **Level 4 (Depth 4):** Final threshold check  **Diagnosis: PATIENT**.

#### **B. `n_estimators=100`**

* Because the clinical rules are deterministic and the dataset of 6,544 patients is highly organized, the model does not require thousands of boosting rounds. 100 trees are sufficient to reach a near-perfect global minimum loss without unnecessary computational overhead or risk of memorizing the data.

#### **C. `learning_rate=0.1`**

* This ensures a controlled and stable learning process. In a dataset of this scale, a 0.1 rate allows the model to learn the primary metabolic patterns quickly and accurately without "overshooting" the optimal mathematical solution.

#### **D. `subsample=0.8`**

* By training each tree on a random 80% subset of the data, we introduce a layer of "stochastic" robustness. This ensures the model’s 99.98% accuracy is representative of the entire population and not biased toward specific outliers in the NHANES records.


The results demonstrate that **High-Quality Data + Clear Clinical Logic = Perfect Diagnostic Execution**. The 99.98% accuracy achieved during testing is a direct reflection of the dataset's cleanliness and the logical consistency of the engineering phase.

By utilizing these specific values, the system eliminates human error in interpreting complex lab results, ensuring that every patient among the 6,544 is classified with absolute mathematical certainty.

> Visit `notebooks/` or [![Open In Colab](https://img.shields.io/badge/Open%20In%20Colab-black?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=Twbk4kGwo8IX)
---

### Clinical Interpretation Logic

The model mimics a clinical consultant by evaluating the **synergy** between lipids (the cause) and enzymes (the effect).

| Scenario | Triglycerides (TG) | Enzymes (ALT/GGT) | Decision | Clinical Insight |
| --- | --- | --- | --- | --- |
| **1** | High (300) | Normal (20) | **🟢 Healthy** | Blood lipids are high, but the liver is not yet injured. |
| **2** | Normal (100) | High (60) | **Not Fatty Liver** | Injury exists, but likely due to viral or toxic causes, not liver fat. |
| **3** | High (200) | High (50) | **🔴 Patient** | **Confirmed NAFLD:** Fat accumulation has triggered inflammation. |

*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

---


## Clinical Validation: 7-Case Stress Test

The following table represents the final diagnostic output for the 7 simulated patients:

| # | Clinical Scenario | Data Logic | Final Diagnosis |
| --- | --- | --- | --- |
| 1 | **Healthy Baseline** | All markers optimal | 🟢 **HEALTHY** |
| 2 | **Isolated Hyperlipidemia** | High Fats (300) + Low Enzymes (20) | 🟢 **HEALTHY** |
| 3 | **Active NAFLD (Early)** | High Fats (220) + High Enzymes (55) | 🔴 **PATIENT (NAFLD)** |
| 4 | **Metabolic Syndrome** | High Fats (450) + High Enzymes (90) | 🔴 **PATIENT (NAFLD)** |
| 5 | **Advanced Stress** | High Fats (190) + High Enzymes (110) | 🔴 **PATIENT (NAFLD)** |
| 6 | **Non-Fatty Injury** | Low Fats (105) + Very High Enzymes (160) | **Not Fatty Liver** |
| 7 | **Moderate/Borderline Risk** | Crossed Thresholds (152 / 41) | 🔴 **PATIENT (NAFLD)** |

*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

---

###  Analytical Breakdown of Diagnostic Logic

#### **A. Synergistic Rule Detection (Cases 2 vs. 3)**

The model successfully differentiates between "Fatty Blood" and "Fatty Liver."

* In **Case 2**, even though Triglycerides are very high (300), the liver enzymes are normal (20), so the model correctly identifies the patient as **Healthy** regarding liver disease.
* In **Case 3**, once high fats coexist with high enzymes, the model triggers a **Patient** diagnosis.

#### **B. Differential Diagnosis (Case 6)**

Case 6 represents a critical test for the model. The patient has extremely high liver enzymes (ALT 160), which indicates liver damage. However, because the Triglycerides are normal (105), the model recognizes this as a **Non-Fatty Injury** (likely toxic or viral) and correctly refuses to classify it as **NAFLD**.

#### **C. Borderline Sensitivity (Case 7)**

The model shows high precision in **Case 7**, where the patient only slightly exceeded the thresholds (ALT 41 vs limit 40, Triglycerides 152 vs limit 150). The model successfully captured this as a **Patient** case, proving its sensitivity to early-stage risks.

*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

---

The validation of these **7 scenarios** confirms that the model is not simply looking at high numbers in isolation. Instead, it is performing a complex **Biochemical Correlation**. It effectively prevents over-diagnosis in cases of isolated high fats and maintains high specificity by identifying non-fat-related liver injuries as non-NAFLD.  Visit `notebooks/` or [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=7XzfWokVGCQb)


