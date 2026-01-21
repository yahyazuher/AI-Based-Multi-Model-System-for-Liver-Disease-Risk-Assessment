# Fatty Liver Diagnosis Model (NAFLD)

This section is dedicated to the detection of **Non-Alcoholic Fatty Liver Disease (NAFLD)**. The model distinguishes between general hyperlipidemia (high blood fats) and actual liver injury caused by hepatic steatosis. It bridges biochemical laboratory data with advanced clinical logic using the **XGBoost** algorithm, stored as `fatty_liver_model.pkl`.

---

### Dataset Overview

| Name | Database Location | Function |
| --- | --- | --- |
| **fatty_liver_model.pkl** | `models/` | The trained model containing the optimized weights for NAFLD detection. |
| **train_fatty_liver.py** | `code/` | Source code for data merging (`SEQN` logic) and model training. |
| **test_fatty_liver.py** | `code/` | Source code dedicated to testing and evaluating the model performance. |
| **FattyLiver.csv** | `data/processed/` | Engineered dataset from NHANES 2013-2014 cycles containing 6,544 patient records. |
| **XGBoost.md** | `docs/` | Technical documentation of the underlying boosting mechanism. |

---

### Data Engineering & Integration Strategy

The integrity of the `FattyLiver.csv` dataset is built on a surgical data integration strategy. The primary challenge involved merging records from three different laboratory files where patient counts were inconsistent (e.g., Biochemistry: 6,946 vs. CBC: 9,249).

#### **The "SEQN" Surgical Merge**

To eliminate **Data Shift** errors—where one patient’s results are incorrectly mapped to another—we utilized the **SEQN (Sequence Number)** as a Primary Key.

* **Method:** Data was aligned in **LibreOffice Calc** using the `VLOOKUP` function.
* **Refinement:** Any patient missing one of the three core laboratory components was purged from the training set, ensuring **100% data integrity** for every record.

---

### Feature Selection: The "Biological Fingerprint"

The dataset was reduced by over **50%** to eliminate statistical noise and prevent **Multicollinearity**, focusing only on markers with high clinical weight.

| Feature Type | Markers | Engineering Logic |
| --- | --- | --- |
| **Inflammation** | ALT, AST, GGT, ALP | Core indicators of active hepatocellular injury and bile duct stress. |
| **Metabolic Fuel** | Triglycerides (TG), Glucose | Identifies the excess lipids available for liver fat storage. |
| **The Veto Factor** | **Platelets** | A critical marker for detecting underlying Fibrosis/Cirrhosis. |
| **Synthetic Capacity** | Albumin, Bilirubin | Evaluates the liver's ability to manufacture protein and clear toxins. |

---

### Positional Logic: The "Mathematical Matrix"

The `FattyLiver_Model.pkl` file is a **Mathematical Matrix**. It does not interpret column headers; instead, it relies strictly on **Positional Indices** (the order of data).

**⚠️ Critical Execution Requirement:**
Feeding data in the wrong sequence (e.g., placing Glucose in the Triglycerides slot) will lead to a total diagnostic failure. Data must be submitted in this exact order:

```python
# The Immutable Positional Array:
[
    'Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol', 
    'Creatinine', 'Glucose', 'GGT', 'Bilirubin', 
    'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL'
]

```

---

### Clinical Interpretation Logic

The model mimics a clinical consultant by evaluating the **synergy** between lipids (the cause) and enzymes (the effect).

| Scenario | Triglycerides (TG) | Enzymes (ALT/GGT) | Decision | Clinical Insight |
| --- | --- | --- | --- | --- |
| **1** | High (300) | Normal (20) | **🟢 Healthy** | Blood lipids are high, but the liver is not yet injured. |
| **2** | Normal (100) | High (60) | **🟢 Healthy** | Injury exists, but likely due to viral or toxic causes, not fat. |
| **3** | High (200) | High (50) | **🔴 Patient** | **Confirmed NAFLD:** Fat accumulation has triggered inflammation. |

---

### Virtual Clinic: 7 Case Analysis

The following clinical scenarios were designed to test the model's stability and its ability to distinguish between "Blood Fat" and "Liver Fat."

| Case | Clinical Description | Result | Risk % | Clinical Interpretation |
| --- | --- | --- | --- | --- |
| **1** | [Pending Data...] | **TBD** | **--%** | [Awaiting Input] |
| **2** | [Pending Data...] | **TBD** | **--%** | [Awaiting Input] |
| **3** | [Pending Data...] | **TBD** | **--%** | [Awaiting Input] |
| **4** | [Pending Data...] | **TBD** | **--%** | [Awaiting Input] |
| **5** | [Pending Data...] | **TBD** | **--%** | [Awaiting Input] |
| **6** | [Pending Data...] | **TBD** | **--%** | [Awaiting Input] |
| **7** | [Pending Data...] | **TBD** | **--%** | [Awaiting Input] |

---

### Technical Note for Developers

* **Automated Tuning:** The **Automated Hyperparameter Tuning Strategy** used for this model is documented in the Colab notebook under the **"Fatty Liver Model"** cell.
* **Dataset Cleaning:** The training file `FattyLiver_Learning_db.csv` is the processed version of the original NHANES data. The Colab code is responsible for the final removal of the `SEQN` column and non-numeric noise before training.
* **Parallel Processing:** Training was executed with `n_jobs=-1` to optimize performance on multi-core processors.

> **Scientific Insight:** The model identifies that lifestyle markers like **Uric Acid** and **Glucose** act as "amplifiers" for fatty liver risk when combined with elevated **GGT**, forming a complete metabolic profile for the patient.

---

**Would you like me to analyze the 7 clinical cases now so we can fill in the Virtual Clinic table?**
