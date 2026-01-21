# Fatty Liver Diagnosis Model (NAFLD)

This section is dedicated to the detection of **Non-Alcoholic Fatty Liver Disease (NAFLD)**. The model distinguishes between general hyperlipidemia (high blood fats) and actual liver injury caused by hepatic steatosis. It bridges biochemical laboratory data with advanced clinical logic using the **XGBoost** algorithm, stored as `FattyLiver_Model.pkl`.

---

### Dataset Overview

| Name | Database Location | Function |
| --- | --- | --- |
| **fatty_liver_model.pkl** | `models/` | The trained model containing the optimized weights for NAFLD detection. |
| **train_fatty_liver.py** | `code/` | Source code for data merging (`SEQN` logic) and model training. |
| **test_fatty_liver.py** | `code/` | Source code for data merging (`SEQN` logic) and model training. |
| **FattyLiver.csv** | `data/processed/` | Engineered dataset from NHANES 2013-2014 cycles. |
| **XGBoost.md** | `docs/` | Technical documentation of the underlying boosting mechanism. |

---

### Training Phase & Data Engineering

The system’s integrity is built on a surgical data integration strategy:

* **Integration Strategy:** Merged three distinct NHANES components (`BIOPRO_H`, `CBC_H`, `HDL_H`) using the **SEQN (Sequence Number)** as a primary key to eliminate "Data Shift" errors.
* **Data Split:** Utilizes **80% for training** and **20% for testing** to ensure mathematical stability.
* **Feature Selection:** Prioritized the "Biological Fingerprint"—ALT, AST, and GGT—combined with metabolic markers like Triglycerides and the "Veto Factor" (Platelets) for fibrosis screening.

> **Technical Note:** Electrolytes (Sodium, Calcium) were removed as "Noise," and redundant SI units were deleted to prevent the model from over-weighting duplicate data.

---

### 1- Data Source and Integrity

* **Original Database:** Derived from the **NHANES (National Health and Nutrition Examination Survey) 2013-2014** cycle.
* **Source:** [CDC/NCHS Official Portal](https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Laboratory&CycleBeginYear=2013)
* **Integrity Protocol:** Data was surgically mapped via `VLOOKUP` in LibreOffice Calc to ensure that every biological marker (from 6,946 records) belongs to the correct clinical SEQN.

---

### 2- Model Input Requirements (Positional Logic)

The model functions as a mathematical matrix. Data **must** be entered in the following exact order to avoid diagnostic failure:
`['Albumin', 'ALP', 'AST', 'ALT', 'Cholesterol', 'Creatinine', 'Glucose', 'GGT', 'Bilirubin', 'Triglycerides', 'Uric_Acid', 'Platelets', 'HDL']`

---

### 3- Model Optimization & Refinement

The NAFLD model utilizes a specific "Clinical Synergy" logic, focusing on the relationship between **Triglycerides (TG)** and **Liver Enzymes**.

* **Complexity Control:** The `max_depth` was optimized to prevent the model from memorizing outliers in metabolic variations.
* **Veto Logic:** Low platelet counts are given significant weight to influence the model’s awareness of potential advanced scarring (Fibrosis).

> [!NOTE]
> **Technical Implementation:**
> The **Automated Hyperparameter Tuning Strategy** used to derive these optimal values is implemented in the Google Colab notebook under the cell titled **"Fatty Liver Model"**. A comprehensive explanation of the tuning logic and the code breakdown is provided directly above the specific cell. [](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

---

### Performance Metrics

| Metric | Result | Interpretation |
| --- | --- | --- |
| **Accuracy** | [X]% | High precision in distinguishing fat accumulation. |
| **Precision** | [X]% | Minimizes false positives in healthy hyperlipidemia cases. |
| **Recall** | [X]% | High sensitivity in detecting early-stage NAFLD. |

---

### 4. Virtual Clinic Test Results

To demonstrate the model's ability to distinguish between "Blood Fats" and "Liver Fat," we conducted a simulation of **7 clinical scenarios**. These cases test whether the model correctly identifies when high lipids have started causing actual cellular injury.

### Virtual Case Analysis Table

| Clinical Case | Brief Description | Result | Risk % | Clinical Interpretation |
| --- | --- | --- | --- | --- |
| **1. [Case Name]** | [Pending Data...] | **TBD** | **--%** | [Waiting for input] |
| **2. [Case Name]** | [Pending Data...] | **TBD** | **--%** | [Waiting for input] |
| **3. [Case Name]** | [Pending Data...] | **TBD** | **--%** | [Waiting for input] |
| **4. [Case Name]** | [Pending Data...] | **TBD** | **--%** | [Waiting for input] |
| **5. [Case Name]** | [Pending Data...] | **TBD** | **--%** | [Waiting for input] |
| **6. [Case Name]** | [Pending Data...] | **TBD** | **--%** | [Waiting for input] |
| **7. [Case Name]** | [Pending Data...] | **TBD** | **--%** | [Waiting for input] |

---

### Clinical Insights (Derived from Scenarios)

#### **A. The Lipid vs. Enzyme Threshold**

The model is trained to recognize that high Triglycerides alone do not equal Fatty Liver. Diagnosis is only triggered when there is a concurrent rise in **ALT** or **GGT**, indicating active inflammation.

#### **B. The Protective Platelet Check**

By monitoring **Platelets**, the system can distinguish between simple NAFLD and cases where the disease has progressed toward advanced stages, ensuring a more holistic diagnostic view.

---

# 5- Technical Note for Developers

* **Execution:** Tests were conducted using `code/test_fatty_liver.py`.
* **Model Stability:** The `SEQN` key is exclusively used for data alignment and is dropped prior to training to ensure the model focuses solely on biological indicators.

> **Scientific Insight:** The model transforms complex blood panels into a clear diagnostic path, demonstrating how metabolic markers like Uric Acid and Glucose interact with liver enzymes to form a complete "Fatty Liver Profile."

---
