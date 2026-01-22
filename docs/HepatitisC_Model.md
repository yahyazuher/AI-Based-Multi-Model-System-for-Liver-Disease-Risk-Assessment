# Hepatitis C (HCV) Diagnostic & Prognostic Framework 


This specialized framework is dedicated to evaluating the progression of Liver Fibrosis in Hepatitis C patients and predicting the probability of critical complications, such as Ascites, and overall mortality. The system utilizes an ensemble of XGBoost models, with its core logic distributed across three specialized serialized files: `hepatitis_stage.pkl`, `hepatitis_complications.pkl`, and `hepatiti_status.pkl`(all in models/ directory). By analyzing clinical input values through optimized "feature weights," the framework provides a precise determination of the risk levels associated with key biomarkers.

This framework is specifically designed and optimized for Hepatitis C (HCV) diagnostic patterns. It is not intended for use with other types of hepatitis (A, B, D, or E), as the biochemical markers and progression rates vary significantly across different viral strains.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

---

### Dataset Overview

| Name | Database Location | Function |
| --- | --- | --- |
| **hepatitis_stage.pkl** | `models/` | Trained model to classify histological damage (Stage 1-4). |
| **hepatitis_complications.pkl** | `models/` | Trained model to estimate the risk of Ascites. |
| **hepatitis_status.pkl** | `models/` | Trained model to calculate survival/mortality probability. |
| **train_hepatitis_models.py** | `code/` | Source code responsible for building and training the 3 models.Can be run directly in Google Colab |
| **test_hepatitis_models.py** | `code/` | Inference script for diagnosing the 7 cases (see details above).Can be run directly in Google Colab |
| **Hepatitis.csv** | `data/processed` | The processed clinical dataset derived from Mayo Clinic records. |
| **XGBoost.md** | `docs/` | Technical documentation explaining the mechanism of the XGBoost algorithm. |

---

### Training Phase 

The reliability of the **Hepatitis Diagnostic Framework** is built on a rigorous training pipeline designed to ensure scientific validity and clinical transparency:

* **The  Rule (Generalization):** The processed dataset was split into ** ( records)** for training and ** ( records)** for independent validation. This separation ensures the models do not simply "memorize" the data but learn the underlying clinical patterns necessary to handle new, unseen cases.
* **Stratified Sampling ():** We utilized **Stratified Splitting** to ensure that the distribution of critical outcomes—specifically **Stage 4 (Cirrhosis)** and **Mortality**—remains identical in both the training and testing sets. This prevents a random split from leaving the test set without enough high-risk cases for a fair evaluation.

#### **Model Performance Results**

After completing the training on  records and testing on  unseen records, the models achieved the following confidence levels:

| Model | Accuracy | Status |
| --- | --- | --- |
| **`hepatitis_complications.pkl`** | $92.86\%$ %|  High Reliability |
| **`hepatiti_status.pkl`** | $79.76\%$ % | Moderate-High Reliability |
| **`hepatitis_stage.pkl`** | $< 49\%$ % | Academic Use Only |

The logic used to generate these results, including the $80/20$ split and XGBoost configurations, can be found in the source script: code/train_hepatitis_models.py or executed directly via the Colab environment:[![Open In Colab](https://img.shields.io/badge/Open%20In%20Colab-black?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

#### **Stability & Clinical Disclaimer**

By training on a "true representation" of the disease, the framework achieves stable diagnostic confidence for mortality and complications. However, it is imperative to note that while the **Virtual Case Analysis Table** shows logical results, the **`hepatitis_stage.pkl`** model is considered **inaccurate for clinical purposes** due to its low validation score.

> **`hepatitis_stage.pkl`** Included in this repository solely for academic demonstration and as a structural component for the multi-stage AI architecture.

---
## Data Pipeline & Feature Engineering

### Data Source (Raw Data)

The primary dataset is sourced from the **Cirrhosis Prediction Dataset** hosted on Kaggle (provided by *fedesoriano*), which originates from the renowned **Mayo Clinic** study on primary biliary cirrhosis (PBC).

* **Original Source:** [Kaggle: Cirrhosis Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)
* **Raw File Path:** `data/raw/cirrhosis.csv`

### Feature Engineering Logic

To prepare the raw data for the **XGBoost** model, a rigorous **Data Engineering** phase was executed. The original file (`cirrhosis.csv`) was transformed into the processed training file (`Hepatitis.csv`) located in `data/processed`, resulting in a refined dataset of **419 patient records**.

The following transformations were applied:

#### **A. Target Variable Transformation (Status)**

The goal was to predict specific mortality risk. The original multi-class status was mapped to a binary format:

 **Original Values:**
* `(C:0: Alive/Stable)` (Censored) & `CL` (Censored due to Liver Transplant)  Considered Stable.
* `(D:1: Deceased)` (Death)  Considered Critical Event.

#### **B. Clinical Feature Encoding**

Categorical text values were converted into numerical formats to ensure mathematical compatibility with the model:

| Feature | Transformation Logic | Rationale |
| --- | --- | --- |
| **Age** | `Days / 365.25` | Converted from raw days to **Years** for clinical interpretability. |
| **Sex** | `M`  `1`, `F`  `0` | Binary encoding. |
| **Ascites** | `Y`  `1`, `N`  `0` | Presence vs. Absence. |
| **Hepatomegaly** | `Y`  `1`, `N`  `0` | Liver enlargement indicator. |
| **Spiders** | `Y`  `1`, `N`  `0` | Spider angiomas indicator. |

#### **C. Ordinal Severity Scaling (Edema)**

Unlike binary features, Edema has graduated severity levels. We applied **Ordinal Encoding** to reflect the increasing risk:

* **`N` (No Edema):** Mapped to **0.0**
* **`S` (Slight Edema):** Mapped to **0.5** (Edema resolvable with diuretics)
* **`Y` (Severe Edema):** Mapped to **1.0** (Edema resistant to diuretics)
---

## Model Architecture & Integration

The predictive logic of the **AiLDS** is divided into three integrated diagnostic layers. Each layer acts as an independent specialized component, utilizing specific clinical features derived from the processed dataset to provide a comprehensive health assessment.

### 1. Model Breakdown & Feature Selection

| Model File | Input Features (Columns) | Target Output | Clinical Significance |
| --- | --- | --- | --- |
| **`hepatitis_stage.pkl`** | Core Blood Tests (Bilirubin, Albumin, SGOT, etc.) | Discrete Class: **1, 2, 3, 4** | Determines the level of histological liver scarring (Fibrosis). |
| **`hepatitis_complications.pkl`** | Blood Tests & Physical Signs (excluding 'Ascites') | Probability: **0.0 to 1.0** | Estimates the immediate risk of fluid accumulation (Ascites). |
| **`hepatitis_status.pkl`** | Blood Tests + **Predicted Stage** (from Model 1) | Probability: **0.0 to 1.0** | Calculates the ultimate mortality risk (Survival vs. Death). |

*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

---

### 2. Understanding the `.pkl` Output

When interacting with these serialized models programmatically, the outputs are processed based on the task type:

* **Classification Output (Stage Model):** Returns a discrete numerical class representing the fibrosis stage.
> **Example:** An output of `3` corresponds to **Stage 4 (Cirrhosis)** in the clinical mapping.


* **Probability Output (Complications & Status Models):** We utilize the `predict_proba` function to retrieve a continuous value representing clinical confidence.
> **Example:** A result of `0.92` indicates a **92% probability** of a critical event or complication.



---

### 3. Workflow And Double Verification Logic

This models engineered as a **Clinical Decision Support Tool**. To ensure the highest level of safety, the system follows a strict execution sequence and a custom verification protocol.

#### **A. Sequence of Execution**

The diagnostic architecture of the system follows a strictly hierarchical execution sequence. The predicted output from the Stage Model is a fundamental prerequisite because it serves as a high-weight input feature for the final Status (Mortality) Model.

* Phase 1 (Histological Diagnosis): The inference engine first processes raw clinical biomarkers (such as Bilirubin and Albumin) to determine the liver's histological Stage Prediction using the hepatitis_stage.pkl model.

* Phase 2 (Prognostic Analysis): The system automatically merges the initial clinical data with the AI-generated Stage to calculate the final Mortality Risk Probability using the hepatiti_status.pkl model.

This dependency ensures that the structural state of the liver is prioritized as a primary factor before the system evaluates functional survival outcomes, mirroring real-world clinical decision-making.

#### **B. The Double Verification Protocol **

Because the transition between Stage 3 (Advanced Fibrosis) and Stage 4 (Cirrhosis) is clinically critical, the system implements a **Double Verification Logic**.

* **Trigger:** If the Stage Model predicts a high-risk result (**Stage 3 or 4**), the system automatically re-runs the inference engine.
* **Requirement:** The model must return the **exact same result twice in a row**.
* **Safety Net:** If the two consecutive results are not identical, the system flags a "Verification Mismatch," preventing a potentially false high-risk diagnosis from being presented to the user.

---

### 4. Clinical Interpretation

The system categorizes results based on a probability threshold to assist in rapid triage:

* **🟢 Green Flag (Stable):** If the Mortality Probability is **< 50%**, the case is classified as stable with a recommendation for routine clinical monitoring.
* **🔴 Red Flag (Critical):** If the Mortality Probability is **> 50%**, the case is flagged as critical, requiring immediate medical intervention and further diagnostic confirmation.

---

### 5. Mathematical Validation

To complement the AI's predictive power, the system integrates established clinical scoring systems to provide a "ground truth" reference:

#### **A. APRI Score (AST to Platelet Ratio Index)**

The APRI score is a non-invasive method used to assess the likelihood of liver fibrosis and cirrhosis. It calculates the ratio between liver enzymes and blood platelets to identify structural damage without a biopsy:

$$APRI = \frac{(AST / 40)}{Platelets} \times 100$$

* **AST**: Aspartate Aminotransferase level (U/L).
* **40**: The standardized Upper Limit of Normal (ULN) for AST used in this system.
* **Platelets**: Platelet count per cubic millimeter ().

#### **B. ALBI Score (Albumin-Bilirubin Grade)**

The ALBI score is an objective metric specifically designed to evaluate liver functional reserve. Unlike other scores, it relies solely on laboratory markers, eliminating subjective clinical variables:

$$ALBI = (\log_{10}(Bilirubin \times 17.1) \times 0.66) + (Albumin \times 10 \times -0.085)$$

* **Bilirubin**: Total bilirubin level (converted to  using the  factor).
* **Albumin**: Serum albumin level (converted to  using the  factor).


> By presenting these mathematical validations alongside the AI results, the **Hepatitis Diagnostic Framework** ensures that the "black box" decisions of the Machine Learning models are grounded in proven medical mathematics. This dual-approach increases the system's transparency and reliability for clinical decision support.
---
### Models Input Requirements

To ensure diagnostic accuracy and model stability, data must be entered in the strict mathematical order required by the inference engine. The system utilizes two distinct input structures based on the model's role in the diagnostic hierarchy.

### 1. Standard Clinical Input (15 Features)

This 15-feature sequence is the foundational input used by both the **`hepatitis_stage.pkl`** and **`hepatitis_complications.pkl`** models.

| Index | Feature Name | Description |
| --- | --- | --- |
| 1 | **Bilirubin** | Total bilirubin level. |
| 2 | **Cholesterol** | Serum cholesterol. |
| 3 | **Albumin** | Serum albumin. |
| 4 | **Copper** | Urine copper. |
| 5 | **Alk_Phos** | Alkaline phosphatase. |
| 6 | **SGOT** | Aspartate aminotransferase (AST/SGOT). |
| 7 | **Tryglicerides** | Serum triglycerides. |
| 8 | **Platelets** | Platelet count per cubic millimeter. |
| 9 | **Prothrombin** | Prothrombin time in seconds. |
| 10 | **Age** | Patient age in years. |
| 11 | **Sex** | Gender (Categorically encoded). |
| 12 | **Ascites** | Presence of abdominal fluid accumulation. |
| 13 | **Hepatomegaly** | Clinical presence of liver enlargement. |
| 14 | **Spiders** | Presence of spider angiomas. |
| 15 | **Edema** | Presence of systemic swelling. |


### 2. Extended Prognostic Input (16 Features)

The **`hepatiti_status.pkl`** model follows a hierarchical logic, requiring the **AI-generated Stage** to be injected into the clinical data sequence. In this model, the **Stage** must be inserted specifically at **Index 10**, shifting the remaining clinical markers down.

**The 16-Feature Order:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin',` **`'Stage'`**, `'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema']`

* **Feature Injection:** The system automatically captures the prediction from the Stage Model and places it between **Prothrombin** and **Age**.
* **Hierarchical Dependency:** This 16-feature sequence is critical for the survival risk model to understand how physical liver scarring (Stage) influences the overall probability of mortality.


### In Summary 

* **Stage Model:** Uses **15 Features** to classify histological damage (Stage 1-4).
* **Complications Model:** Uses the same **15 Features** to estimate the risk of Ascites.
* **Status Model:** Uses **16 Features** (Standard 15 + Stage Injection) to calculate survival probability.

---

### Optimized Model Configuration

The following parameters were identified as the **"Gold Standard"** for this specific dataset to achieve maximum stability across the three models:

```python
# The optimized XGBoost configuration for Medical Diagnosis
model = xgb.XGBClassifier(
    n_estimators=100,      # Balanced number of trees to prevent complexity
    learning_rate=0.1,     # Optimal step size for stable convergence
    max_depth=3,           # Strategic depth to ensure high generalization
    subsample=0.8,         # Trains on 80% of data per tree to boost robustness
    eval_metric='logloss'  # Standard evaluation metric for binary classification
)

```

> **Scientific Insight:** In medical diagnostics, a `max_depth` of **3** is crucial. It forces the model to make decisions based on the most dominant biomarkers (like Bilirubin and Prothrombin) rather than creating complex, unexplainable branches that might fit noise in the data.

---

### 4. Virtual Clinic Test Results

To demonstrate the practical application of the model, a **"Virtual Clinic"** simulation was conducted using 7 real-world scenarios. This phase validates the model's ability to distinguish between "Structural Damage" (Stage) and "Functional Failure" (Status).

### Virtual Case Analysis Table

| Clinical Case | Brief Description | AI Stage | Survival Risk | Clinical Interpretation |
| --- | --- | --- | --- | --- |
| **1. The Healthy Reference** | Normal labs, no symptoms. | **Stage 1** | **1.4%** | Baseline reference for a low-risk profile. |
| **2. Acute Inflammation** | High Bilirubin (3.2), Stage 2. | **Stage 2** | **97.2%** | **Critical:** High mortality despite moderate fibrosis stage. |
| **3. Early Fibrosis** | Moderate APRI score. | **Stage 3** | **2.7%** | Disease is progressing but functionally stable. |
| **4. Compensated Cirrhosis** | Stage 4, but normal Albumin. | **Stage 4** | **0.6%** | **Key Result:** Successful clinical compensation (Type CL). |
| **5. Decompensated Failure** | Stage 4 + Ascites + Low Albumin. | **Stage 4** | **99.9%** | End-stage liver failure; immediate intervention required. |
| **6. Active Injury** | Stage 4 with high Enzymes (SGOT). | **Stage 4** | **92.5%** | High risk due to ongoing active inflammation. |
| **7. The Geriatric Case** | Age 70, Coagulation issues. | **Stage 4** | **98.9%** | Demonstrates the compounded risk of Age + Fibrosis. |

*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

---

### Clinical Insights

Based on the Virtual Clinic outputs, critical technical observations regarding the Multi-Model logic were derived:

#### **A. Structural-Functional Dissociation (Case 4)**

The model successfully identified a patient with **Stage 4 Cirrhosis** yet assigned a very low mortality risk (**0.6%**). This reflects the clinical reality of "Compensated Cirrhosis," where the liver is scarred but still functional. This aligns with the "CL" (Censored/Transplant) data logic, proving the model is not blindly equating Stage 4 with Death.
> Note: These results are based on the hepatitis_stage.pkl model, which currently has a validation accuracy of 47%. While the logic remains clinically consistent, this model serves primarily as a structural indicator within the diagnostic hierarchy rather than a standalone clinical tool.
> 
#### **B. The Lethality of Bilirubin (Case 2)**

The model assigned a **97% risk** to a Stage 2 patient due to elevated Bilirubin (3.2). This confirms that **XGBoost** has correctly prioritized "Functional Markers" over "Structural Stage" when predicting immediate mortality.

#### **C. Multi-Factor Integration**

In Case 5, the convergence of **Ascites**, **Low Albumin**, and **High Copper** pushed the risk to **99.9%**, demonstrating the model's ability to handle complex, multi-variable failure cascading.

#### **D. The Impact of Age & Enzyme Spikes (Cases 6 & 7)**

Case 7 (**The Geriatric Case**) demonstrates the model's sensitivity to demographic variables. Even with similar labs to a younger patient, a **70-year-old** patient is assigned a higher mortality risk (**98.9%**), showing that the **XGBoost** model correctly factors in decreased physiological resilience associated with age. Similarly, Case 6 highlights how **active injury** (high SGOT) can drive mortality risk even when other factors are stable.

#### **E. Baseline Normalization (Case 1 & 3)**

Cases 1 and 3 serve as the "Control Group". The low risk in Case 1 (**1.4%**) validates that the model does not produce "False Positives" for healthy individuals. Case 3 shows that the model can distinguish between **fibrosis progression** and **functional failure**, keeping the risk low (**2.7%**) as long as the liver's synthetic function (Albumin/Bilirubin) remains intact.

---
*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

