# Hepatitis C (HCV)


This specialized framework is dedicated to evaluating the progression of Liver Fibrosis in Hepatitis C patients and predicting the probability of critical complications, such as Ascites, and overall mortality. The system utilizes an ensemble of XGBoost models, with its core logic distributed across three specialized serialized files: `hepatitisC_stage_model.pkl`, `hepatitisC_complications.pkl`, and `hepatitiC_status_model.pkl`(all in models/ directory). By analyzing clinical input values through optimized "feature weights," the framework provides a precise determination of the risk levels associated with key biomarkers.

This framework is specifically designed and optimized for Hepatitis C (HCV) diagnostic patterns. It is not intended for use with other types of hepatitis (A, B, D, or E), as the biochemical markers and progression rates vary significantly across different viral strains.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=3oQ6HJTl0F1V)

---

### Dataset Overview

| Name | Database Location | Function |
| --- | --- | --- |
| **hepatitisC_stage_model.pkl** | `models/` | Trained model to classify histological damage (Stage 1-3). |
| **hepatitisC_complications.pkl** | `models/` | Trained model to estimate the risk of Ascites. |
| **hepatitiC_status_model.pkl** | `models/` | Trained model to calculate survival/mortality probability. |
| **train_hepatitisC_models** | `notebooks/code/` | Source code responsible for building and training the 3 models.Can be run directly in Google Colab |
| **test_hepatitisC_models** | `notebooks/code/` | Inference script for diagnosing the 7 cases (see details above).Can be run directly in Google Colab |
| **`HepatitisC.csv`** **`hepatitisC_status.csv`**  **`hepatitisC_stage.csv`** | `data/processed` | The processed clinical dataset derived from Mayo Clinic records. |
| **XGBoost.md** | `docs/` | Technical documentation explaining the mechanism of the XGBoost algorithm. |

---

### Training Phase 

The reliability of the **Hepatitis Diagnostic Framework** is built on a rigorous training pipeline designed to ensure scientific validity and clinical transparency:

* The processed dataset was split into ** ( records)** for training and ** ( records)** for independent validation. This separation ensures the models do not simply "memorize" the data but learn the underlying clinical patterns necessary to handle new, unseen cases.
* We utilized **Stratified Splitting** to ensure that the distribution of critical outcomes—specifically **Stage 4 (Cirrhosis)** and **Mortality**—remains identical in both the training and testing sets. This prevents a random split from leaving the test set without enough high-risk cases for a fair evaluation.

#### **Model Performance Results**

After completing the training on  records and testing on  unseen records, the models achieved the following confidence levels:

| Model | Accuracy | Status |
| --- | --- | --- |
| **`hepatitisC_complications.pkl`** | $92.86\%$ %|  High Reliability |
| **`hepatitiC_status_model.pkl`** | $71.43\%$ % | Moderate-High Reliability |
| **`hepatitisC_stage_model.pkl`** | $62.50\%$ % | Academic Use |

The logic used to generate these results, including the $80/20$ split and XGBoost configurations, can be found in the source script: `notebooks/code/train_HC_models` or executed directly via the Colab environment: [![Open In Colab](https://img.shields.io/badge/Open%20In%20Colab-black?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=3oQ6HJTl0F1V)


---

##  Data Pipeline & Feature Engineering

### Data Sources (Raw Data)

The system relies on two distinct primary datasets. these datasets were selected for their clinical relevance and statistical significance.

#### **1. Primary Biliary Cirrhosis (PBC) Dataset**

* **Purpose:** Used to train the **Mortality Risk (Status) and (Complications)** models.
* **Origin:** Sourced from the renowned **Mayo Clinic** study on primary biliary cirrhosis (PBC).
* **Raw File Path:** `data/raw/cirrhosis.csv`
* **Source:** [Kaggle: Cirrhosis Prediction Dataset (fedesoriano)](https://www.kaggle.com/datasets/fedesoriano/cirrhosis-prediction-dataset)

#### **2. Hepatitis C (HCV) Clinical Dataset**

* **Purpose:** Used to train the **Fibrosis Staging (Stage)** model.
* **Origin:** A comprehensive clinical dataset comprising **8,323 patient records**, specifically curated to track Hepatitis C progression and fibrosis stages.
* **Raw File Path:** `data/raw/master8323pationt.csv`
* **Source:** Publicly available clinical dataset (Aggregated from open-access medical repositories for research purposes).
  
---

## 1. Model Optimization & Dataset Curation

This section details the rigorous data engineering strategies applied to ensure the models learn from valid clinical pathology rather than statistical noise.

### 1.1. Mortality Prediction Model (Status Classification)

* Predict patient survival outcomes (**Deceased** vs. **Alive**).
* **Initial Baseline:** Wen the preliminary model was trained on the full raw dataset, achieving an accuracy of **79.76%**. However, a data quality audit revealed that **105 records** contained missing values in critical biomarkers (*Triglycerides, SGOT, Alk_Phos, Copper, and Cholesterol*). The initial high accuracy was artificially inflated. The model was learning patterns from imputed or missing data structures rather than actual clinical pathology, leading to unreliable real-world predictions.
* **Refinement Strategy (Listwise Deletion):** To ensure the model trains on "Ground Truth" data only, all records with missing critical values were strictly removed.
* **Final Dataset:** data/processed/hepatitisC_status.csv ,with 312 case (187 Stable/Alive vs. 125 Deceased).

* Finally the refined model achieved an accuracy of **71.43%**. While numerically lower than the baseline, this metric represents .The model is no longer "guessing" on missing data. Furthermore, the class balance (approx. 60/40) encourages the model to adopt a slightly "optimistic" baseline (favoring survival) while remaining sensitive to critical deterioration markers. To see confusion matrix go to:

---

### 1.2. Disease Staging Model (Severity Classification)

* **Objective:** Classify the histological stage of Hepatitis C (**Stages 1–4**) using the large-scale `master8323patient.csv` dataset.
* **Challenge (Class Overlap):** Initial attempts to classify all four stages individually ( per class) resulted in poor performance (**Accuracy: 45%**). Confusion Matrix analysis indicated significant misclassification between **Stage 2** and **Stage 3**.
* **Clinical Insight:**


Stages 2 and 3 share highly similar biochemical profiles (blood test values), making them mathematically indistinguishable for the classifier in high-dimensional space.

* **Optimization Strategy (Class Merging):** To improve model reliability, the classes were restructured into three distinct severity tiers. The "Mid-Stage" tier was engineered by geometrically aggregating patients from Stages 2 and 3.
* **Final Class Structure:**
1. **Grade 1 (Early Stage):** 413 Patients.
2. **Grade 2 (Intermediate Stage):** 413 Patients (Composed of 207 Stage 2 + 207 Stage 3).
3. **Grade 3 (Advanced/Cirrhotic):** 413 Patients (Originally Stage 4).


* **Result:** This strategic restructuring increased the model accuracy to **62.50%**. This is considered the optimal ceiling for this specific dataset, as it successfully distinguishes between **Early**, **Intermediate**, and **Advanced** disease progression, effectively mitigating the noise caused by the Stage 2/3 biological overlap.
  
---

### 1.3 Feature Engineering Logic

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

| Model File | Input Features | Target Output | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **`hepatitisC_stage_model.pkl`** | Blood Tests + **`APRI`**, **`Bilirubin_Albumin`**, **`Copper_Platelets`** | Discrete Class: **1, 2, 3** | Determines the level of histological liver scarring (Fibrosis). |
| **`hepatitisC_status_model.pkl`** | Blood Tests + **`APRI`**, **`ALBI_Score`**, **`Bili_Alb_Ratio`** | Probability: **0.0 to 1.0** | Calculates the ultimate mortality risk (Survival vs. Death). |
| **`hepatitisC_complications.pkl`** | Blood Tests Only (Raw Clinical Markers) | Probability: **0.0 to 1.0** | Estimates the immediate risk of fluid accumulation (Ascites). |



---

### 2. Workflow And Double Verification Logic

This models engineered as a **Clinical Decision Support Tool**. To ensure the highest level of safety, the system follows a strict execution sequence and a custom verification protocol.

#### **A. Sequence of Execution**

The diagnostic architecture of the system follows a strictly hierarchical execution sequence. The predicted output from the Stage Model is a fundamental prerequisite because it serves as a high-weight input feature for the final Status (Mortality) Model.

* Phase 1 (Histological Diagnosis): The inference engine first processes raw clinical biomarkers (such as Bilirubin and Albumin) to determine the liver's histological Stage Prediction using the hepatitis_stage.pkl model.

* Phase 2 (Prognostic Analysis): The system automatically merges the initial clinical data with the AI-generated Stage to calculate the final Mortality Risk Probability using the hepatiti_status.pkl model.

This dependency ensures that the structural state of the liver is prioritized as a primary factor before the system evaluates functional survival outcomes, mirroring real-world clinical decision-making.

---

### 3. Clinical Interpretation

The system categorizes results based on a probability threshold to assist in rapid triage:

* **Stable:** If the Mortality Probability is **< 50%**, the case is classified as stable with a recommendation for routine clinical monitoring.
* **Critical:** If the Mortality Probability is **> 50%**, the case is flagged as critical, requiring immediate medical intervention and further diagnostic confirmation.

---

### 4. Mathematical Validation

To complement the AI's predictive power, the system integrates established clinical scoring systems to provide a "ground truth" reference:

#### **A. APRI Score (AST to Platelet Ratio Index)**

The APRI score is a non-invasive method used to assess the likelihood of liver fibrosis and cirrhosis. It calculates the ratio between liver enzymes and blood platelets to identify structural damage without a biopsy:

$$APRI = \frac{(AST / 40)}{Platelets} \times 100$$

**Python Implementation:**


```python
def calculate_apri(ast_val, platelets):
    """
    Calculate APRI Score based on AST and Platelets.
    Formula: ((AST / 40) / Platelets) * 100
    """
    if platelets == 0: return 0.0  # Prevent division by zero
    return ((ast_val / 40.0) / platelets) * 100

```


* **AST**: Aspartate Aminotransferase level (U/L).
* **40**: The standardized Upper Limit of Normal (ULN) for AST used in this system.
* **Platelets**: Platelet count per cubic millimeter ().

#### **B. ALBI Score (Albumin-Bilirubin Grade)**

The ALBI score is an objective metric specifically designed to evaluate liver functional reserve. Unlike other scores, it relies solely on laboratory markers, eliminating subjective clinical variables:

$$ALBI = (\log_{10}(Bilirubin \times 17.1) \times 0.66) + (Albumin \times 10 \times -0.085)$$

**Python Implementation:**

```python
import math

def calculate_albi(bilirubin, albumin):
    """
    Calculate ALBI Score based on Bilirubin and Albumin.
    Formula: (log10(Bili * 17.1) * 0.66) + (Alb * 10 * -0.085)
    """
    # Ensure bilirubin is non-zero for log calculation
    bili_adj = max(bilirubin, 0.1) 
    
    term1 = math.log10(bili_adj * 17.1) * 0.66
    term2 = albumin * 10 * -0.085
    
    return term1 + term2

```

* **Bilirubin**: Total bilirubin level (converted to  using the  factor).
* **Albumin**: Serum albumin level (converted to  using the  factor).


> By presenting these mathematical validations alongside the AI results, the **Hepatitis Diagnostic Framework** ensures that the "black box" decisions of the Machine Learning models are grounded in proven medical mathematics. This dual-approach increases the system's transparency and reliability for clinical decision support.

---

### Models Input Requirements & Data Structure

To ensure the inference engine functions correctly, data must be structured into precise vector formats specific to each model:

**A. Complications Prediction Model**

* **Target File:** `models/hepatitisC_complications.pkl`
* **Clinical Goal:** Predicts risk of Ascites (Fluid Retention).
* **Input Dimension:** **14 Features** (Excludes 'Ascites' to prevent data leakage).
* **Required Vector:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex', 'Hepatomegaly', 'Spiders', 'Edema']`
> Since the Hepatitis_C 3-models are interconnected and operate in coordination with one another, the 'Stage' is treated as a user request that the system provides in Stage model, assuming the user does not already know it(More info: docs/XGBoost.md).

**B. Stage Prediction Model**

* **Target File:** `models/hepatitisC_stage_model.pkl`
* **Clinical Goal:** Classifies Histological Fibrosis Stage (1, 2, or 3).
* **Input Dimension:** **19 Features** (Includes engineered indices like APRI).
* **Required Vector:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Status', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI', 'Bilirubin_Albumin', 'Copper_Platelets']`

**C. Status Prediction Model**

* **Target File:** `models/hepatitisC_status_model.pkl`
* **Clinical Goal:** Calculates Mortality Risk Probability.
* **Input Dimension:** **18 Features** (Includes ALBI Score).
* **Required Vector:**
`['Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Age', 'Sex', 'Ascites', 'Hepatomegaly', 'Spiders', 'Edema', 'APRI', 'ALBI_Score', 'Bili_Alb_Ratio']`
---

### **Performance & Technical Reference**

For a deeper dive into the model evaluation metrics and architectural logic, please refer to the following resources:


* **Visual Analysis (Confusion Matrices):** To view the performance visualizations and confusion matrices for all models, visit the main analysis notebook: **[notebooks/AI_Liver_Disease_Diagnosis_System.ipynb](https://github.com/yahyazuher/AI-Liver-Disease-Diagnosis-System/blob/main/notebooks/AI_Liver_Diseases_Diagnosis_System.ipynb)**
* **Technical Documentation:** For detailed information on XGBoost hyperparameters, vector logic, and training methodologies, refer to: **[docs/XGBoost.md](./XGBoost.md)**

---

## Virtual Case Analysis Table

| Clinical Case | Brief Description | AI Stage | Survival Risk | Ascites Risk | AI Assessment |
| --- | --- | --- | --- | --- | --- |
| **1. The Healthy Reference** | Normal labs, no symptoms. | **Stage 1** | **12.3%** | **0.3%** |  **STABLE** |
| **2. Acute Inflammation** | High Bilirubin (3.2), Low Platelets. | **Stage 1** | **91.7%** | **2.1%** |  **CRITICAL** |
| **3. Moderate Fibrosis** | Elevated APRI, Stable Albumin. | **Stage 2** | **37.9%** | **0.3%** |  **STABLE** |
| **4. Compensated Cirrhosis** | Fibrosis profile, Normal function. | **Stage 2** | **2.5%** | **0.3%** |  **STABLE** |
| **5. Decompensated Failure** | Ascites + Low Albumin + High Copper. | **Stage 2** | **99.0%** | **99.5%** |  **CRITICAL** |
| **6. Active Injury** | High Enzymes (SGOT), Low Platelets. | **Stage 2** | **83.7%** | **1.0%** |  **CRITICAL** |
| **7. The Geriatric Case** | Age 70, Coagulation issues. | **Stage 2** | **92.0%** | **4.6%** |  **CRITICAL** |

> *The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges. They do not represent medical diagnosis or professional medical judgment.*


### Clinical Insights & Logic Validation

Based on the latest inference outputs, critical technical observations regarding the Multi-Model logic were derived:

#### **A. The "Acute Failure" Paradox (Case 2)**

The system demonstrated sophisticated reasoning in Case 2. It assigned **Stage 1** (indicating early structural damage) but alerted a **91.7% Mortality Risk**. This correctly reflects an "Acute-on-Chronic" scenario where the liver structure is intact, but functional failure (High Bilirubin) poses an immediate threat to life.

#### **B. Differentiation Within Stage 2 (Case 4 vs. Case 5)**

The model grouped both "Compensated" and "Decompensated" patients under **Stage 2**, yet the **Status Model** successfully distinguished their urgency:

* **Case 4 (Compensated):** Identified as low risk (**2.5%**), recognizing that despite fibrosis, the liver is still functioning.
* **Case 5 (Decompensated):** Identified as critical (**99.0%**), driven by the multi-organ failure markers.

#### **C. Precise Complication Detection (Case 5)**

While most cases showed negligible Ascites risk (<5%), **Case 5** triggered a massive **99.5% Ascites Risk**. This confirms the **Complications Model** is acting as an independent "Safety Net," accurately flagging fluid retention even when the Staging model provides a general classification.

#### **D. The Impact of Age (Case 7)**

In **Case 7**, the patient's advanced age (70 years) significantly amplified the mortality risk to **92.0%**, despite having a similar structural profile to Case 3. This confirms the XGBoost architecture weights **Age** as a critical non-linear multiplier for prognosis.

The logic used to generate these results, can be found in: `notebooks/code/test_HC_models` or executed directly via the Colab environment: [![Open In Colab](https://img.shields.io/badge/Open%20In%20Colab-black?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=MjHUnRyuYZMA)


---
*The medical descriptions provided are illustrative summaries derived from publicly available clinical reference ranges and were generated with the assistance of large language models for documentation clarity only. They do not represent medical diagnosis or professional medical judgment.*

