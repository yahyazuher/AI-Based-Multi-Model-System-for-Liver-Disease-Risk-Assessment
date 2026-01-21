# Liver Cancer Risk Assessment Model

This section is dedicated to evaluating the probability of developing Hepatocellular Carcinoma (HCC) by analyzing the complex interplay between genetic predisposition and environmental triggers. The system relies on an **XGBoost** training model, with its core file built as `cancer_model.pkl`. The model analyzes input values based on "weights" acquired during the training phase, allowing for a precise determination of the risk level associated with each analytical factor.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

---

### Dataset Overview

| Name | Database Location | Function |
| --- | --- | --- |
| **cancer_model.pkl** | `models/` | The trained model containing the final decision-making weights. |
| **train_cancer_model.py** | `code/` | Source code responsible for building and training the model. |
| **test_cancer_model.py** | `code/` | Source code dedicated to testing the efficiency of the trained model. |
| **The_Cancer_data_1500.csv** | `data/processed` | Training dataset containing 1,500 patient records with required analytics. |
| **XGBoost.md** | `docs/` | Technical documentation explaining the mechanism of the XGBoost algorithm. |

---

### Training Phase

The system's efficiency depends on a data split of **80% for training** and **20% for testing**.

* **Training Data:** The model was trained on data from **1,200 patients** from the `The_Cancer_data_1500.csv` file.
* **Testing Data:** Data from **300 patients** was reserved to test the accuracy and validity of the model on unseen data.

> **Technical Note:** This split is considered the "Golden Standard" for building a "Smart System." It prevents the model from "hallucinating" or suffering from Overfitting, which can occur if trained on 100% of the data, especially in datasets with limited size (for more information: `docs/XGBoost.md`).

---

### 1- Data Source and Integrity

* **Original Database:** Obtained from the "Cancer Prediction Dataset" on **Kaggle** (by researcher *Rabie El Kharoua*).
* **Data Link:** [Source on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset?resource=download)
* **Data Integrity:** No manual modifications or additional cleaning were performed. The file was used in its original state as it is technically optimized directly from the source for handling Machine Learning algorithms.

---

### 2- Model Input Requirements

To ensure result accuracy, data must be entered in the strict mathematical order used during model training:
`['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']`

---

### 3- Model Optimization & Refinement

The model underwent a critical optimization phase where the `max_depth` parameter was adjusted from **5** to **3**. This strategic reduction in complexity, combined with optimized learning rates, led to a measurable improvement in diagnostic performance:

* **Accuracy Boost:** The overall predictive accuracy increased from **92%** to **94%**.
* **Generalization vs. Overfitting:** By limiting the tree depth, the model stopped "memorizing" noise in the training data (Overfitting) and instead focused on the most significant clinical patterns.
* **Precision Improvement:** The precision for detecting high-risk cases (Class 1) rose from **93%** to **95%**, significantly reducing false positives.
* **Recall Optimization:** The model's ability to correctly identify actual cancer risks (Recall) improved from **87%** to **90%**.

[NOTE] Technical Implementation: The Automated Hyperparameter Tuning Strategy used to derive these optimal values is implemented in the Google Colab notebook under the cell titled "Cancer Risk Model". For better clarity, a comprehensive explanation of the tuning logic is provided directly above the cell, followed by a step-by-step documentation of the code's functionality. [![Open In Colab](https://img.shields.io/badge/Open%20In%20Colab-black?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/drive/1sr0GzN9SEN2H5wC3t0REaPVXUMlFYzfG#scrollTo=OGcBn26-pcsQ)

---

### Performance Comparison Table

| Metric | Initial Model (`depth: 5`) | Optimized Model (`depth: 3`) | Improvement |
| --- | --- | --- | --- |
| **Accuracy** | 92% | **94%** | **+2%** |
| **F1-Score (Risk)** | 0.90 | **0.92** | **+0.02** |
| **Precision (Risk)** | 0.93 | **0.95** | **+0.02** |
| **Recall (Risk)** | 0.87 | **0.90** | **+0.03** |

---

### Optimized Model Configuration

The following parameters were identified as the **"Gold Standard"** for this specific dataset (1,500 records) to achieve maximum stability and performance:

```python
# The optimized XGBoost configuration
model = xgb.XGBClassifier(
    n_estimators=100,      # Balanced number of trees to prevent complexity
    learning_rate=0.1,     # Optimal step size for stable convergence
    max_depth=3,           # Strategic depth to ensure high generalization
    subsample=0.8,         # Trains on 80% of data per tree to boost robustness
    eval_metric='logloss'  # Standard evaluation metric for binary classification
)

```

> **Scientific Insight:** For a dataset of 1,500 records, a `max_depth` of **3** provides the optimal balance between "learning" and "generalizing." This prevents the model from becoming too rigid, ensuring the system remains robust and reliable when encountering new patients in a real-world clinical setting.

---
### 4. Virtual Clinic Test Results

To demonstrate the practical application of the model, a **"Virtual Clinic"** simulation was conducted. This phase serves as a real-world validation of the Model Logic, moving beyond raw training metrics. **7 clinical scenarios** reflecting diverse societal realities were designed to test the model's accuracy in comprehending the complex interaction between behavior, age, and genetics.

### Virtual Case Analysis Table

| Clinical Case | Brief Description | Result | Risk % | Clinical Interpretation |
| --- | --- | --- | --- | --- |
| **1. The Athletic Youth** | 25y, high physical activity, no triggers. | **🟢 Healthy** | **0.61%** | Baseline reference for a low-risk profile. |
| **2. The Heavy Smoker** | 55y, chronic smoking/alcohol use. | **🔴 High Risk** | **99.82%** | Clear impact of continuous environmental toxins. |
| **3. Genetics vs. Lifestyle** | High Genetic Risk, but ideal lifestyle. | **🟢 Healthy** | **45.80%** | **Key Result:** Lifestyle effectively balanced high genetic risk. |
| **4. The Catastrophic Case** | Multiple risks, obesity, and old age. | **🔴 High Risk** | **99.98%** | Confluence of all major carcinogenic factors. |
| **5. Obesity Only** | BMI 40, non-smoker, active. | **🟢 Healthy** | **12.45%** | Obesity is a risk but secondary to lifestyle habits. |
| **6. The Healthy Elderly** | 80y, athletic, non-smoker. | **🟢 Healthy** | **0.94%** | Age is not a standalone trigger without toxins. |
| **7. The Borderline Case** | Moderate risks across most factors. | **🟢 Healthy** | **5.42%** | Demonstrates model stability in moderate scenarios. |

---

### Clinical Insights

Based on the Virtual Clinic outputs, three critical technical observations regarding the model can be derived:

#### **A. Resilience of the "Healthy Elderly" (Case 6)**

The model assigned an exceptionally low risk percentage (**0.94%**) to an 80-year-old individual. This proves that the system does not treat "Age" as a primary cause of disease; instead, it assigns much higher weights to **Physical Activity** and **Non-Smoking**, **rewarding long-term healthy habits** .

#### **B. Decoupling Genetics from Disease (Case 3)**

When a patient with high genetic predisposition (`GeneticRisk = 2`) and family history but an ideal lifestyle was introduced, the model returned **45.80%**. While this is higher than the baseline, it remains below the **50% diagnostic threshold**. This confirms the `cancer_model.pkl` file has learned that genetics represent a "vulnerability" rather than an "inevitable fate," provided preventive measures are strictly followed.

#### **C. The Lethal Weight of Toxins (Case 2 & 4)**

The data confirms that environmental triggers are the primary drivers of risk. Once heavy smoking and alcohol consumption are introduced, the risk probability immediately spikes above **99%**. This reflects the **high feature importance** assigned by the **XGBoost** algorithm to toxins over all other variables(more info: docs/XGBoost).

---
# 5- Technical Note for Developers 

Execution: These tests were conducted using the code/test_cancer_model.py file. 

Source Integrity: The source file is used exactly as provided by the original source. No modifications or changes have been made to the internal content of the file.

Original Data & Code: 
* The original training dataset is preserved and available at: data/raw/The_Cancer_data_1500_V2.csv.
* The original model training code is available at: code/train_cancer_model.py.

> The model is not merely a numerical classifier; it is a system capable of offering "preventive advice" by demonstrating how altering a single factor (such as quitting smoking) can drop the risk percentage from 99% to less than 20%.

---

### Preventive Screening Tool

This model provides AI-based diagnostic specialists with an innovative method to support medical decision-making, contributing to the reduction of human error. For the patient, the model offers a clear vision of how effective early intervention and lifestyle modification—such as quitting smoking and alcohol—can be in neutralizing non-. The system transforms complex digital data into a tangible preventive message that contributes to saving lives, by the will of Allah.
