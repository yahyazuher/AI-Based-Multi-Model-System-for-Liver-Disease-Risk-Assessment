# Liver Cancer Risk Assessment Model

This section is dedicated to evaluating the probability of developing Hepatocellular Carcinoma (HCC) by analyzing the complex interplay between genetic predisposition and environmental triggers. The system relies on an **XGBoost** training model, with its core file built as `cancer_model.pkl`. The model analyzes input values based on "weights" acquired during the training phase, allowing for a precise determination of the risk level associated with each analytical factor.

---

### Dataset Overview

| Name | Database Location | Function |
| --- | --- | --- |
| **cancer_model.pkl** | `models/` | The trained model containing the final decision-making weights. |
| **train_cancer_model.py** | `code/` | Source code responsible for building and training the model. |
| **test_cancer_model.py** | `code/` | Source code dedicated to testing the efficiency of the trained model. |
| **The_Cancer_data_1500.csv** | `data/raw` | Training dataset containing 1,500 patient records with required analytics. |
| **XGBoost.md** | `docs/` | Technical documentation explaining the mechanism of the XGBoost algorithm. |

---

### Training Phase

The system's efficiency depends on a data split of **80% for training** and **20% for testing**.

* **Training Data:** The model was trained on data from **1,200 patients** from the `The_Cancer_data_1500.csv` file.
* **Testing Data:** Data from **300 patients** was reserved to test the accuracy and validity of the model on unseen data.

> **Technical Note:** This split is considered the "Golden Standard" for building a "Smart System." It prevents the model from "hallucinating" or suffering from Overfitting, which can occur if trained on 100% of the data, especially in datasets with limited size (for more information: `docs/XGBoost.md`).

---

### Data Source and Integrity

* **Original Database:** Obtained from the "Cancer Prediction Dataset" on **Kaggle** (by researcher *Rabie El Kharoua*).
* **Data Link:** [Source on Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset?resource=download)
* **Data Integrity:** No manual modifications or additional cleaning were performed. The file was used in its original state as it is technically optimized directly from the source for handling Machine Learning algorithms.

---

### 2- Model Input Requirements

To ensure result accuracy, data must be entered in the strict mathematical order used during model training:
`['Age', 'Gender', 'BMI', 'Smoking', 'GeneticRisk', 'PhysicalActivity', 'AlcoholIntake', 'CancerHistory']`

---

### 3- Virtual Clinic Test Results

To demonstrate the practical application of the model, a "Virtual Clinic" simulation was conducted. This phase serves as a real-world test of the Model Logic, moving beyond raw training metrics. **7 clinical scenarios** reflecting societal realities were designed. The results demonstrated the model's accuracy in comprehending the interaction between behavior, age, and genetics.

### Virtual Case Analysis Table

| Clinical Case | Brief Description | Result | Risk % | Clinical Interpretation |
| --- | --- | --- | --- | --- |
| **1. The Athletic Youth** | 25 years old, high physical activity, no triggers. | **🟢 Healthy** | **1.3%** | Reference case for biosafety. |
| **2. The Heavy Smoker** | 55 years old, continuous smoking and alcohol use. | **🔴 High Risk** | **99.8%** | Cumulative effect of environmental toxins. |
| **3. High Genetics / Healthy Life** | High Genetic Risk, but ideal lifestyle. | **🟢 Healthy** | **17.4%** | **Key Result:** Lifestyle neutralized genetic risk. |
| **4. The Catastrophic Case** | Morbid obesity, alcohol, smoking, genetics, old age. | **🔴 High Risk** | **99.9%** | Confluence of all carcinogenic factors. |
| **5. Obesity Only** | BMI 40, no smoking or alcohol. | **🟢 Healthy** | **12.6%** | Obesity is a risk factor but less severe than smoking. |
| **6. The Healthy Elderly** | 80 years old, athletic, non-smoker. | **🟢 Healthy** | **0.8%** | Age is not a direct cause of cancer without triggers. |
| **7. The Borderline Case** | Moderate risks across most factors. | **🟢 Healthy** | **2.7%** | Model's ability to distinguish stable cases. |

---

### Clinical Insights

Based on the Virtual Clinic outputs, three critical technical observations regarding the model can be derived:

#### **A. Resilience of the "Healthy Elderly" (Case 6):**

The model assigned the lowest risk percentage (**0.8%**) to an 80-year-old individual. This proves that the model does not treat "Age" as a death sentence; instead, it assigned higher weights to **Physical Activity** and **Non-Smoking**.

#### **B. Decoupling Genetics from Disease (Case 3):**

When a patient with high genetics (GeneticRisk = 2) but a healthy lifestyle was introduced, the result was **17.4%** (50% below the risk threshold). This confirms that the `cancer_model.pkl` file has learned that genetics represent a "predisposition," not an "inevitable fate."

#### **C. The Lethal Weight of Toxins (Case 2 & 4):**

Once smoking and alcohol entered the equation, the risk percentage immediately spiked above **99%**. This reflects the **High Weights** assigned by the **XGBoost** system to these specific factors during its training on 1,200 patients.

---

### Technical Note for Developers (Testing Logic)

These tests were conducted using the `test_cancer_model.py` file.

> The model is not merely a numerical classifier; it is a system capable of offering "preventive advice" by demonstrating how altering a single factor (such as quitting smoking) can drop the risk percentage from 99% to less than 20%.

---

### Preventive Screening Tool

This model provides AI-based diagnostic specialists with an innovative method to support medical decision-making, contributing to the reduction of human error. For the patient, the model offers a clear vision of how effective early intervention and lifestyle modification—such as quitting smoking and alcohol—can be in neutralizing non-. The system transforms complex digital data into a tangible preventive message that contributes to saving lives, by the will of Allah.
