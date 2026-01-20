Liver Cancer Risk Assessment Model
نموذج تقييم مخاطر سرطان الكبد
This module evaluates the probability of developing Hepatocellular Carcinoma by analyzing the interplay between genetic predisposition and environmental triggers.

1. Dataset Overview

Source: The_Cancer_data_1500.csv.

Original Dataset: This data was obtained from the "Cancer Prediction Dataset" on Kaggle (by Rabie El Kharoua).


URL: https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset?resource=download


Size: 1,500 patient records.


Data Integrity Note: No manual modifications or data cleaning were performed on this specific file. The dataset was used in its original state as it was already optimized for Machine Learning training directly from the source.

2. Model Input Requirements (Column Order)
For the trained model (cancer_model.pkl) to function correctly and provide accurate predictions, the input data must follow the exact mathematical order used during training.

3. Virtual Clinic Test Results
The model was validated using 7 diverse clinical scenarios to test its sensitivity to different risk factors:

4. Scientific Interpretation
4.1 The Epigenetic Effect (Lifestyle vs. Genetics)
The model discovered a crucial medical fact: "Lifestyle can modulate genetic expression".


Finding: A patient with high genetic risk but a healthy lifestyle (no smoking, no alcohol, regular exercise) remained below the 50% risk threshold (17.4%).


Conclusion: While genetics increase the baseline risk (up to 17 times higher), they do not trigger cancer without environmental catalysts.

4.2 Feature Importance (Weight Analysis)
Technically, the model assigns significantly higher weights to behavioral factors than to hereditary factors.


Primary Triggers: Smoking and Alcohol Intake are identified as the most critical predictors of high risk.


Observation: In Case 2, the presence of smoking and alcohol consumption spiked the risk to 99.8%, demonstrating the model's focus on preventable risk factors.

5. Clinical Significance
This model serves as a preventive screening tool, illustrating how early intervention in lifestyle habits can effectively counteract non-modifiable genetic risks.
