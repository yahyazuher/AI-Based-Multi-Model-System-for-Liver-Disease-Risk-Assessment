# Medical Disclaimer & System Limitations

## 1. General Disclaimer
**This software is for research and educational purposes only.**
The content, models, and predictions provided by this repository are not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or calculated using this system.

## 2. Limitations of "Ground Truth" (Labeling Methodology)

* [cite_start]**Method:** The target labels (Healthy vs. Sick) were inferred based on biochemical thresholds derived from clinical guidelines.

## 3. Data Imputation & The "Veto" Safety Net
In scenarios where user input is incomplete, the system may impute "Normal" values to facilitate processing, or repace it with value zero 0 , or output error.
This imputation can theoretically lead to **False Negatives**.
* [cite_start]**Mitigation:** A "Veto System" has been implemented to override favorable donor predictions if the separate **Fibrosis Model** detects high-risk stages (Stage 2-3-4) using independent markers (Platelets/Prothrombin). However, no safety system is fail-safe.

## 4. Applicability
This model is calibrated using specific demographic data and may not generalize perfectly to populations with significantly different genetic or environmental profiles.
