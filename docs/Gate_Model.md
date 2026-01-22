
# The Gate Model: First Line of Defense

The **Gate Model** is the primary entry point for the diagnostic system. It is a binary classification model (XGBoost) trained to distinguish between a **Healthy Individual** and a **Liver Patient** based on standard blood test markers.

* **Algorithm:** XGBoost Classifier
* **Role:** Screening & Triage
* **Input:** Basic biochemical liver function tests (Age, Bilirubin, Enzymes, Albumin, etc.)
* **Output:** `0` (Sick/Patient) or `1` (Healthy)

---

## ⚡ The "Gateway" Architecture (Efficiency First)
The system is designed with a **resource-efficient workflow**. Instead of running all diagnostic models (Hepatitis, Fatty Liver, Cancer) simultaneously—which consumes battery and processing power—the Gate Model acts as a smart filter.

### How it Works:
1.  **Screening:** The user's data is first processed *only* by the Gate Model.
2.  **Decision Making:**
    * 🟢 **If Healthy:** The workflow terminates immediately. No further analysis is needed. This ensures **zero unnecessary computation**.
    * 🔴 **If Patient:** The system recognizes a risk and *only then* activates the secondary specialized models to diagnose the specific condition.

> **Key Benefit:** This "Conditional Computation" approach ensures that the application remains lightweight and fast, saving device battery and server resources by preventing the execution of complex models on healthy users.

---

## 🔄 Logic Flowchart

```mermaid
graph TD
    Input[User Blood Data] --> Gate{🛡️ Gate Model}
    
    Gate -- Predicted: Healthy --> Stop((✅ Stop Process))
    Stop -.-> Msg[User is Healthy - Save Battery/Resources]
    
    Gate -- Predicted: Patient --> Trigger[⚠️ Activate Sub-Models]
    
    subgraph "Advanced Analysis Layer"
    Trigger --> M1[Hepatitis Model]
    Trigger --> M2[Fatty Liver Model]
    Trigger --> M3[Cancer Risk Model]
    end
