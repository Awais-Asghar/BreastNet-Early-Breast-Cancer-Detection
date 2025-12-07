# BreastNet: Early Breast Cancer Detection
![Project Status](https://img.shields.io/badge/status-Completed-brightgreen.svg)
![Platform](https://img.shields.io/badge/platform-Kaggle-pink.svg)
![Environment](https://img.shields.io/badge/environment-Jupyter%20Notebook-orange.svg)
![Language](https://img.shields.io/badge/language-Python-blue.svg) 
![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)


BreastNet is a neural network based binary classifier designed to detect breast cancer at an early stage. Using nine numerical clinical features from the **Breast Cancer Dataset (Kaggle: yasserh)**, it predicts whether a tumor is **benign** or **malignant**.

<img width="1875" height="1011" alt="Image" src="https://github.com/user-attachments/assets/48c523b1-78e4-44ab-b1a4-f40a87bdaf95" />

# 1. Problem Statement

Breast cancer is a major global health issue and early detection is critical for improving survival rates.
Traditional diagnosis involves manual interpretation of cell measurements, which can be time consuming and sometimes inconsistent.

**Goal:**
Develop a neural network that can automatically classify tumors as benign or malignant based on structured numeric data, offering a fast and reliable clinical decision support tool.

**Why this matters:**

* Early-stage detection significantly improves treatment outcomes.
* Algorithmic assistance can support pathologists and clinicians.
* Machine learning can analyze patterns too subtle for manual inspection.

<img width="1873" height="1012" alt="Image" src="https://github.com/user-attachments/assets/02246fa6-95b8-45c6-9847-ad6903935083" />

<img width="1876" height="1009" alt="Image" src="https://github.com/user-attachments/assets/db90213e-825e-4238-976e-83f75286319c" />

---

# 2. Dataset Description

**Source:** Kaggle (Breast Cancer Dataset by yasserh)
**Link:** [https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

### Dataset Details

* **Total Samples:** 699
* **Total Features:** 10
* **Usable Features for Learning:** 9
* **Target Label Column:** Class

  * 2 → Benign
  * 4 → Malignant

<img width="1874" height="1020" alt="Image" src="https://github.com/user-attachments/assets/4efb64a2-c9f1-4c34-9c73-b30231ccf9a1" />

### Feature List

These features describe visual and morphological characteristics of cell nuclei:

1. Clump Thickness
2. Uniformity of Cell Size
3. Uniformity of Cell Shape
4. Marginal Adhesion
5. Single Epithelial Cell Size
6. Bare Nuclei
7. Bland Chromatin
8. Normal Nucleoli
9. Mitoses

The **Id** column is removed because it does not contain predictive information.

<img width="1872" height="1006" alt="Image" src="https://github.com/user-attachments/assets/7909fe95-788a-4107-a78b-1e9d21f77918" />

### Nature of Features

All nine features are integer scores between 1 and 10.
Higher values generally correlate with higher malignancy risk.

---

# 3. Data Preprocessing Workflow

Your notebook follows a structured preprocessing pipeline:

### Step 1: Load Dataset

Load CSV into pandas and inspect for missing values.

### Step 2: Drop the Id Column

`Id` is irrelevant for prediction.

### Step 3: Handle Missing Values

Commonly found in “Bare Nuclei”.
Rows with missing values are removed or corrected.

### Step 4: Encode the Class Label

Convert (2, 4) → (0, 1):

* 0 → Benign
* 1 → Malignant

### Step 5: Train Test Split

Example: 80 percent training, 20 percent testing.

### Step 6: Feature Scaling

Apply **StandardScaler:**

* Ensures all features are on similar scales
* Improves convergence during training

### Step 7: Model Input Preparation

Convert scaled arrays into formats suitable for the neural network training loop.

<img width="1876" height="1012" alt="Image" src="https://github.com/user-attachments/assets/2da412a4-3d22-4422-adfe-804d9e469586" />

<img width="1870" height="1013" alt="Image" src="https://github.com/user-attachments/assets/19511529-402a-4f96-9c85-26d5d644ecf0" />

---

# 4. Model Architecture (BreastNet)

BreastNet is a simple feed forward neural network built from scratch or using deep learning libraries.

### Architecture Summary

**Input Layer:**

* 9 neurons (one per feature)

**Hidden Layer 1:**

* ~16 or 32 neurons
* Activation: ReLU

**Hidden Layer 2 (optional):**

* ~8 or 16 neurons
* Activation: ReLU

**Output Layer:**

* 1 neuron
* Activation: Sigmoid
* Outputs probability of malignancy

### Why This Architecture?

* Dataset is small, so simple architecture avoids overfitting
* ReLU speeds up training
* Sigmoid enables true binary probability output
* Easy to interpret

### Loss and Optimization

* **Loss:** Binary Cross Entropy
* **Optimizer:** Gradient Descent or Adam-like strategy
* **Epochs:** ~1300 (your notebook trained until convergence)
* **Learning Rate:** around 0.03

---

# 5. Training Behavior and Convergence

Your model shows a clear improvement in testing cost as training progresses:

### Example Logs (taken from your notebook):

* Epoch 0:

  * Train Cost = 0.8021
  * Test Cost = 0.9719

* Epoch 300:

  * Train Cost = 0.1148
  * Test Cost = 0.1515

* Epoch 700:

  * Train Cost = 0.0614
  * Test Cost = 0.1170

* Epoch 990+:

  * Best test costs reach **0.1122**

This demonstrates strong convergence and stable learning.

---

# 6. Final Evaluation Metrics (Extracted From Notebook)

### Confusion Matrix

```
TP: 26  
TN: 83  
FP: 5  
FN: 0  
```

### Direct Results

* **Final Test Cost:** 0.1122
* **F1 Score:** 0.9123

### Derived Performance Metrics

**Accuracy:**
(26 + 83) / (26 + 83 + 5 + 0)
= 109 / 114
= **95.6 percent**

**Precision:**
26 / (26 + 5)
= **83.87 percent**

**Recall (Sensitivity):**
26 / (26 + 0)
= **100 percent**

**Specificity:**
83 / (83 + 5)
= **94.3 percent**

**F1 Score:**
= **91.23 percent**

### Interpretation

* Zero false negatives → the model **never misses malignant cases**
* High recall → clinically important for early cancer detection
* High specificity → good at distinguishing benign cases
* F1 score balances both precision and recall

This is a strong result given the small dataset size.

---

# 7. Key Observations

1. The model achieves excellent recall, critical for cancer screening.
2. Training stabilizes after about 900 epochs, with minimal test cost.
3. Misclassifications mainly occur as false positives, which is safer than false negatives.
4. Neural network captures relationships between features effectively.
5. Scaling features greatly helps training stability.



<img width="998" height="723" alt="Image" src="https://github.com/user-attachments/assets/829fa397-2971-4a08-a9ab-0a547cc679e3" />

<img width="1052" height="680" alt="Image" src="https://github.com/user-attachments/assets/1f788f4d-2094-4a8e-9207-8ea94f640902" />

<img width="1874" height="1012" alt="Image" src="https://github.com/user-attachments/assets/f36385cd-337f-451b-90a3-df5684f95986" />

<img width="1873" height="1012" alt="Image" src="https://github.com/user-attachments/assets/484b7315-f8f2-432f-add2-d9fd28935898" />

<img width="1874" height="1015" alt="Image" src="https://github.com/user-attachments/assets/43fb3147-ba7f-440a-a85d-e2d6da26452d" />

<img width="1869" height="1014" alt="Image" src="https://github.com/user-attachments/assets/a01523d1-2f77-4fad-9757-11ebd870ef6c" />

<img width="1871" height="1012" alt="Image" src="https://github.com/user-attachments/assets/8bcf7a7b-5c3f-4d6c-bd0e-47053847553f" />

---

# 8. Project Conclusion

BreastNet demonstrates the effectiveness of neural networks in early breast cancer detection using structured numeric medical data.

The model:

* Achieves **95.6 percent accuracy**
* Achieves **100 percent sensitivity** to malignant cases
* Maintains strong generalization on test data
* Provides fast and consistent predictions

This project showcases how AI can augment medical decision making in real clinical settings.

<img width="1871" height="1007" alt="Image" src="https://github.com/user-attachments/assets/86ff6c60-a595-4647-80af-4ecee84e8645" />

---

# 9. Limitations

* Dataset is small and may not fully generalize to real-world populations.
* Based solely on tabular data; real diagnosis typically uses imaging (mammograms).
* The model is not medically validated.
* Potential overfitting due to small dataset and long training time.

<img width="1874" height="1013" alt="Image" src="https://github.com/user-attachments/assets/33203808-9c08-4985-a999-6aafd6bc41c0" />

---

# 10. Future Improvements

* Incorporate deeper networks or regularization techniques
* Add ROC AUC calculation and threshold tuning
* Use oversampling or class weights to handle imbalance
* Combine tabular data with image based models
* Deploy as a web-based prediction tool
* Validate on additional breast cancer datasets

<img width="1873" height="1014" alt="Image" src="https://github.com/user-attachments/assets/856acd4b-9db6-4b03-8c3c-023f50b78434" />

---

# 11. Tools and Technologies Used

* Python
* pandas, NumPy
* StandardScaler
* Custom Neural Network
* Matplotlib for cost/metric visualization
* Jupyter Notebook
* Kaggle Breast Cancer Dataset

---
