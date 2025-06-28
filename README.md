# COMSYS-HACKATHON2025-FACECOM

This repository contains little discussion for both tasks from the COMSYS Hackathon - 5:

* **Task A :---** Gender Classification (Binary Classification)
* **Task B :---** Face Recognition (Multi-class Classification)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🔹 Task A :--- Gender Classification

### 📝 Description

The objective of Task A is to build a gender classification model that identifies whether a given face image is of a **male** or **female**.

### 📁 Dataset Structure

```
Comys_Hackathon5
├──
Task_A/
├── train/
│   ├── male/
│   └── female/
├── val/
│   ├── male/
│   └── female/
```

### 🧠 Model

* Base model: `EfficientNetB0`
* Transfer Learning with fine-tuning
* Final layer: Dense(1, activation='sigmoid')

### ⚙️ How to Train

```python
model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
```

### ⚙️ Fine Tuning

```python
for layer in base_model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    epochs=5,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)
```

### 💾 Model Weights

Due to GitHub size limits, model weights (`.h5` files) are stored in Google Drive and linked in the submission.

### 📊 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score

### ✅ Evaluation Metrices Results

Evaluation Results on Validation Set :---
|   Metric   |  Score   |
| ---------- | -------- |
| Accuracy   | \~0.8626 |
| Precision  | \~0.9246 |
|   Recall   | \~0.8896 |
|  F1-Score  | \~0.9068 |


-------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 🔹 Task B: Face Recognition

### 📝 Description

Task B aims to recognize distorted face images by matching them to clean reference identities. The TASK_B repo contains code using a **Siamese Neural Network** with contrastive learning.

### 📁 Dataset Structure

```
Comys_Hackathon5
├──
Task_B/
├── train/
│   ├── <identity>/
│       ├── <identity>.jpg
│       └── distortion/
├── val/
    ├── <identity>/
        ├── <identity>.jpg   
        └── distorted/
```

### 🧠 Model

* Siamese Network using Conv layers
* Embedding size : 512
* Distance metric : Euclidean distance
* Optimizer : Adam, Loss: Binary Crossentropy

### ⚙️ Training

* Training pairs : 10,000
* Epochs : 30 (can increase if needed)
* Batch size : 16

### 📊 Evaluation Metrics

* Top-1 Accuracy
* Macro-averaged F1-Score

### ✅ Evaluation Metrices Results

| Metric   | Score    |
| -------- | -------- |
| Accuracy | \~0.7162 |
| F1-Score | \~0.5856 |

### 💾 Model Weights

Due to GitHub size limits, model weights (`.h5` files) are stored in Google Drive and linked in the submission.


-------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 📁 Directory Layout

```
COMSYS-HACKATHON-FACECOM/
├── README.md (this file)
├── Task_A/
├── Task_B/
├── gender_classification_best_model_h5 (TASK_A)
├── embedding_model_h5 (TASK_B)
├── face_matching_h5 model (TASK_B)
├── 1 page technical summary
```


-------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 📌 Submission Checklist

* [x] Code for Task A & B
* [x] Pretrained weights uploaded to Google Drive
* [x] 1 page Technical summary
* [x] README with full documentation


-------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Coders

* \Saikat Munshib(Leader)
* \Team :--- Alpha Algorithm
* \Government College of Engineering and Textile Technology, Serampore
* \saikatmunshib@gmail.com

---

Thank you for reviewing this submission for the **COMSYS Hackathon 5**!
