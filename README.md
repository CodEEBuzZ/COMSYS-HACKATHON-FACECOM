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

### 📝 Steps for running the code

***NEED TO DOWNLOAD***

```bash
   !pip install torch torchvision torchaudio
   !pip install scikit-learn
   !pip install matplotlib opencv-python
   ```

***STEPS***

* **Step 1 :-** For Colab, then mount need to be done, code present in TASK_A.py. If this program runs in jupyter notebook or any other platform then mount no need to do with drive. Just change the path according to the path of the system where the datas will be present either in C or D or any other place or any drive.
* **Step 2 :-** Import all the libraries which are present.
*  **Step 3 :-** Then paste configuration, now changing those path need to be done.
*  **Step 4 :-** After that all the code have to be run as it is and it will give result as well as it will train the the dats along with fine tunning.
*  **Step 5 :-** At last loading need to be done so for that model path need to be give correctly that where the model is save, then it will give all the metrices. OR. The model is provided by me it can also be used then from that also model will provide correct result.
*  **Step 6 :-** This section is optional or it can be also done that this section is for the hidden test checking, as 'Final Evaluation for Hidden Test'.

### 📊 Evaluation Metrics

The whole code is present in the TASK_A.py file, their all the paths which are present need to change according to the system paths where the program will be run and then after that the program need to be run after that the whole output will be generate.

* Accuracy
* Precision
* Recall
* F1-Score

### ✅ Evaluation Metrices Results

Evaluation Results on Validation Set :---
|   Metric   |  Score |
| ---------- | ------ |
| Accuracy   | 0.8626 |
| Precision  | 0.9246 |
|   Recall   | 0.8896 |
|  F1-Score  | 0.9068 |

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

### ⚙️ How to Train

```python
if os.path.exists(MODEL_SAVE_PATH):
    print("\n--> Loading existing model to continue training...")
    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'euclidean_distance': euclidean_distance})
    embedding_model = tf.keras.models.load_model(EMBEDDING_SAVE_PATH)
else:
    print("\n--> Building new model...")
    model, embedding_model = build_siamese_network(INPUT_SHAPE)
    model.compile(loss='binary_crossentropy', optimizer=Adam(1e-4), metrics=['accuracy'])

print("\n--> Preparing generator and starting training chunk...")
generator = PairGenerator(IDENTITY_DIR, batch_size=BATCH_SIZE, steps_per_epoch=NUM_PAIRS // BATCH_SIZE)
model.fit(generator, epochs=EPOCHS)

print("--> Saving models after training chunk...")
model.save(MODEL_SAVE_PATH)
embedding_model.save(EMBEDDING_SAVE_PATH)


embedding_model = tf.keras.models.load_model(EMBEDDING_SAVE_PATH)
```

### ⚙️ Metrices Fixing

```python
import os
import shutil
import random

# Paths
train_dir = "/content/drive/MyDrive/Colab Notebooks/Comys_Hackathon5/Task_B/train"
val_dir = "/content/drive/MyDrive/Colab Notebooks/Comys_Hackathon5/Task_B/val"

# Ensure val dir exists
os.makedirs(val_dir, exist_ok=True)

# Select random 10 identities
identities = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
selected_ids = random.sample(identities, 450)

for identity in selected_ids:
    src_id_folder = os.path.join(train_dir, identity)
    distortion_folder = os.path.join(src_id_folder, "distortion")

    if not os.path.exists(distortion_folder):
        continue

    distorted_images = [f for f in os.listdir(distortion_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Pick 2 distorted images randomly
    selected_imgs = random.sample(distorted_images, min(3, len(distorted_images)))

    # Make destination folder in val
    dst_id_folder = os.path.join(val_dir, identity)
    os.makedirs(dst_id_folder, exist_ok=True)

    # Copy images
    for img_name in selected_imgs:
        src_path = os.path.join(distortion_folder, img_name)
        dst_path = os.path.join(dst_id_folder, img_name)
        shutil.copy2(src_path, dst_path)

print(f"Copied distorted images for {len(selected_ids)} identities into val set.")
```

### ⚙️ Training

* Training pairs : 10,000
* Epochs : 30 (can increase if needed)
* Batch size : 16

### 💾 Model Weights

Due to GitHub size limits, model weights (`.h5` files) are stored in Google Drive and linked in the submission.

### 📝 Steps for running the code

***NEED TO DOWNLOAD***
* Install Microsoft Visual C++ Redistributable
* Install Miniconda
* Create a conda environment
```bash
   conda create --name tf python=3.9
   conda deactivate
   conda activate tf
   ```
* GPU setup
```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```
* Install TensorFlow
```bash
   pip install --upgrade pip
   pip install "tensorflow<2.11"
   ```
* Verify the installation
```bash
   python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
   ```
* In Jupyter Notebook
```bash
   !pip install scikit-learn tqdm opencv-python matplotlib pandas
   !pip install pillow
   ```
  
***STEPS***
* **Step 1 :-** This program can be runs in jupyter notebook or any other platform then mount no need to do with drive. Just change the path according to the path of the system where the datas will be present either in C or D or any other place or any drive.
* **Step 2 :-** Import all the libraries which are present.
*  **Step 3 :-** Then paste configuration, image size and everything, then locations or file path need to be changed according to the system where the program will run.
*  **Step 4 :-** After that all the code have to be run as it is and it will give result as well as it will train the the dats along with fine tunning.
*  **Step 5 :-** Before metrices running one have to do that due to some problems in test and validation data in TASK_B so for that some distortion image have to transfer from test to val data after that metrices giving best accuracy or result. If this problem not occurs in any dataset then no need to do or run that part, if problem happens or present then it have to do. Because if the dataset becomes wrong then answer will come as 0. 
*  **Step 6 :-** At last loading need to be done so for that model path need to be give correctly that where the model is save the model is embidding model, then it will give all the metrices. OR. The embedding model is provided by me it can also be used then from that also model will provide correct result.
*  **Step 7 :-** That matrices section can be done in colab also as it will give better result in colab also. Just ignore the training portion 'Training Data' no need to re run and then run the whole remaining code with accurate path and metrices code then it will give best result.
*  **Step 8 :-** Model file already given in .h5 format it can also be used for the running the whole code and also for the metrices checking also.

### 📊 Evaluation Metrics

The whole code is present in the TASK_B.py file, their all the paths which are present need to change according to the system paths where the program will be run and then after that the program need to be run after that the whole output will be generate.

* Top-1 Accuracy
* Macro-averaged F1-Score

### ✅ Evaluation Metrices Results

| Metric   | Score  |
| -------- | ------ |
| Accuracy | 0.6164 |
| F1-Score | 0.5672 |

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

* Saikat Munshib(Leader)
* Team :--- Alpha Algorithm
* Government College of Engineering and Textile Technology, Serampore
* saikatmunshib@gmail.com

---

Thank you for reviewing this submission for the **COMSYS Hackathon 5**!
