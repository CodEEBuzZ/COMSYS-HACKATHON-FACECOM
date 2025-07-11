#Mount google drive to access the whole dataset
from google.colab import drive
drive.mount('/content/drive')

#All necessary libraries
import os
import numpy as np
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

#Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
TRAIN_DIR = '/content/drive/MyDrive/Colab Notebooks/Comys_Hackathon5/Task_A/train'
VAL_DIR = '/content/drive/MyDrive/Colab Notebooks/Comys_Hackathon5/Task_A/val'
MODEL_PATH = '/content/drive/My Drive/Colab Notebooks/Comys_Hackathon5/Task_A/gender_classification_best_model.h5'

#Training Images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.6, 1.4],
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

#Train set load and validation data
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

#Making 0 = female and 1 = male
class_weights = {0: 2.0, 1: 1.0}  # Assuming 0 = female, 1 = male
print("Adjusted Class Weights:", class_weights)

#Model building done by using EfficientNetB0
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)


for layer in base_model.layers:
    layer.trainable = False
#Compile model
model.compile(optimizer=Adam(1e-3), loss='binary_crossentropy', metrics=['accuracy'])

#Call backs from epochs
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.3, min_lr=1e-6)
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)

#Training 1
model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr, checkpoint]
)

#Fine tunning training 2
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

#Load best model
model = load_model(MODEL_PATH)

#Evaluation validation set
preds_prob = model.predict(val_generator)
preds = (preds_prob > 0.5).astype(int).flatten()
y_true = val_generator.classes

acc = accuracy_score(y_true, preds)
prec = precision_score(y_true, preds)
rec = recall_score(y_true, preds)
f1 = f1_score(y_true, preds)

print("\n.......... Evaluation Results on Validation Set...........")
print(f"--> Accuracy:  {acc:.4f}")
print(f"--> Precision: {prec:.4f}")
print(f"--> Recall:    {rec:.4f}")
print(f"--> F1 Score:  {f1:.4f}")
print("\n--> Classification Report:\n", classification_report(y_true, preds, target_names=val_generator.class_indices.keys()))

# Evaluate on training set
train_generator_for_eval = val_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

preds_train_prob = model.predict(train_generator_for_eval)
preds_train = (preds_train_prob > 0.5).astype(int).flatten()
y_train_true = train_generator_for_eval.classes

# Compute training metrics
acc_train = accuracy_score(y_train_true, preds_train)
prec_train = precision_score(y_train_true, preds_train)
rec_train = recall_score(y_train_true, preds_train)
f1_train = f1_score(y_train_true, preds_train)

print("\n.......... Evaluation Results on Training Set ...........")
print(f"--> Accuracy:  {acc_train:.4f}")
print(f"--> Precision: {prec_train:.4f}")
print(f"--> Recall:    {rec_train:.4f}")
print(f"--> F1 Score:  {f1_train:.4f}")
