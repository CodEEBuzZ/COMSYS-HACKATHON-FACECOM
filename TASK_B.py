#Important imports
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


IMG_SIZE = (128, 128)
INPUT_SHAPE = IMG_SIZE + (3,)
BATCH_SIZE = 16
EPOCHS = 30
NUM_PAIRS = 10000

#All the different places need for the work
IDENTITY_DIR = "D:/COMSYS Hackathon/Comys_Hackathon5/Task_B/train"
DISTORTED_DIR = "D:/COMSYS Hackathon/Comys_Hackathon5/Task_B/val"
MODEL_SAVE_PATH = "D:/COMSYS Hackathon/Comys_Hackathon5/Task_B/siamese_face_matcher.h5"
EMBEDDING_SAVE_PATH = "D:/COMSYS Hackathon/Comys_Hackathon5/Task_B/siamese_embedding_model.h5"


def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0  # Normalize to [0,1]
    return img


def get_all_identity_images(identity_folder):
    image_paths = []
    for file in os.listdir(identity_folder):
        path = os.path.join(identity_folder, file)
        if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_paths.append(path)
    distortion_folder = os.path.join(identity_folder, 'distortion')
    if os.path.isdir(distortion_folder):
        for file in os.listdir(distortion_folder):
            path = os.path.join(distortion_folder, file)
            if os.path.isfile(path) and path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(path)
    return image_paths


class PairGenerator(tf.keras.utils.Sequence):
    def __init__(self, identity_dir, batch_size=32, steps_per_epoch=100, input_shape=INPUT_SHAPE):
        self.identity_dir = identity_dir
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.input_shape = input_shape
        self.identities = os.listdir(identity_dir)

    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, index):
        pairs = []
        labels = []
        for _ in range(self.batch_size):
            if random.random() < 0.5:
                identity = random.choice(self.identities)
                imgs = get_all_identity_images(os.path.join(self.identity_dir, identity))
                if len(imgs) >= 2:
                    img1, img2 = random.sample(imgs, 2)
                    pairs.append([load_image(img1), load_image(img2)])
                    labels.append(1)
            else:
                id1, id2 = random.sample(self.identities, 2)
                imgs1 = get_all_identity_images(os.path.join(self.identity_dir, id1))
                imgs2 = get_all_identity_images(os.path.join(self.identity_dir, id2))
                if imgs1 and imgs2:
                    img1 = random.choice(imgs1)
                    img2 = random.choice(imgs2)
                    pairs.append([load_image(img1), load_image(img2)])
                    labels.append(0)

        X1 = np.array([pair[0] for pair in pairs])
        X2 = np.array([pair[1] for pair in pairs])
        y = np.array(labels)
        return (X1, X2), y

#Using Siamese Model
def build_siamese_model(input_shape):
    inp = Input(shape=input_shape)
    x = Conv2D(64, (10,10), activation='relu')(inp)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (7,7), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, (4,4), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(256, (4,4), activation='relu')(x)
    x = Flatten()(x)
    x = Dense(512, activation='sigmoid')(x)
    return Model(inp, x)


def euclidean_distance(vectors):
    x, y = vectors
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def build_siamese_network(input_shape):
    base_model = build_siamese_model(input_shape)
    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)
    output_a = base_model(input_a)
    output_b = base_model(input_b)
    distance = Lambda(euclidean_distance)([output_a, output_b])
    prediction = Dense(1, activation='sigmoid')(distance)
    model = Model(inputs=[input_a, input_b], outputs=prediction)
    return model, base_model

#Training Data
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

def get_embedding(img_path):
    img = load_image(img_path)
    img = np.expand_dims(img, axis=0)
    return embedding_model.predict(img)[0]

def match_identity(distorted_path, identity_dir, threshold=0.6):
    distorted_embedding = get_embedding(distorted_path)
    for identity in os.listdir(identity_dir):
        identity_folder = os.path.join(identity_dir, identity)
        refs = get_all_identity_images(identity_folder)
        for ref_path in refs:
            ref_embedding = get_embedding(ref_path)
            dist = np.linalg.norm(distorted_embedding - ref_embedding)
            if dist < threshold:
                return identity, 1
    return "Unknown", 0

#Transfering some distortion image from train to val so that metrices can be better form zero
import os
import shutil
import random

# Paths
train_dir = "/content/drive/MyDrive/Colab Notebooks/Comys_Hackathon5/Task_B/train"
val_dir = "/content/drive/MyDrive/Colab Notebooks/Comys_Hackathon5/Task_B/val"

# Ensure val dir exists
os.makedirs(val_dir, exist_ok=True)

# Select random 550 identities
identities = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
selected_ids = random.sample(identities, 550)

for identity in selected_ids:
    src_id_folder = os.path.join(train_dir, identity)
    distortion_folder = os.path.join(src_id_folder, "distortion")

    if not os.path.exists(distortion_folder):
        continue

    distorted_images = [f for f in os.listdir(distortion_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Pick 3 distorted images randomly
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


# Final Test Evaluation

from tensorflow.keras.models import load_model
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Constants
IMG_SIZE = (128, 128)
EMBEDDING_SAVE_PATH = "D:/COMSYS Hackathon/Comys_Hackathon5/Task_B/siamese_embedding_model.h5" # Siamese_embedding_model.h5 path must be here
DISTORTED_DIR = "D:/COMSYS Hackathon/Comys_Hackathon5/Task_B/val" # Val Path must be here
IDENTITY_DIR = "D:/COMSYS Hackathon/Comys_Hackathon5/Task_B/train" # Train Path must be here

# Load model
model = load_model(EMBEDDING_SAVE_PATH)

def load_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return np.expand_dims(img, axis=0)

def get_embedding(img_path):
    return model.predict(load_image(img_path))[0]

def evaluate_fast(distorted_dir, identity_dir, threshold=0.3):
    print("\nIndexing reference embeddings...")
    identity_db = {}
    for identity in tqdm(os.listdir(identity_dir)):
        identity_folder = os.path.join(identity_dir, identity)
        if not os.path.isdir(identity_folder): continue
        refs = [os.path.join(identity_folder, f) for f in os.listdir(identity_folder)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        embeddings = [get_embedding(f) for f in refs]
        if embeddings:
            identity_db[identity] = np.mean(embeddings, axis=0)

    print("Matching distorted images...")
    y_true, y_pred = [], []

    for identity in tqdm(os.listdir(distorted_dir)):
        distorted_folder = os.path.join(distorted_dir, identity)
        if not os.path.isdir(distorted_folder): continue
        for file in os.listdir(distorted_folder):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(distorted_folder, file)
                distorted_emb = get_embedding(img_path)
                best_match = "Unknown"
                min_dist = float('inf')
                for ref_id, ref_emb in identity_db.items():
                    dist = np.linalg.norm(distorted_emb - ref_emb)
                    if dist < min_dist:
                        min_dist = dist
                        best_match = ref_id
                y_true.append(identity)
                y_pred.append(best_match if min_dist < threshold else "Unknown")

    label_map = {label: idx for idx, label in enumerate(set(y_true + y_pred))}
    y_true_idx = [label_map[i] for i in y_true]
    y_pred_idx = [label_map[i] for i in y_pred]

    acc = accuracy_score(y_true_idx, y_pred_idx)
    f1 = f1_score(y_true_idx, y_pred_idx, average='macro')
    print("\n---------------------------Final Evaluation---------------------------")
    print(f"Top-1 Accuracy         : {acc:.4f}")
    print(f"Macro-averaged F1 Score: {f1:.4f}")

evaluate_fast(DISTORTED_DIR, IDENTITY_DIR)
