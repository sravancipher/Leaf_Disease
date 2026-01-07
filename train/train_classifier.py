import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models

# ---- PATHS (MATCH CODE 1 STYLE) ----
BASE_DIR = os.path.dirname(__file__)
IMAGE_ROOT = os.path.abspath(os.path.join(BASE_DIR, "../PlantVillage"))
CSV_PATH = os.path.abspath(os.path.join(BASE_DIR, "../labels.csv"))
MODEL_OUT = os.path.abspath(os.path.join(BASE_DIR, "../models/vgg19_disease_classifier.keras"))

print(f"IMAGE_ROOT: {IMAGE_ROOT}")
print(f"CSV_PATH: {CSV_PATH}")

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 25

# ---- LOAD LABELS ----
df = pd.read_csv(CSV_PATH)

# image column contains: Disease_Name/image.jpg
df["image_path"] = df["image"].apply(lambda x: os.path.join(IMAGE_ROOT, x))

num_classes = df["label"].nunique()

# ---- DATA LOADER ----
def load_image_label(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img, tf.one_hot(label, num_classes)

paths = df["image_path"].values
labels = df["label"].values

dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
dataset = dataset.shuffle(1000)
dataset = dataset.map(load_image_label, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# ---- MODEL ----
base_model = VGG19(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ---- TRAIN ----
model.fit(
    dataset,
    epochs=EPOCHS
)

# ---- SAVE ----
model.save(MODEL_OUT)
print(f"Model saved at: {MODEL_OUT}")
