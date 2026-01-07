import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tensorflow as tf
from models.unet import build_unet
from utils.losses import combined_loss

IMAGE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PlantVillage"))
MASK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../PlantVillage_Masks"))

print(f"IMAGE_ROOT: {IMAGE_ROOT}")
print(f"MASK_ROOT: {MASK_ROOT}")
print(f"Current working directory: {os.getcwd()}")

IMG_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 25

# ------------------ LOADERS ------------------
def load_image_mask(img_path, mask_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1)
    mask.set_shape([None, None, 1])
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = tf.cast(mask, tf.float32) / 255.0

    return img, mask

# ------------------ GENERATOR ------------------
def dataset_generator(image_root, mask_root):
    for disease in sorted(os.listdir(image_root)):
        img_disease_dir = os.path.join(image_root, disease)
        mask_disease_dir = os.path.join(mask_root, disease)

        if not os.path.isdir(img_disease_dir):
            continue
        if not os.path.isdir(mask_disease_dir):
            continue

        for img_name in sorted(os.listdir(img_disease_dir)):
            if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            img_path = os.path.join(img_disease_dir, img_name)
            mask_path = os.path.join(mask_disease_dir, img_name)

            if not os.path.exists(mask_path):
                continue

            yield img_path, mask_path

# ------------------ DATASET ------------------
def build_dataset(image_root, mask_root):
    ds = tf.data.Dataset.from_generator(
        lambda: dataset_generator(image_root, mask_root),
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.string),
            tf.TensorSpec(shape=(), dtype=tf.string),
        )
    )

    ds = ds.map(
        load_image_mask,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.shuffle(1000)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# ------------------ TRAIN ------------------
if __name__ == "__main__":

    # OPTIONAL DEBUG (run once)
    count = 0
    for _ in dataset_generator(IMAGE_ROOT, MASK_ROOT):
        count += 1
    print("TOTAL IMAGE-MASK PAIRS:", count)

    train_ds = build_dataset(IMAGE_ROOT, MASK_ROOT)

    model = build_unet(input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=combined_loss,
        metrics=["accuracy"]
    )

    print("Testing one batch...")
    for x, y in train_ds.take(1):
        print("Batch loaded:", x.shape, y.shape)
    print("Batch OK, starting training...")

    model.fit(
        train_ds,
        epochs=1,
        verbose=2
    )

    model.save(os.path.join(os.path.dirname(__file__), "../models/unet_leaf_segmentation.keras"))
