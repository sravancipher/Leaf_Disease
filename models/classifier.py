import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models

def build_classifier(input_shape=(224, 224, 3), num_classes=3):
    base_model = VGG19(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])

    return model
