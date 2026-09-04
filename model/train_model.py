"""
THERMAL IMAGE BASED FEVER & INFECTION DETECTION
------------------------------------------------
CNN Training Script

This script trains a Convolutional Neural Network to classify facial
thermal images into two categories:
    0 -> normal      (normal body temperature)
    1 -> fever        (elevated / abnormal body temperature)

DATASET STRUCTURE EXPECTED (place your thermal images accordingly):

    dataset/
        train/
            normal/   *.jpg / *.png
            fever/    *.jpg / *.png
        val/
            normal/   *.jpg / *.png
            fever/    *.jpg / *.png

Run:
    python train_model.py

Output:
    model/thermal_fever_cnn.h5   -> trained Keras model used by the Flask app
    model/training_history.png   -> accuracy / loss curves
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------
IMG_SIZE = (128, 128)       # input image size fed to the CNN
BATCH_SIZE = 16
EPOCHS = 25
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "..", "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
MODEL_OUT = os.path.join(BASE_DIR, "thermal_fever_cnn.h5")
HISTORY_PLOT_OUT = os.path.join(BASE_DIR, "training_history.png")

CLASS_NAMES = ["fever", "normal"]  # alphabetical order used by Keras flow_from_directory


# ----------------------------------------------------------------------
# 1. IMAGE PREPROCESSING & AUGMENTATION
#    (Chapter 5.2 - Image Preprocessing Module)
# ----------------------------------------------------------------------
def build_data_generators():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,          # normalization
        rotation_range=10,
        width_shift_range=0.08,
        height_shift_range=0.08,
        zoom_range=0.1,
        brightness_range=[0.85, 1.15],  # simulate ambient temperature/lighting variation
        horizontal_flip=True,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=True,
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="rgb",
        class_mode="binary",
        classes=CLASS_NAMES,
        shuffle=False,
    )

    return train_gen, val_gen


# ----------------------------------------------------------------------
# 2. CNN ARCHITECTURE
#    (Chapter 5.2 - CNN-Based Feature Extraction & Classification Module)
# ----------------------------------------------------------------------
def build_cnn(input_shape=(128, 128, 3)):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),  # binary classification: fever vs normal
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall")],
    )
    return model


# ----------------------------------------------------------------------
# 3. TRAINING
# ----------------------------------------------------------------------
def count_images(directory):
    total = 0
    if not os.path.isdir(directory):
        return 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(directory, cls)
        if os.path.isdir(cls_dir):
            total += len([f for f in os.listdir(cls_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return total


def main():
    n_train = count_images(TRAIN_DIR)
    n_val = count_images(VAL_DIR)

    if n_train == 0 or n_val == 0:
        print("=" * 70)
        print("No training images found.")
        print(f"Add images under:\n  {TRAIN_DIR}/normal , {TRAIN_DIR}/fever")
        print(f"  {VAL_DIR}/normal , {VAL_DIR}/fever")
        print("Then re-run this script.")
        print("=" * 70)
        return

    print(f"Found {n_train} training images and {n_val} validation images.")

    train_gen, val_gen = build_data_generators()
    model = build_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6
        ),
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_OUT)
    print(f"\nModel saved to: {MODEL_OUT}")

    # Plot accuracy / loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"], label="train_acc")
    axes[0].plot(history.history["val_accuracy"], label="val_acc")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="train_loss")
    axes[1].plot(history.history["val_loss"], label="val_loss")
    axes[1].set_title("Loss")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(HISTORY_PLOT_OUT)
    print(f"Training curves saved to: {HISTORY_PLOT_OUT}")


if __name__ == "__main__":
    main()
