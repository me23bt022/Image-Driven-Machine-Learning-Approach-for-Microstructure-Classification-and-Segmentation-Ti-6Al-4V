"""
----------------------------------ABOUT-----------------------------------
Updated for TensorFlow 2.x (2026 compatible)
Original Author: Arun Baskaran
---------------------------------------------------------------------------
"""

import os
import random
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from sklearn.model_selection import train_test_split
from PIL import Image
from skimage import exposure, morphology
from skimage.filters import sobel
from skimage.color import label2rgb
from scipy import ndimage as ndi

import model_params

# ==============================
# Helper Functions
# ==============================

def smooth(img):
    return 0.5 * img + 0.5 * (
        np.roll(img, +1, axis=0) + np.roll(img, -1, axis=0) +
        np.roll(img, +1, axis=1) + np.roll(img, -1, axis=1)
    )


def returnIndex(a, value):
    for i in range(len(a)):
        if a[i] == value:
            return i
    return None


# ==============================
# MODEL CREATION (Modernized)
# ==============================

def create_model():
    model = keras.Sequential([
        layers.Input(shape=(model_params.width, model_params.height, 1)),

        layers.Conv2D(
            16, (5, 5),
            padding='valid',
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l1(0.001),
            activation='relu'
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            32, (5, 5),
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l1(0.001),
            activation='relu'
        ),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(
            64, (3, 3),
            kernel_initializer="glorot_uniform",
            kernel_regularizer=regularizers.l1(0.001),
            activation='relu'
        ),

        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(3, activation='softmax')
    ])

    return model


# ==============================
# DATA LOADING (Improved)
# ==============================

def load_images_labels():

    df = pd.read_excel('labels.xlsx', header=None, names=['id', 'label'])
    labels = df['label'].values - 1   # zero-based labels

    images = []

    for i in range(1, model_params.total_size + 1):
        filename = f'image_{i}.png'
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image,
                           (model_params.width, model_params.height))
        image = cv2.GaussianBlur(image, (5, 5), 0)

        # Safe normalization
        image = image.astype("float32") / 255.0

        images.append(image)

    images = np.array(images)
    images = np.expand_dims(images, axis=-1)

    # Split dataset (modern way)
    X_train, X_temp, y_train, y_temp = train_test_split(
        images, labels,
        train_size=model_params.train_size,
        stratify=labels,
        random_state=42
    )

    val_size_adjusted = model_params.validation_size / (
        model_params.validation_size + model_params.test_size
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=1 - val_size_adjusted,
        stratify=y_temp,
        random_state=42
    )

    # One-hot encode
    y_train = keras.utils.to_categorical(y_train, 3)
    y_val = keras.utils.to_categorical(y_val, 3)
    y_test = keras.utils.to_categorical(y_test, 3)

    return X_train, y_train, X_test, y_test, X_val, y_val


# ==============================
# TRAINING
# ==============================

def train_model(X_train, y_train, X_val, y_val):

    model = create_model()

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    os.makedirs("weights", exist_ok=True)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            "weights/classification.keras",
            save_best_only=True
        )
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks
    )

    return model, history


# ==============================
# TESTING
# ==============================

def test_accuracy(model, X_test, y_test):
    loss, acc = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {acc * 100:.2f}%")


def get_predicted_classes(model, X_test):
    y_prob = model.predict(X_test)
    return np.argmax(y_prob, axis=-1)
