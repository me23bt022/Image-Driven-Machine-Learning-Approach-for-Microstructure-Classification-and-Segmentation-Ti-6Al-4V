"""
----------------------------------ABOUT-----------------------------------
Author: Arun Baskaran
Modernized for TensorFlow 2.x (2026 compatible)
--------------------------------------------------------------------------
"""

# ========================
# Core Libraries
# ========================
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

from PIL import Image

# ========================
# TensorFlow / Keras
# ========================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    MaxPooling2D,
    Reshape,
    Flatten,
    Dropout
)
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import regularizers

# ========================
# Image Processing
# ========================
from scipy import ndimage as ndi

from skimage import exposure, morphology
from skimage.segmentation import watershed   # ✅ updated location
from skimage.color import label2rgb
from skimage.filters import sobel
from skimage.feature import peak_local_max, hog
