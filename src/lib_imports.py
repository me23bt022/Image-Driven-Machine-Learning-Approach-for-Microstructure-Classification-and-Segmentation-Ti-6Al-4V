"""
----------------------------------ABOUT-----------------------------------
Author: Arun Baskaran
--------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os.path
import random
tf.get_logger().setLevel('ERROR')


# Keras (correct modern imports)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, Dense, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

# Scientific libraries
from scipy import ndimage as ndi

# Updated scikit-image imports
from skimage.segmentation import watershed
from skimage.morphology import disk
from skimage.feature import peak_local_max, hog
from skimage.filters import sobel
from skimage import exposure, data, morphology
from skimage.color import label2rgb
