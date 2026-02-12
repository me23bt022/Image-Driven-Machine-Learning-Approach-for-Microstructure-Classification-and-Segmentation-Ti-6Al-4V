"""
Model and Dataset Configuration
Author: Arun Baskaran (Modified for TF2 compatibility)
"""

from pathlib import Path

# =========================
# IMAGE PARAMETERS
# =========================
IMG_WIDTH = 200
IMG_HEIGHT = 200
IMG_CHANNELS = 1  # Grayscale

# =========================
# DATASET PARAMETERS
# =========================
TOTAL_SIZE = 1225
TRAIN_SIZE = 900
VALIDATION_SIZE = 100
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE - VALIDATION_SIZE

NUM_CLASSES = 3

# =========================
# TRAINING PARAMETERS
# =========================
BATCH_SIZE = 32
EPOCHS = 50  # Reduced from 1500 (more realistic for Colab)

LEARNING_RATE = 1e-3

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR_1 = BASE_DIR / "Images1"
IMAGES_DIR_2 = BASE_DIR / "Images2"
LABELS_FILE = BASE_DIR / "labels.xlsx"
WEIGHTS_DIR = BASE_DIR / "weights"
