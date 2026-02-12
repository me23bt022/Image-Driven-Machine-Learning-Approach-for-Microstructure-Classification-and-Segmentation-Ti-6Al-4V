"""
----------------------------------ABOUT-----------------------------------
Author: Arun Baskaran
Modernized for TensorFlow 2.x
---------------------------------------------------------------------------
"""

import sys
import model_params
from aux_funcs import (
    load_images_labels,
    train_model,
    load_model,
    test_accuracy,
    get_predicted_classes,
    feature_segmentation
)


def main(mode="training"):

    # Load dataset
    X_train, y_train, X_test, y_test, X_val, y_val = load_images_labels()

    if mode == "training":
        model, history = train_model(X_train, y_train, X_val, y_val)

    elif mode == "load":
        model = load_model()

    else:
        raise ValueError("Mode must be 'training' or 'load'")

    # Evaluate model
    test_accuracy(model, X_test, y_test)

    # Get predictions
    y_classes = get_predicted_classes(model, X_test)

    # Run segmentation
    feature_segmentation(y_classes)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(sys.argv[1])
    else:
        main("training")
