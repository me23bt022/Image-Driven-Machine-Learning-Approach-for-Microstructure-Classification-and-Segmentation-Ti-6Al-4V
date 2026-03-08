"""
----------------------------------ABOUT-----------------------------------
Author: Arun Baskaran
--------------------------------------------------------------------------
"""

# Testing edits
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import lib_imports
from aux_funcs import *
import model_params
import sys
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: main <mode>", file=sys.stderr)
        sys.exit(-1)
    mode = sys.argv[1] 
    train_images, train_labels, test_images, test_labels, validation_images, validation_labels = load_images_labels()
    
    if mode == "training" :
        model = train_model()
    
    elif mode =="load":
        model = load_model()
        
    test_accuracy(model, test_images, test_labels)
    
    y_classes = get_predicted_classes(model, test_images)
    print(y_classes)
    feature_segmentation(y_classes)
    



