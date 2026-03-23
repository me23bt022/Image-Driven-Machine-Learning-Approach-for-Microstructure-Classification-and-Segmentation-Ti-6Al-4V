from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

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
    
    if mode == "training":
        model = train_model()
    elif mode == "load":
        model = load_model()

    test_accuracy(model, test_images, test_labels)
    
    y_classes = get_predicted_classes(model, test_images)
    print(y_classes)

    # ❌ REMOVED THIS LINE (causing spam)
    # feature_segmentation(y_classes)

    # ------------------ EVALUATION ------------------

    y_true = np.argmax(test_labels, axis=1)

    # Confusion Matrix (3x3)
    cm = confusion_matrix(y_true, y_classes)

    plt.figure(figsize=(6,5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        xticklabels=['Lamellae','Bi-modal','Acicular'],
        yticklabels=['Lamellae','Bi-modal','Acicular']
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("3x3 Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    plt.show()
    plt.close()

    # Metrics
    print("\nAccuracy:", accuracy_score(y_true, y_classes))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        y_classes,
        target_names=['Lamellae','Bi-modal','Acicular']   # ✅ added
    ))
