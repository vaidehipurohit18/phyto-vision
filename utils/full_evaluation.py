import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

from config import (
    DATASET_DIR,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    IMAGE_SIZE,
    RESULTS_DIR
)

from training.attention import CBAMLayer


def main():

    print("\n" + "=" * 80)
    print("FULL VALIDATION DATASET EVALUATION")
    print("=" * 80)

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    print("\nLoading model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "CBAMLayer": CBAMLayer
        }
    )

    print("Model loaded successfully.")

    # --------------------------------------------------
    # LOAD CLASS NAMES
    # --------------------------------------------------

    with open(CLASS_NAMES_PATH, "r") as f:
        class_mapping = json.load(f)

    class_names = [
        class_mapping[str(i)]
        for i in range(len(class_mapping))
    ]

    # --------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------

    val_dir = DATASET_DIR / "validation"

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode="categorical",
        shuffle=False
    )

    print("\nTotal validation images:", val_generator.samples)
    print("Total classes:", val_generator.num_classes)

    # --------------------------------------------------
    # MODEL PREDICTIONS
    # --------------------------------------------------

    print("\nRunning predictions on FULL validation dataset...")

    predictions = model.predict(
        val_generator,
        verbose=1
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    y_true = val_generator.classes

    # --------------------------------------------------
    # OVERALL ACCURACY
    # --------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    print("\n" + "=" * 80)
    print("OVERALL VALIDATION ACCURACY")
    print("=" * 80)

    print(f"\nAccuracy: {accuracy * 100:.2f}%")

    # --------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(report)

    # --------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save accuracy

    metrics = {
        "validation_accuracy": float(accuracy),
        "validation_accuracy_percent": round(
            float(accuracy * 100),
            2
        ),
        "total_images": int(len(y_true)),
        "total_classes": int(len(class_names))
    }

    metrics_path = RESULTS_DIR / "full_evaluation_metrics.json"

    with open(metrics_path, "w") as f:
        json.dump(
            metrics,
            f,
            indent=4
        )

    # Save confusion matrix

    cm_path = RESULTS_DIR / "confusion_matrix.npy"

    np.save(
        cm_path,
        cm
    )

    # Save classification report

    report_path = RESULTS_DIR / "classification_report.txt"

    with open(report_path, "w") as f:
        f.write(report)

    # --------------------------------------------------
    # FIND MOST CONFUSED CLASSES
    # --------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP CONFUSED CLASS PAIRS")
    print("=" * 80)

    confused_pairs = []

    for true_class in range(len(class_names)):

        for predicted_class in range(len(class_names)):

            if true_class != predicted_class:

                count = cm[
                    true_class,
                    predicted_class
                ]

                if count > 0:

                    confused_pairs.append(
                        (
                            count,
                            class_names[true_class],
                            class_names[predicted_class]
                        )
                    )

    confused_pairs.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    for count, true_name, predicted_name in confused_pairs[:20]:

        print(
            f"\n{count} images"
        )

        print(
            f"Actual    : {true_name}"
        )

        print(
            f"Predicted : {predicted_name}"
        )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)

    print("\nSaved files:")

    print(metrics_path)

    print(cm_path)

    print(report_path)


if __name__ == "__main__":
    main()