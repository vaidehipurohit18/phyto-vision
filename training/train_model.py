import os
import json
import time
from pathlib import Path
import sys

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Add root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR,
    MODEL_PATH,
    CLASS_NAMES_PATH,
    MODEL_METADATA_PATH,
    IMAGE_SIZE,
    RESULTS_DIR
)

from training.model_architecture import build_attention_model


def validate_dataset_structure():
    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation"

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print(f"\n[ERROR] Training dataset is missing or empty:")
        print(train_dir)
        return False, None, None

    if not val_dir.exists() or not any(val_dir.iterdir()):
        print(f"\n[ERROR] Validation dataset is missing or empty:")
        print(val_dir)
        return False, None, None

    return True, train_dir, val_dir


def train_pipeline(epochs=8, batch_size=32, learning_rate=1e-3):

    valid, train_dir, val_dir = validate_dataset_structure()

    if not valid:
        return False

    print("=" * 60)
    print("STARTING ATTENTION-BASED CNN TRAINING PIPELINE")
    print("=" * 60)

    print("\nTraining directory:", train_dir)
    print("Validation directory:", val_dir)
    print("Image size:", IMAGE_SIZE)
    print("Maximum epochs:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)

    # ---------------------------------------------------------
    # DATA AUGMENTATION
    # ---------------------------------------------------------
    # IMPORTANT:
    # No rescale=1/255 here because MobileNetV2 preprocessing
    # is already performed inside model_architecture.py.
    # ---------------------------------------------------------

    train_datagen = ImageDataGenerator(
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

    val_datagen = ImageDataGenerator()

    # ---------------------------------------------------------
    # TRAINING DATA
    # ---------------------------------------------------------

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    # ---------------------------------------------------------
    # VALIDATION DATA
    # ---------------------------------------------------------

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    num_classes = train_generator.num_classes

    class_indices = train_generator.class_indices

    # Convert:
    # {"Apple___Apple_scab": 0}
    #
    # into:
    # {"0": "Apple___Apple_scab"}

    class_names = {
        str(v): k
        for k, v in class_indices.items()
    }

    print(f"\nDetected {num_classes} plant disease classes:")

    for idx, name in class_names.items():
        print(f"  [{idx}]: {name}")

    # ---------------------------------------------------------
    # SAVE CLASS MAPPING
    # ---------------------------------------------------------

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved class mapping to:")
    print(CLASS_NAMES_PATH)

    # ---------------------------------------------------------
    # BUILD MODEL
    # ---------------------------------------------------------

    print("\nBuilding MobileNetV2 + CBAM model...")

    model, base_model = build_attention_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=num_classes
    )

    print("\nMobileNetV2 base model is frozen.")
    print("Training CBAM + classification layers.")

    # ---------------------------------------------------------
    # COMPILE
    # ---------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )

    model.summary()

    # ---------------------------------------------------------
    # CALLBACKS
    # ---------------------------------------------------------

    callbacks = [

        ModelCheckpoint(
            filepath=str(MODEL_PATH),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=1,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING STARTED")
    print("=" * 60)

    start_time = time.time()

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    training_duration = time.time() - start_time

    # ---------------------------------------------------------
    # SAVE METADATA
    # ---------------------------------------------------------

    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_classes": num_classes,
        "epochs_requested": epochs,
        "epochs_completed": len(history.history["loss"]),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_duration_seconds": round(
            training_duration,
            2
        ),
        "final_train_accuracy": float(
            history.history["accuracy"][-1]
        ),
        "final_val_accuracy": float(
            history.history["val_accuracy"][-1]
        ),
        "final_val_loss": float(
            history.history["val_loss"][-1]
        ),
        "architecture": "MobileNetV2 + CBAM Attention",
        "base_model_frozen": True
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    # ---------------------------------------------------------
    # SAVE TRAINING HISTORY
    # ---------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history_file = RESULTS_DIR / "training_history.json"

    hist_dict = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with open(history_file, "w") as f:
        json.dump(hist_dict, f, indent=2)

    # ---------------------------------------------------------
    # FINAL MESSAGE
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("[SUCCESS] MODEL TRAINING COMPLETE")
    print("=" * 60)

    print("\nBest model saved to:")
    print(MODEL_PATH)

    print("\nClass mapping saved to:")
    print(CLASS_NAMES_PATH)

    print("\nMetadata saved to:")
    print(MODEL_METADATA_PATH)

    print("\nTraining time:")
    print(
        round(training_duration / 60, 2),
        "minutes"
    )

    print("\nFinal validation accuracy:")
    print(
        round(
            history.history["val_accuracy"][-1] * 100,
            2
        ),
        "%"
    )

    return True


if __name__ == "__main__":
    train_pipeline()