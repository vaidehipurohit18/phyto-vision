import sys
import json
import time
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR,
    MODEL_PATH,
    IMAGE_SIZE,
    RESULTS_DIR
)

from training.attention import CBAMLayer


def fine_tune(
    epochs=8,
    batch_size=32,
    learning_rate=1e-5,
    trainable_layers=30
):

    print("=" * 70)
    print("STARTING FINE-TUNING")
    print("=" * 70)

    train_dir = DATASET_DIR / "train"
    val_dir = DATASET_DIR / "validation"

    # --------------------------------------------------
    # LOAD EXISTING MODEL
    # --------------------------------------------------

    print("\nLoading existing trained model...")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "CBAMLayer": CBAMLayer
        }
    )

    print("Existing model loaded successfully.")

    # --------------------------------------------------
    # FIND MOBILENETV2
    # --------------------------------------------------

    base_model = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.Model
        ) and "mobilenet" in layer.name.lower():

            base_model = layer
            break

    if base_model is None:

        raise ValueError(
            "MobileNetV2 base model could not be found."
        )

    print(
        f"\nFound base model: {base_model.name}"
    )

    # --------------------------------------------------
    # FREEZE EVERYTHING FIRST
    # --------------------------------------------------

    base_model.trainable = True

    for layer in base_model.layers:

        layer.trainable = False

    # --------------------------------------------------
    # UNFREEZE ONLY LAST LAYERS
    # --------------------------------------------------

    for layer in base_model.layers[-trainable_layers:]:

        # Keep BatchNormalization frozen
        if not isinstance(
            layer,
            tf.keras.layers.BatchNormalization
        ):

            layer.trainable = True

    trainable_count = sum(
        1
        for layer in base_model.layers
        if layer.trainable
    )

    print(
        f"\nUnfroze {trainable_count} "
        f"MobileNetV2 layers."
    )

    # --------------------------------------------------
    # DATA AUGMENTATION
    # --------------------------------------------------

    train_datagen = ImageDataGenerator(

        rotation_range=20,

        width_shift_range=0.10,

        height_shift_range=0.10,

        zoom_range=0.10,

        horizontal_flip=True,

        brightness_range=[0.9, 1.1],

        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator()

    # --------------------------------------------------
    # DATA GENERATORS
    # --------------------------------------------------

    train_generator = train_datagen.flow_from_directory(

        train_dir,

        target_size=IMAGE_SIZE,

        batch_size=batch_size,

        class_mode="categorical",

        shuffle=True
    )

    val_generator = val_datagen.flow_from_directory(

        val_dir,

        target_size=IMAGE_SIZE,

        batch_size=batch_size,

        class_mode="categorical",

        shuffle=False
    )

    # --------------------------------------------------
    # COMPILE WITH SMALL LR
    # --------------------------------------------------

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate
        ),

        loss="categorical_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    # --------------------------------------------------
    # SAVE BEST FINE-TUNED MODEL
    # --------------------------------------------------

    fine_tuned_model_path = (
        MODEL_PATH.parent /
        "plant_disease_finetuned.keras"
    )

    callbacks = [

        ModelCheckpoint(

            filepath=str(
                fine_tuned_model_path
            ),

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

            factor=0.3,

            patience=1,

            min_lr=1e-7,

            verbose=1
        )
    ]

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    print("\n" + "=" * 70)

    print("FINE-TUNING STARTED")

    print("=" * 70)

    print(f"\nMaximum epochs: {epochs}")

    print(
        "Early stopping is enabled."
    )

    print(
        "Training may stop before "
        f"{epochs} epochs."
    )

    start_time = time.time()

    history = model.fit(

        train_generator,

        validation_data=val_generator,

        epochs=epochs,

        callbacks=callbacks
    )

    duration = time.time() - start_time

    # --------------------------------------------------
    # SAVE HISTORY
    # --------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    history_path = (
        RESULTS_DIR /
        "fine_tuning_history.json"
    )

    history_dict = {

        key: [
            float(v)
            for v in values
        ]

        for key, values
        in history.history.items()
    }

    with open(
        history_path,
        "w"
    ) as f:

        json.dump(
            history_dict,
            f,
            indent=4
        )

    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    print("\n" + "=" * 70)

    print("FINE-TUNING COMPLETE")

    print("=" * 70)

    print(
        f"\nTraining time: "
        f"{duration / 60:.2f} minutes"
    )

    print(
        f"\nBest fine-tuned model:"
    )

    print(
        fine_tuned_model_path
    )

    print(
        f"\nFinal validation accuracy: "
        f"{history.history['val_accuracy'][-1] * 100:.2f}%"
    )


if __name__ == "__main__":

    fine_tune()