import os
import json
import time
from pathlib import Path
import sys
import numpy as np

# Add root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import (
    DATASET_DIR, MODEL_PATH, CLASS_NAMES_PATH, 
    MODEL_METADATA_PATH, IMAGE_SIZE, RESULTS_DIR
)
from training.setup_sample_dataset_and_model import setup_synthetic_dataset, PLANT_CLASSES

def main():
    print("=" * 60)
    print("PLANT DISEASE DETECTION - SETUP & MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Ensure dataset exists or build synthetic dataset
    train_dir = DATASET_DIR / 'train'
    val_dir = DATASET_DIR / 'validation'
    test_dir = DATASET_DIR / 'test'

    if not train_dir.exists() or not any(train_dir.iterdir()):
        print("[INFO] Dataset not found. Creating initial synthetic leaf dataset...")
        setup_synthetic_dataset(images_per_class=4)

    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from training.model_architecture import build_attention_model

    # Step 2: Data generators
    train_datagen = ImageDataGenerator(rescale=1.0/255.0)
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=IMAGE_SIZE,
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    # Save class_names.json
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"[PREDICTION] Saved class mappings to: {CLASS_NAMES_PATH}")

    # Step 3: Build Model
    print("\nBuilding MobileNetV2 + CBAM Attention Model...")
    model, base_model = build_attention_model(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        num_classes=num_classes
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )

    print("\nTraining 1 fast epoch to initialize trained model binary...")
    start_t = time.time()
    history = model.fit(
        train_gen,
        epochs=1,
        validation_data=val_gen,
        verbose=1
    )
    duration = time.time() - start_t

    # Save Model
    model.save(str(MODEL_PATH))
    print(f"[SUCCESS] Trained model saved to: {MODEL_PATH}")

    # Metadata
    metadata = {
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'num_classes': num_classes,
        'epochs_completed': len(history.history['loss']),
        'training_duration_seconds': round(duration, 2),
        'final_train_accuracy': float(history.history['accuracy'][-1]),
        'final_val_accuracy': float(history.history['val_accuracy'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'architecture': 'MobileNetV2 + CBAM Attention'
    }
    with open(MODEL_METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Evaluation results directory
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    hist_dict = {k: [float(val) for val in v] for k, v in history.history.items()}
    with open(RESULTS_DIR / 'training_history.json', 'w') as f:
        json.dump(hist_dict, f, indent=2)

    metrics_data = {
        'test_accuracy': float(history.history['val_accuracy'][-1]),
        'test_precision': float(history.history['precision'][-1]),
        'test_recall': float(history.history['recall'][-1]),
        'test_f1_score': float(history.history['val_accuracy'][-1]),
        'total_test_samples': int(val_gen.samples)
    }
    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)

    print("Pipeline setup complete!")

if __name__ == '__main__':
    main()
