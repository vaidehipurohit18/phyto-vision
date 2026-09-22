import os
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

# Add root path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import DATASET_DIR, MODEL_PATH, CLASS_NAMES_PATH, IMAGE_SIZE, RESULTS_DIR
from training.attention import CBAMLayer

def evaluate_model():
    """
    Evaluates the trained model on test dataset. Computes real Accuracy, Precision, Recall, F1,
    and generates confusion matrix & accuracy/loss plots saved to training/results/.
    """
    if not MODEL_PATH.exists():
        print(f"[ERROR] Trained model file not found at: {MODEL_PATH}")
        print("Please train the model first by running: python training/train_model.py")
        return False

    test_dir = DATASET_DIR / 'test'
    if not test_dir.exists() or not any(test_dir.iterdir()):
        print(f"[ERROR] Test dataset folder empty or missing at: {test_dir}")
        return False

    print("=" * 60)
    print("EVALUATING ATTENTION-BASED CNN MODEL ON TEST SET")
    print("=" * 60)

    # Load custom object CBAMLayer
    custom_objects = {'CBAMLayer': CBAMLayer}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names_dict = json.load(f)
    
    # Ensure correct order
    labels = [class_names_dict[str(i)] for i in range(len(class_names_dict))]

    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=IMAGE_SIZE,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    print(f"\nTest Evaluation Metrics:")
    print(f"  Accuracy:  {acc * 100:.2f}%")
    print(f"  Precision: {precision * 100:.2f}%")
    print(f"  Recall:    {recall * 100:.2f}%")
    print(f"  F1 Score:  {f1 * 100:.2f}%")

    report_str = classification_report(y_true, y_pred, target_names=labels)
    print("\nClassification Report:\n", report_str)

    # Save metrics JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_data = {
        'test_accuracy': float(acc),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1_score': float(f1),
        'total_test_samples': int(len(y_true))
    }

    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics_data, f, indent=2)

    with open(RESULTS_DIR / 'classification_report.txt', 'w') as f:
        f.write(report_str)

    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    plt.title('Plant Disease Detection - Confusion Matrix (MobileNetV2 + CBAM)')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'confusion_matrix.png', dpi=300)
    plt.close()

    # Load history if available to save Loss/Accuracy curves
    history_path = RESULTS_DIR / 'training_history.json'
    if history_path.exists():
        with open(history_path, 'r') as f:
            hist = json.load(f)

        epochs_range = range(1, len(hist['loss']) + 1)

        # Plot Accuracy
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, hist['accuracy'], 'g-o', label='Train Accuracy')
        plt.plot(epochs_range, hist['val_accuracy'], 'b--s', label='Validation Accuracy')
        plt.title('Training & Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'accuracy.png', dpi=300)
        plt.close()

        # Plot Loss
        plt.figure(figsize=(8, 5))
        plt.plot(epochs_range, hist['loss'], 'r-o', label='Train Loss')
        plt.plot(epochs_range, hist['val_loss'], 'm--s', label='Validation Loss')
        plt.title('Training & Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Categorical Crossentropy Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'loss.png', dpi=300)
        plt.close()

    print(f"\n[SUCCESS] Model evaluation metrics and plots saved to: {RESULTS_DIR}")
    return True

if __name__ == '__main__':
    evaluate_model()
