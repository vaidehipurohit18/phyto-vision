import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "training",
    "results"
)

CONFUSION_MATRIX_PATH = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Apple Scab",
    "Apple Black Rot",
    "Apple Cedar Rust",
    "Apple Healthy",
    "Blueberry Healthy",
    "Cherry Powdery Mildew",
    "Cherry Healthy",
    "Corn Gray Leaf Spot",
    "Corn Common Rust",
    "Corn Northern Leaf Blight",
    "Corn Healthy",
    "Grape Black Rot",
    "Grape Esca",
    "Grape Leaf Blight",
    "Grape Healthy",
    "Orange Citrus Greening",
    "Peach Bacterial Spot",
    "Peach Healthy",
    "Bell Pepper Bacterial Spot",
    "Bell Pepper Healthy",
    "Potato Early Blight",
    "Potato Late Blight",
    "Potato Healthy",
    "Raspberry Healthy",
    "Soybean Healthy",
    "Squash Powdery Mildew",
    "Strawberry Leaf Scorch",
    "Strawberry Healthy",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Yellow Leaf Curl Virus",
    "Tomato Mosaic Virus",
    "Tomato Healthy"
]


# ============================================================
# LOAD CONFUSION MATRIX
# ============================================================

print("\nLoading confusion matrix...")

cm = np.load(CONFUSION_MATRIX_PATH)

print("Confusion matrix shape:", cm.shape)


# ============================================================
# RAW CONFUSION MATRIX
# ============================================================

plt.figure(figsize=(22, 18))

plt.imshow(cm, interpolation="nearest")

plt.title("Confusion Matrix - Plant Disease Classification")

plt.colorbar()

ticks = np.arange(len(CLASS_NAMES))

plt.xticks(
    ticks,
    CLASS_NAMES,
    rotation=90,
    fontsize=7
)

plt.yticks(
    ticks,
    CLASS_NAMES,
    fontsize=7
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

raw_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_heatmap.png"
)

plt.savefig(
    raw_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Raw confusion matrix saved:")
print(raw_path)


# ============================================================
# NORMALIZED CONFUSION MATRIX
# ============================================================

cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(22, 18))

plt.imshow(
    cm_normalized,
    interpolation="nearest",
    vmin=0,
    vmax=1
)

plt.title("Normalized Confusion Matrix - Plant Disease Classification")

plt.colorbar()

plt.xticks(
    ticks,
    CLASS_NAMES,
    rotation=90,
    fontsize=7
)

plt.yticks(
    ticks,
    CLASS_NAMES,
    fontsize=7
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")

plt.tight_layout()

normalized_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix_normalized.png"
)

plt.savefig(
    normalized_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Normalized confusion matrix saved:")
print(normalized_path)

print("\n================================================")
print("HEATMAP GENERATION COMPLETE")
print("================================================")