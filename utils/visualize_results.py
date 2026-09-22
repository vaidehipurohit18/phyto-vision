import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RESULTS_DIR = os.path.join(BASE_DIR, "training", "results")
OUTPUT_DIR = os.path.join(RESULTS_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD FILES
# ============================================================

confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.npy"
)

metrics_path = os.path.join(
    RESULTS_DIR,
    "full_evaluation_metrics.json"
)


print("\nLoading evaluation results...")

cm = np.load(confusion_matrix_path)

with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = [
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
# 1. CONFUSION MATRIX
# ============================================================

print("Generating confusion matrix...")

fig, ax = plt.subplots(figsize=(20, 20))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    values_format="d",
    colorbar=True
)

plt.title(
    "Confusion Matrix - Plant Disease Classification",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 2. NORMALIZED CONFUSION MATRIX
# ============================================================

print("Generating normalized confusion matrix...")

cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

fig, ax = plt.subplots(figsize=(20, 20))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm_normalized,
    display_labels=class_names
)

display.plot(
    ax=ax,
    xticks_rotation=90,
    values_format=".2f",
    colorbar=True
)

plt.title(
    "Normalized Confusion Matrix - Plant Disease Classification",
    fontsize=18
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "normalized_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# 3. PER-CLASS F1 SCORE
# ============================================================

print("Generating F1-score chart...")

classification_report = metrics["classification_report"]

f1_scores = []

for class_name in class_names:
    matching_key = None

    for key in classification_report.keys():
        if key in [
            "accuracy",
            "macro avg",
            "weighted avg"
        ]:
            continue

        clean_key = key.replace("___", " ")

        if class_name.lower().split()[0] in clean_key.lower():
            pass

    f1_scores.append(None)


# Extract directly from classification report
raw_class_names = [
    key
    for key in classification_report.keys()
    if key not in [
        "accuracy",
        "macro avg",
        "weighted avg"
    ]
]

f1_scores = [
    classification_report[name]["f1-score"]
    for name in raw_class_names
]

short_names = [
    name.replace("___", "\n")
    for name in raw_class_names
]


fig, ax = plt.subplots(figsize=(14, 18))

positions = np.arange(len(short_names))

ax.barh(
    positions,
    f1_scores
)

ax.set_yticks(positions)

ax.set_yticklabels(
    short_names,
    fontsize=8
)

ax.set_xlabel("F1 Score")

ax.set_title(
    "Per-Class F1 Scores",
    fontsize=18
)

ax.set_xlim(0, 1.05)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "per_class_f1_scores.png"
    ),
    dpi=300
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print("\n==========================================")
print("VISUALIZATIONS CREATED SUCCESSFULLY")
print("==========================================")

print("\nSaved to:")

for filename in os.listdir(OUTPUT_DIR):
    print(
        os.path.join(
            OUTPUT_DIR,
            filename
        )
    )