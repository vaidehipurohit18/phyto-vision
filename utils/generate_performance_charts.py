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
# FINE-TUNED MODEL METRICS
# Taken from your full evaluation output
# ============================================================

precision = np.array([
    0.9860, 0.9881, 0.9977, 0.9940, 0.9956,
    0.9952, 0.9956, 0.9262, 0.9896, 0.9569,
    1.0000, 0.9894, 0.9855, 0.9977, 1.0000,
    0.9980, 0.9891, 0.9796, 0.9695, 0.9704,
    1.0000, 0.9873, 0.9866, 0.9867, 0.9843,
    0.9977, 1.0000, 1.0000, 0.9925, 0.9496,
    0.9316, 0.9532, 0.8813, 0.9458, 0.7945,
    0.9938, 0.9862, 0.9353
])

recall = np.array([
    0.9782, 1.0000, 1.0000, 0.9841, 0.9934,
    0.9857, 0.9956, 0.9488, 1.0000, 0.9308,
    1.0000, 0.9852, 0.9896, 1.0000, 1.0000,
    0.9980, 0.9847, 1.0000, 0.9979, 0.9879,
    0.9856, 0.9608, 0.9693, 1.0000, 0.9901,
    0.9977, 1.0000, 0.9846, 0.9318, 0.8250,
    0.9417, 0.9532, 0.9197, 0.8828, 0.9475,
    0.9735, 0.9576, 0.9917
])

f1_score = np.array([
    0.9821, 0.9940, 0.9989, 0.9890, 0.9945,
    0.9905, 0.9956, 0.9373, 0.9948, 0.9437,
    1.0000, 0.9873, 0.9875, 0.9988, 1.0000,
    0.9980, 0.9869, 0.9897, 0.9835, 0.9791,
    0.9927, 0.9739, 0.9779, 0.9933, 0.9872,
    0.9977, 1.0000, 0.9923, 0.9612, 0.8829,
    0.9366, 0.9532, 0.9001, 0.9132, 0.8643,
    0.9835, 0.9717, 0.9627
])


# ============================================================
# F1 SCORE CHART
# ============================================================

plt.figure(figsize=(14, 12))

positions = np.arange(len(class_names))

plt.barh(positions, f1_score)

plt.yticks(
    positions,
    class_names,
    fontsize=8
)

plt.xlabel("F1 Score")
plt.ylabel("Disease Class")

plt.title(
    "F1 Score for Each Plant Disease Class"
)

plt.xlim(0, 1.05)

plt.tight_layout()

f1_path = os.path.join(
    RESULTS_DIR,
    "f1_scores.png"
)

plt.savefig(
    f1_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("F1 score chart saved:")
print(f1_path)


# ============================================================
# PRECISION VS RECALL
# ============================================================

plt.figure(figsize=(14, 12))

positions = np.arange(len(class_names))
width = 0.4

plt.barh(
    positions - width / 2,
    precision,
    height=width,
    label="Precision"
)

plt.barh(
    positions + width / 2,
    recall,
    height=width,
    label="Recall"
)

plt.yticks(
    positions,
    class_names,
    fontsize=8
)

plt.xlabel("Score")
plt.ylabel("Disease Class")

plt.title(
    "Precision vs Recall for Plant Disease Classification"
)

plt.xlim(0, 1.05)

plt.legend()

plt.tight_layout()

pr_path = os.path.join(
    RESULTS_DIR,
    "precision_recall.png"
)

plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Precision-Recall chart saved:")
print(pr_path)


# ============================================================
# TOP 10 LOWEST F1 CLASSES
# ============================================================

lowest_indices = np.argsort(f1_score)[:10]

lowest_names = [
    class_names[i]
    for i in lowest_indices
]

lowest_scores = f1_score[
    lowest_indices
]

plt.figure(figsize=(12, 7))

plt.barh(
    lowest_names,
    lowest_scores
)

plt.xlabel("F1 Score")

plt.title(
    "10 Most Challenging Disease Classes"
)

plt.xlim(0.5, 1.0)

plt.tight_layout()

challenging_path = os.path.join(
    RESULTS_DIR,
    "most_challenging_classes.png"
)

plt.savefig(
    challenging_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Most challenging classes chart saved:")
print(challenging_path)


# ============================================================
# SUMMARY
# ============================================================

print("\n================================================")
print("PERFORMANCE CHART GENERATION COMPLETE")
print("================================================")

print(f"\nOverall Validation Accuracy: 97.31%")
print(f"Macro F1 Score: {np.mean(f1_score) * 100:.2f}%")
print(f"Weighted-level performance is approximately 97.32%")