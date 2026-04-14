import supervision as sv
import numpy as np
import matplotlib.pyplot as plt
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=os.environ["API"],
)
config = InferenceConfiguration(confidence_threshold=0.1)

#Load dataset
dataset = sv.DetectionDataset.from_yolo(
    images_directory_path=r"C:\Users\ossam\Desktop\Desktop\University\sem 8\AI\project\Object-detection-1\test\images",
    annotations_directory_path=r"C:\Users\ossam\Desktop\Desktop\University\sem 8\AI\project\Object-detection-1\test\labels",
    data_yaml_path=r"C:\Users\ossam\Desktop\Desktop\University\sem 8\AI\project\Object-detection-1\data.yaml"
)

#Run inference
all_predictions, all_targets = [], []
print("Running inference on", len(dataset), "test images...")

for i, (image_name, image, targets) in enumerate(dataset, 1):
    with client.use_configuration(config):
        result = client.infer(image, model_id="object-detection-ojs42/1")
    all_predictions.append(sv.Detections.from_inference(result))
    all_targets.append(targets)
    if i % 10 == 0:
        print(" ", i, "/", len(dataset), "done")

#Compute confusion matrix
confusion_matrix = sv.ConfusionMatrix.from_detections(
    predictions=all_predictions,
    targets=all_targets,
    classes=dataset.classes
)

#Calculate precision, recall, F1 per class
matrix = confusion_matrix.matrix
tp = np.diag(matrix)[:-1]
fp = matrix[-1, :-1]
fn = matrix[:-1, -1]
precision = np.where((tp + fp) > 0, tp / (tp + fp), 0)
recall    = np.where((tp + fn) > 0, tp / (tp + fn), 0)
f1        = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0)

#Print summary
print("\n--- EVALUATION SUMMARY ---")
for i, cls in enumerate(dataset.classes):
    print(cls)
    print("  Precision:", round(precision[i], 3))
    print("  Recall:   ", round(recall[i], 3))
    print("  F1:       ", round(f1[i], 3))

print("\nMean Precision:", round(precision.mean(), 3))
print("Mean Recall:   ", round(recall.mean(), 3))
print("Mean F1:       ", round(f1.mean(), 3))
print("mAP@0.5:        0.88") #directly from roboflow

#Plot confusion matrix
classes_with_bg = dataset.classes + ["background"]
cm = matrix.astype(int)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Campus Safety Detection — Evaluation Results", fontsize=14, fontweight="bold")

# Left: Confusion matrix heatmap
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(classes_with_bg)))
ax.set_yticks(range(len(classes_with_bg)))
ax.set_xticklabels(classes_with_bg, rotation=20, ha="right", fontsize=9)
ax.set_yticklabels(classes_with_bg, fontsize=9)
ax.set_xlabel("Predicted", fontsize=11)
ax.set_ylabel("Actual", fontsize=11)
ax.set_title("Confusion Matrix", fontsize=12)
plt.colorbar(im, ax=ax)

for i in range(len(classes_with_bg)):
    for j in range(len(classes_with_bg)):
        # determine label type
        last = len(classes_with_bg) - 1
        if i == j and i != last:
            label_type = "TP"
        elif i == last and j != last:
            label_type = "FP"
        elif j == last and i != last:
            label_type = "FN"
        elif i == last and j == last:
            label_type = "TN"
        else:
            label_type = ""

        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=12, fontweight="bold", color=color)
        ax.text(j, i + 0.35, label_type, ha="center", va="center",
                fontsize=7, color=color, alpha=0.8)

# Right: Per-class bar chart
ax2 = axes[1]
x = np.arange(len(dataset.classes))
width = 0.25
short_names = [cls[:12] + "..." if len(cls) > 12 else cls for cls in dataset.classes]

bars1 = ax2.bar(x - width, precision, width, label="Precision", color="#4CAF50")
bars2 = ax2.bar(x,          recall,    width, label="Recall",    color="#2196F3")
bars3 = ax2.bar(x + width,  f1,        width, label="F1",        color="#FF9800")

ax2.set_xticks(x)
ax2.set_xticklabels(short_names, fontsize=9)
ax2.set_ylim(0, 1.15)
ax2.set_ylabel("Score")
ax2.set_title("Per-Class Metrics", fontsize=12)
ax2.legend()

for bars in [bars1, bars2, bars3]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                str(round(bar.get_height(), 2)), ha="center", fontsize=8)

plt.tight_layout()
plt.show()
