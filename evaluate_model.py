import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
REPORTS_DIR = PROJECT_ROOT / "reports"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32


class MRITestDataset(Dataset):
    """Load test images and their labels from the manifest."""

    def __init__(self):
        with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
            self.rows = [
                row
                for row in csv.DictReader(file)
                if row["split"] == "test"
            ]

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
            ]
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        image_path = PROJECT_ROOT / row["path"]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        image_tensor = self.transform(image)
        label = CLASS_TO_INDEX[row["class_name"]]

        return image_tensor, label, row["path"]


def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_loader = DataLoader(
        MRITestDataset(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(input_features, len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []

    with torch.inference_mode():
        for images, labels, paths in test_loader:
            images = images.to(device)
            probabilities = torch.softmax(model(images), dim=1)
            predictions = probabilities.argmax(dim=1)

            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_paths.extend(paths)

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)
    y_prob = np.asarray(all_probabilities)

    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=range(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    print("Specificity by class:")
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = matrix[class_index, class_index]
        false_positive = matrix[:, class_index].sum() - true_positive
        false_negative = matrix[class_index, :].sum() - true_positive
        true_negative = matrix.sum() - (
            true_positive + false_positive + false_negative
        )
        specificity = true_negative / (true_negative + false_positive)
        print(f"  {class_name}: {specificity:.3f}")

    macro_auc = roc_auc_score(
        y_true,
        y_prob,
        labels=range(len(CLASS_NAMES)),
        multi_class="ovr",
        average="macro",
    )
    print(f"\nMacro one-vs-rest ROC-AUC: {macro_auc:.3f}")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=range(len(CLASS_NAMES)),
        yticks=range(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Test-set confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=30, ha="right")

    for row_index in range(len(CLASS_NAMES)):
        for column_index in range(len(CLASS_NAMES)):
            axis.text(
                column_index,
                row_index,
                matrix[row_index, column_index],
                ha="center",
                va="center",
            )

    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)

    misclassified_path = REPORTS_DIR / "misclassified_images.csv"
    with misclassified_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "true_class", "predicted_class"])

        for path, true_index, predicted_index in zip(
            all_paths,
            y_true,
            y_pred,
        ):
            if true_index != predicted_index:
                writer.writerow(
                    [
                        path,
                        CLASS_NAMES[true_index],
                        CLASS_NAMES[predicted_index],
                    ]
                )

    print(f"\nConfusion matrix saved to: {REPORTS_DIR / 'confusion_matrix.png'}")
    print(f"Misclassified images saved to: {misclassified_path}")


if __name__ == "__main__":
    main()
