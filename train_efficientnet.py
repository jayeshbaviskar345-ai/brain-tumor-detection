import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
START_MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
OUTPUT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "efficientnet_b0_finetuned_best.pth"
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

BATCH_SIZE = 32
MAX_EPOCHS = 6
EARLY_STOPPING_PATIENCE = 3
RANDOM_SEED = 42


class MRIDataset(Dataset):
    """Load MRI images and labels from the split manifest."""

    def __init__(self, split_name):
        with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
            self.rows = [
                row for row in csv.DictReader(file)
                if row["split"] == split_name
            ]

        if not self.rows:
            raise ValueError(f"No images found for split: {split_name}")

        if split_name == "train":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomRotation(degrees=10),
                    transforms.ToTensor(),
                    transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
                ]
            )
        else:
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
        return image_tensor, label


def run_epoch(model, loader, loss_function, device, optimizer=None):
    """Run one training or validation pass."""
    is_training = optimizer is not None
    model.train(is_training)

    if is_training:
        # Keep the frozen early layers from changing during fine-tuning.
        for layer in model.features[:-2]:
            layer.eval()

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            outputs = model(images)
            loss = loss_function(outputs, labels)

            if is_training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(outputs.argmax(dim=1).cpu().numpy())

    average_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    return average_loss, accuracy


def copy_model_state(model):
    """Make a safe CPU copy of model weights."""
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def main():
    """Fine-tune EfficientNet and keep the best validation checkpoint."""
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    if not START_MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Starting model not found: {START_MODEL_PATH}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(
        MRIDataset("train"),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        MRIDataset("validation"),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Build the same four-class EfficientNet model used in the first training.
    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(input_features, len(CLASS_NAMES))

    starting_checkpoint = torch.load(
        START_MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(starting_checkpoint["model_state_dict"])

    # Freeze the model, then unfreeze its last two feature blocks and classifier.
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.features[-2:].parameters():
        parameter.requires_grad = True

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    model.to(device)
    loss_function = nn.CrossEntropyLoss()

    optimizer = AdamW(
        [
            {
                "params": model.features[-2:].parameters(),
                "lr": 0.00001,
            },
            {
                "params": model.classifier.parameters(),
                "lr": 0.0001,
            },
        ],
        weight_decay=0.0001,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
    )

    # Measure the starting model on validation data for a fair comparison.
    _, baseline_accuracy = run_epoch(
        model,
        validation_loader,
        loss_function,
        device,
    )
    print(f"Starting validation accuracy: {baseline_accuracy:.3f}")

    best_validation_accuracy = baseline_accuracy
    best_model_state = copy_model_state(model)
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            loss_function,
            device,
            optimizer,
        )
        validation_loss, validation_accuracy = run_epoch(
            model,
            validation_loader,
            loss_function,
            device,
        )

        scheduler.step(validation_accuracy)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.3f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.3f}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            best_model_state = copy_model_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print("Stopped early: validation accuracy did not improve.")
            break

    OUTPUT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_model_state,
            "class_names": CLASS_NAMES,
            "best_validation_accuracy": best_validation_accuracy,
        },
        OUTPUT_MODEL_PATH,
    )

    print(f"\nStarting validation accuracy: {baseline_accuracy:.3f}")
    print(f"Best validation accuracy: {best_validation_accuracy:.3f}")
    print(f"Fine-tuned checkpoint saved to: {OUTPUT_MODEL_PATH}")

    if best_validation_accuracy <= baseline_accuracy:
        print("Fine-tuning did not improve validation accuracy.")
        print("The saved checkpoint preserves the starting model.")


if __name__ == "__main__":
    main()
