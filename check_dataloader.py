import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
BATCH_SIZE = 32


class MRIDataset(Dataset):
    """Load MRI images and labels from the split manifest."""

    def __init__(self, split_name: str):
        with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
            self.rows = [
                row
                for row in csv.DictReader(file)
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


def main():
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    train_dataset = MRIDataset("train")
    validation_dataset = MRIDataset("validation")
    test_dataset = MRIDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    images, labels = next(iter(train_loader))

    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    print(f"Classes in this batch: {[CLASS_NAMES[i] for i in labels.tolist()]}")

    # Undo normalization so the batch can be displayed normally.
    mean = torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(1, 3, 1, 1)
    display_images = (images * std + mean).clamp(0, 1)

    figure, axes = plt.subplots(2, 4, figsize=(10, 6))

    for index, axis in enumerate(axes.flat):
        axis.imshow(display_images[index].permute(1, 2, 0))
        axis.set_title(CLASS_NAMES[labels[index].item()])
        axis.axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
