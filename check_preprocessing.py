import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"

IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def find_first_training_image() -> tuple[str, Path]:
    """Get the first training image path and its class from the manifest."""
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == "train":
                return row["class_name"], PROJECT_ROOT / row["path"]

    raise ValueError("No training images were found in the manifest.")


def main() -> None:
    """Resize and normalize one training image, then display it."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    class_name, image_path = find_first_training_image()

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image_file:
        original_image = image_file.convert("RGB")

    original_size = original_image.size

    preprocessing = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )

    image_tensor = preprocessing(original_image)

    print(f"Class: {class_name}")
    print(f"Image: {image_path}")
    print(f"Original size (width, height): {original_size}")
    print(f"Processed tensor shape (channels, height, width): {image_tensor.shape}")
    print(f"Normalized value range: {image_tensor.min():.3f} to {image_tensor.max():.3f}")

    # Undo normalization only for display.
    mean = torch.tensor(IMAGE_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGE_STD).view(3, 1, 1)
    display_tensor = (image_tensor * std + mean).clamp(0, 1)

    figure, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original_image)
    axes[0].set_title(f"Original: {original_size}")
    axes[0].axis("off")

    axes[1].imshow(display_tensor.permute(1, 2, 0))
    axes[1].set_title("Resized: 224 × 224")
    axes[1].axis("off")

    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
