import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "efficientnet_b0_best.pth"
OUTPUT_PATH = PROJECT_ROOT / "reports" / "gradcam_example.png"

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]


def first_test_image():
    """Get the first test image path and its true class from the manifest."""
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == "test":
                return PROJECT_ROOT / row["path"], row["class_name"]

    raise ValueError("No test image was found in the manifest.")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image_path, true_class = first_test_image()
    image = Image.open(image_path).convert("RGB").resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    preprocessing = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
        ]
    )
    input_tensor = preprocessing(image).unsqueeze(0).to(device)

    model = models.efficientnet_b0(weights=None)
    input_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(input_features, len(CLASS_NAMES))

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.inference_mode():
        probabilities = torch.softmax(model(input_tensor), dim=1)[0]
        predicted_index = int(probabilities.argmax())

    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(predicted_index)]

    # Grad-CAM needs gradients, so this is run outside inference_mode().
    with torch.enable_grad():
        cam = GradCAM(model=model, target_layers=target_layers)
        heatmap = cam(input_tensor=input_tensor, targets=targets)[0]

    overlay = show_cam_on_image(image_array, heatmap, use_rgb=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(image_array)
    axes[0].set_title(f"Original — true: {true_class}")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(
        f"Grad-CAM — predicted: {CLASS_NAMES[predicted_index]}"
    )
    axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=150)
    plt.show()

    print(f"Image: {image_path}")
    print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index].item():.3f}")
    print(f"Grad-CAM image saved to: {OUTPUT_PATH}")
    print("Grad-CAM is a visualization, not proof of medical correctness.")


if __name__ == "__main__":
    main()
