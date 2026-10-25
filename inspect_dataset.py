from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


# The script expects Training and Testing inside this folder.
DATA_ROOT = Path(__file__).resolve().parent / "data"

# Only files with these extensions will be treated as images.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def inspect_split(split_path: Path) -> dict:
    """Count images and inspect formats, dimensions, and unreadable files."""
    class_folders = sorted(
        folder for folder in split_path.iterdir() if folder.is_dir()
    )

    if not class_folders:
        print(f"No class folders found in: {split_path}")
        return {}

    print(f"\n{'=' * 60}")
    print(f"Split: {split_path.name}")
    print(f"{'=' * 60}")

    results = {}

    for class_folder in class_folders:
        image_paths = sorted(
            path
            for path in class_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        formats = Counter()
        dimensions = Counter()
        corrupted_files = []

        for image_path in image_paths:
            try:
                # verify() checks whether the file looks like a valid image.
                with Image.open(image_path) as image:
                    image.verify()

                # Reopen the image after verify(), then load its pixel data.
                with Image.open(image_path) as image:
                    image.load()
                    formats[image.format or "Unknown"] += 1
                    dimensions[image.size] += 1

            except Exception as error:
                corrupted_files.append((image_path, str(error)))

        results[class_folder.name] = {
            "image_count": len(image_paths),
            "formats": formats,
            "dimensions": dimensions,
            "corrupted_files": corrupted_files,
        }

        print(f"\nClass: {class_folder.name}")
        print(f"Image files found: {len(image_paths)}")
        print(f"Readable images: {sum(formats.values())}")
        print(f"Image formats: {dict(formats)}")
        print(f"Most common dimensions: {dimensions.most_common(5)}")
        print(f"Unreadable/corrupted files: {len(corrupted_files)}")

        for image_path, error in corrupted_files[:5]:
            print(f"  Could not read: {image_path.name} ({error})")

    return results


def show_training_samples(training_path: Path) -> None:
    """Display one readable training image from each class folder."""
    class_folders = sorted(
        folder for folder in training_path.iterdir() if folder.is_dir()
    )

    samples = []

    for class_folder in class_folders:
        image_paths = sorted(
            path
            for path in class_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for image_path in image_paths:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    samples.append((class_folder.name, image.copy()))
                break
            except Exception:
                # If this file is unreadable, try the next one in the class.
                continue

    if not samples:
        print("\nNo readable training images were found to display.")
        return

    figure, axes = plt.subplots(
        1,
        len(samples),
        figsize=(4 * len(samples), 4),
        squeeze=False,
    )

    for axis, (class_name, image) in zip(axes[0], samples):
        axis.imshow(image.convert("RGB"))
        axis.set_title(class_name)
        axis.axis("off")

    figure.suptitle("One sample MRI image from each training class")
    figure.tight_layout()
    plt.show()


def main() -> None:
    """Check the expected folders, inspect both splits, and show samples."""
    training_path = DATA_ROOT / "Training"
    testing_path = DATA_ROOT / "Testing"

    missing_paths = [
        path for path in (training_path, testing_path) if not path.is_dir()
    ]

    if missing_paths:
        print("Could not find the expected dataset folders:")
        for path in missing_paths:
            print(f"  {path}")
        print("\nCheck that Training and Testing are inside the project's data folder.")
        return

    inspect_split(training_path)
    inspect_split(testing_path)
    show_training_samples(training_path)


if __name__ == "__main__":
    main()
