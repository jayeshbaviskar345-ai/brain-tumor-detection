import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "data"
TRAINING_ROOT = DATA_ROOT / "Training"
TESTING_ROOT = DATA_ROOT / "Testing"
OUTPUT_PATH = DATA_ROOT / "processed" / "split_manifest.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20


def image_files(folder: Path) -> list[Path]:
    """Find supported image files inside a class folder."""
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def sha256_file(path: Path) -> str:
    """Create a fingerprint of the exact file contents."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def main() -> None:
    """Create a manifest that keeps exact copies in the same split."""
    if not TRAINING_ROOT.is_dir() or not TESTING_ROOT.is_dir():
        print("Could not find both data/Training and data/Testing.")
        return

    training_classes = {
        folder.name: folder
        for folder in TRAINING_ROOT.iterdir()
        if folder.is_dir()
    }
    testing_classes = {
        folder.name: folder
        for folder in TESTING_ROOT.iterdir()
        if folder.is_dir()
    }

    if set(training_classes) != set(testing_classes):
        print("Training and Testing class folders do not match.")
        print(f"Training: {sorted(training_classes)}")
        print(f"Testing: {sorted(testing_classes)}")
        return

    rows = []
    training_hashes_by_class = defaultdict(set)
    class_for_hash = {}

    for class_name, class_folder in sorted(training_classes.items()):
        paths = image_files(class_folder)

        if len(paths) < 2:
            print(f"Not enough images to split class: {class_name}")
            return

        hashes = [sha256_file(path) for path in paths]

        for image_hash in hashes:
            previous_class = class_for_hash.get(image_hash)
            if previous_class is not None and previous_class != class_name:
                print(
                    "Identical image files appear under different class labels:"
                )
                print(f"  {previous_class} and {class_name}")
                print("Please review this label conflict before splitting.")
                return
            class_for_hash[image_hash] = class_name

        training_hashes_by_class[class_name].update(hashes)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_FRACTION,
            random_state=RANDOM_SEED,
        )

        train_indexes, validation_indexes = next(
            splitter.split(paths, groups=hashes)
        )

        for index in train_indexes:
            rows.append(
                {
                    "split": "train",
                    "class_name": class_name,
                    "path": paths[index].relative_to(PROJECT_ROOT).as_posix(),
                }
            )

        for index in validation_indexes:
            rows.append(
                {
                    "split": "validation",
                    "class_name": class_name,
                    "path": paths[index].relative_to(PROJECT_ROOT).as_posix(),
                }
            )

    # Check that exact copies from Training do not also occur in Testing.
    for class_name, class_folder in sorted(testing_classes.items()):
        for path in image_files(class_folder):
            image_hash = sha256_file(path)

            if image_hash in class_for_hash:
                print("An exact duplicate exists between Training and Testing:")
                print(f"  Training class: {class_for_hash[image_hash]}")
                print(f"  Testing file: {path}")
                print("Review this before using the test set.")
                return

            rows.append(
                {
                    "split": "test",
                    "class_name": class_name,
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["split", "class_name", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter((row["split"], row["class_name"]) for row in rows)

    print(f"Updated manifest: {OUTPUT_PATH}")
    print("Image counts by split and class:")

    for split_name in ("train", "validation", "test"):
        print(f"\n{split_name}:")
        for class_name in sorted(training_classes):
            print(f"  {class_name}: {counts[(split_name, class_name)]}")

    print("\nExact duplicate images from Training stay in the same split.")
    print("Images were not moved, copied, or modified.")


if __name__ == "__main__":
    main()
