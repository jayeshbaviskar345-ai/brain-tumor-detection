import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "data"
TRAINING_ROOT = DATASET_ROOT / "Training"
TESTING_ROOT = DATASET_ROOT / "Testing"
REPORTS_ROOT = PROJECT_ROOT / "reports"
REPORT_PATH = REPORTS_ROOT / "duplicate_candidates.csv"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

# A Hamming distance at or below this value is only a review candidate.
# It is not proof that two files show the same original scan.
NEAR_DUPLICATE_THRESHOLD = 4


def find_image_paths(split_root: Path) -> list[Path]:
    """Find supported image files below the class folders in a split."""
    image_paths = []

    for class_folder in sorted(split_root.iterdir()):
        if not class_folder.is_dir():
            continue

        for path in sorted(class_folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

    return image_paths


def sha256_file(path: Path) -> str:
    """Return a SHA-256 fingerprint of the file's exact contents."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """Create a 64-bit perceptual hash from a readable image."""
    with Image.open(path) as image:
        grayscale_image = image.convert("L").resize((32, 32))
        pixels = np.asarray(grayscale_image, dtype=np.float32)

    dct_values = cv2.dct(pixels)
    low_frequency_values = dct_values[:8, :8]

    # Exclude the first coefficient when calculating the median.
    median_value = np.median(low_frequency_values.flatten()[1:])
    bits = low_frequency_values > median_value

    hash_value = 0
    for bit in bits.flatten():
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


def inspect_images(paths: list[Path], split_name: str) -> list[dict]:
    """Calculate file and perceptual hashes for all readable images."""
    records = []

    for index, path in enumerate(paths, start=1):
        try:
            records.append(
                {
                    "split": split_name,
                    "class_name": path.parent.name,
                    "path": path,
                    "sha256": sha256_file(path),
                    "perceptual_hash": perceptual_hash(path),
                }
            )
        except Exception as error:
            print(f"Could not inspect {path}: {error}")

        if index % 500 == 0:
            print(f"Inspected {index} {split_name} image files...")

    return records


def find_exact_cross_split_duplicates(
    training_records: list[dict],
    testing_records: list[dict],
) -> list[dict]:
    """Find files with identical contents in Training and Testing."""
    test_records_by_sha = {}

    for record in testing_records:
        test_records_by_sha.setdefault(record["sha256"], []).append(record)

    matches = []

    for training_record in training_records:
        matching_test_records = test_records_by_sha.get(
            training_record["sha256"], []
        )

        for testing_record in matching_test_records:
            matches.append(
                {
                    "match_type": "exact_file",
                    "training_record": training_record,
                    "testing_record": testing_record,
                    "hamming_distance": "",
                }
            )

    return matches


def find_near_cross_split_candidates(
    training_records: list[dict],
    testing_records: list[dict],
) -> list[dict]:
    """Find cross-split perceptual hashes within the review threshold."""
    if not training_records or not testing_records:
        return []

    training_hashes = np.asarray(
        [record["perceptual_hash"] for record in training_records],
        dtype=np.uint64,
    )
    testing_hashes = np.asarray(
        [record["perceptual_hash"] for record in testing_records],
        dtype=np.uint64,
    )

    candidates = []
    batch_size = 128

    for start in range(0, len(training_records), batch_size):
        end = min(start + batch_size, len(training_records))

        # XOR compares each training hash against each testing hash.
        xor_values = (
            training_hashes[start:end, np.newaxis]
            ^ testing_hashes[np.newaxis, :]
        )

        # Count the number of different bits in each pair of hashes.
        bit_values = np.unpackbits(xor_values.view(np.uint8), axis=-1)
        distances = bit_values.sum(axis=-1)

        matching_indexes = np.argwhere(
            distances <= NEAR_DUPLICATE_THRESHOLD
        )

        for local_training_index, testing_index in matching_indexes:
            training_index = start + int(local_training_index)
            testing_index = int(testing_index)

            training_record = training_records[training_index]
            testing_record = testing_records[testing_index]

            # Exact file matches are already reported separately.
            if training_record["sha256"] == testing_record["sha256"]:
                continue

            candidates.append(
                {
                    "match_type": "near_duplicate_candidate",
                    "training_record": training_record,
                    "testing_record": testing_record,
                    "hamming_distance": int(
                        distances[local_training_index, testing_index]
                    ),
                }
            )

        print(f"Compared Training images {start + 1}–{end} with Testing.")

    return candidates


def write_report(matches: list[dict]) -> None:
    """Save exact and near-duplicate matches to a CSV report."""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)

    columns = [
        "match_type",
        "hamming_distance",
        "training_class",
        "training_path",
        "testing_class",
        "testing_path",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for match in matches:
            training_record = match["training_record"]
            testing_record = match["testing_record"]

            writer.writerow(
                {
                    "match_type": match["match_type"],
                    "hamming_distance": match["hamming_distance"],
                    "training_class": training_record["class_name"],
                    "training_path": training_record["path"].relative_to(
                        DATASET_ROOT
                    ),
                    "testing_class": testing_record["class_name"],
                    "testing_path": testing_record["path"].relative_to(
                        DATASET_ROOT
                    ),
                }
            )


def main() -> None:
    """Check the expected folders, scan images, and save candidate matches."""
    missing_folders = [
        folder
        for folder in (TRAINING_ROOT, TESTING_ROOT)
        if not folder.is_dir()
    ]

    if missing_folders:
        print("Could not find the expected dataset folders:")
        for folder in missing_folders:
            print(f"  {folder}")
        print("Check that Training and Testing are inside the data folder.")
        return

    print("Finding image files...")
    training_paths = find_image_paths(TRAINING_ROOT)
    testing_paths = find_image_paths(TESTING_ROOT)

    print(f"Training images found: {len(training_paths)}")
    print(f"Testing images found: {len(testing_paths)}")

    print("\nCalculating hashes for Training images...")
    training_records = inspect_images(training_paths, "Training")

    print("\nCalculating hashes for Testing images...")
    testing_records = inspect_images(testing_paths, "Testing")

    print("\nChecking for exact cross-split duplicates...")
    exact_matches = find_exact_cross_split_duplicates(
        training_records,
        testing_records,
    )

    print("\nChecking for near-duplicate candidates...")
    near_matches = find_near_cross_split_candidates(
        training_records,
        testing_records,
    )

    all_matches = exact_matches + near_matches
    write_report(all_matches)

    print("\nAudit complete.")
    print(f"Exact cross-split duplicate pairs: {len(exact_matches)}")
    print(f"Near-duplicate candidates to review: {len(near_matches)}")
    print(f"CSV report saved to: {REPORT_PATH}")
    print("No dataset images were changed or deleted.")


if __name__ == "__main__":
    main()
