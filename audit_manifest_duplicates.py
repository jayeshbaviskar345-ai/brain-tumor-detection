import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "split_manifest.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "duplicate_candidates_corrected.csv"

# Flag perceptual hashes that differ by at most four bits for review.
HAMMING_DISTANCE_THRESHOLD = 4

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    """Return a fingerprint of the exact file contents."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    """Create a compact visual fingerprint for an image."""
    with Image.open(path) as image:
        pixels = np.asarray(
            image.convert("L").resize((32, 32)),
            dtype=np.float32,
        )

    low_frequencies = cv2.dct(pixels)[:8, :8]
    median = np.median(low_frequencies.flatten()[1:])
    bits = low_frequencies > median

    hash_value = 0
    for bit in bits.flatten():
        hash_value = (hash_value << 1) | int(bit)

    return hash_value


def main() -> None:
    """Compare images between train, validation, and test splits."""
    if not MANIFEST_PATH.is_file():
        print(f"Manifest not found: {MANIFEST_PATH}")
        return

    records_by_split = defaultdict(list)

    with MANIFEST_PATH.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            image_path = PROJECT_ROOT / Path(row["path"])

            if not image_path.is_file():
                print(f"Missing image: {image_path}")
                continue

            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            try:
                records_by_split[row["split"]].append(
                    {
                        "split": row["split"],
                        "class_name": row["class_name"],
                        "path": image_path,
                        "sha256": sha256_file(image_path),
                        "perceptual_hash": perceptual_hash(image_path),
                    }
                )
            except Exception as error:
                print(f"Could not inspect {image_path}: {error}")

    split_pairs = [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]

    matches = []

    for left_name, right_name in split_pairs:
        left_records = records_by_split[left_name]
        right_records = records_by_split[right_name]
        print(
            f"Comparing {len(left_records)} {left_name} images "
            f"with {len(right_records)} {right_name} images..."
        )

        for left in left_records:
            for right in right_records:
                distance = (
                    left["perceptual_hash"] ^ right["perceptual_hash"]
                ).bit_count()

                if distance <= HAMMING_DISTANCE_THRESHOLD:
                    match_type = (
                        "exact_file"
                        if left["sha256"] == right["sha256"]
                        else "near_duplicate_candidate"
                    )

                    matches.append(
                        {
                            "match_type": match_type,
                            "hamming_distance": distance,
                            "left_split": left["split"],
                            "left_class": left["class_name"],
                            "left_path": left["path"].relative_to(
                                PROJECT_ROOT
                            ).as_posix(),
                            "right_split": right["split"],
                            "right_class": right["class_name"],
                            "right_path": right["path"].relative_to(
                                PROJECT_ROOT
                            ).as_posix(),
                        }
                    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "match_type",
        "hamming_distance",
        "left_split",
        "left_class",
        "left_path",
        "right_split",
        "right_class",
        "right_path",
    ]

    with REPORT_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(matches)

    exact_count = sum(match["match_type"] == "exact_file" for match in matches)
    near_count = sum(
        match["match_type"] == "near_duplicate_candidate"
        for match in matches
    )

    print("\nDuplicate scan complete.")
    print(f"Exact file pairs: {exact_count}")
    print(f"Near-duplicate candidates to review: {near_count}")
    print(f"Report saved to: {REPORT_PATH}")
    print("No dataset images were changed or deleted.")


if __name__ == "__main__":
    main()
