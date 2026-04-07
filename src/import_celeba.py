"""
Import a subset of CelebA and auto-label for emoji training.

Downloads from Kaggle (first time), then maps CelebA's 40 binary attributes
to our emoji part categories and copies images + labels into the project.

Setup (one-time):
    pip install kaggle
    # Create API token at https://www.kaggle.com/settings → "Create New Token"
    # Save kaggle.json to ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json

Usage:
    python src/import_celeba.py                  # default 500 images
    python src/import_celeba.py --count 1000     # more images
    python src/import_celeba.py --celeba_dir /path/to/existing/celeba  # skip download
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CELEBA_DIR = PROJECT_ROOT / "data" / "celeba_raw"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw"
LABELS_PATH = PROJECT_ROOT / "data" / "labels" / "manual_labels.csv"

HEADER = ["image_name", "face_shape", "eye_type", "mouth_type", "brow_type", "nose_type", "skin_tone"]


def download_celeba(celeba_dir: Path) -> None:
    """Download CelebA from Kaggle if not already present."""
    attr_file = celeba_dir / "list_attr_celeba.csv"
    img_dir = celeba_dir / "img_align_celeba" / "img_align_celeba"
    if attr_file.exists() and img_dir.exists():
        print("CelebA already downloaded.")
        return

    try:
        import kaggle  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Install kaggle first:\n"
            "  pip install kaggle\n"
            "  Then place your API token at ~/.kaggle/kaggle.json\n"
            "  (Get it from https://www.kaggle.com/settings → Create New Token)"
        )

    print("Downloading CelebA from Kaggle (this may take a while)...")
    celeba_dir.mkdir(parents=True, exist_ok=True)
    import kaggle.api
    kaggle.api.dataset_download_files(
        "jessicali9530/celeba-dataset",
        path=str(celeba_dir),
        unzip=True,
    )
    print("Download complete.")


def parse_attributes(celeba_dir: Path) -> dict[str, dict[str, int]]:
    """Parse list_attr_celeba.csv → {filename: {attr: 1/-1}}."""
    attr_file = celeba_dir / "list_attr_celeba.csv"
    if not attr_file.exists():
        raise FileNotFoundError(f"Attributes file not found: {attr_file}")

    rows = {}
    with open(attr_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.pop("image_id", None) or row.pop("", None)
            if filename is None:
                # Try first column
                first_key = list(row.keys())[0]
                filename = row.pop(first_key)
            rows[filename] = {k.strip(): int(v.strip()) for k, v in row.items()}
    return rows


def map_face_shape(attrs: dict[str, int]) -> str:
    if attrs.get("Oval_Face", -1) == 1:
        return "oval"
    if attrs.get("Chubby", -1) == 1 or attrs.get("Double_Chin", -1) == 1:
        return "round"
    if attrs.get("High_Cheekbones", -1) == 1:
        return "heart"
    return random.choice(["square", "oblong"])


def map_eye_type(attrs: dict[str, int]) -> str:
    if attrs.get("Narrow_Eyes", -1) == 1:
        return "squint"
    if attrs.get("Bags_Under_Eyes", -1) == 1:
        return "squint"
    if attrs.get("Eyeglasses", -1) == 1:
        return "round"
    return random.choice(["round", "wide"])


def map_mouth_type(attrs: dict[str, int]) -> str:
    smiling = attrs.get("Smiling", -1) == 1
    mouth_open = attrs.get("Mouth_Slightly_Open", -1) == 1
    big_lips = attrs.get("Big_Lips", -1) == 1

    if smiling and mouth_open:
        return "smile"
    if smiling and not mouth_open:
        return "smile"
    if mouth_open and not smiling:
        return "open"
    if big_lips:
        return "wide"
    return "straight"


def map_brow_type(attrs: dict[str, int]) -> str:
    if attrs.get("Arched_Eyebrows", -1) == 1:
        return "arched"
    if attrs.get("Bushy_Eyebrows", -1) == 1:
        return "flat"
    return random.choice(["flat", "raised"])


def map_nose_type(attrs: dict[str, int]) -> str:
    if attrs.get("Big_Nose", -1) == 1:
        return "wide"
    if attrs.get("Pointy_Nose", -1) == 1:
        return "pointed"
    return "medium"


def map_attributes(attrs: dict[str, int]) -> dict[str, str]:
    return {
        "face_shape": map_face_shape(attrs),
        "eye_type": map_eye_type(attrs),
        "mouth_type": map_mouth_type(attrs),
        "brow_type": map_brow_type(attrs),
        "nose_type": map_nose_type(attrs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Import CelebA subset for emoji training.")
    parser.add_argument("--count", type=int, default=500, help="Number of images to import.")
    parser.add_argument("--celeba_dir", type=str, default=str(DEFAULT_CELEBA_DIR),
                        help="Path to CelebA dataset (downloads here if missing).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    random.seed(args.seed)
    celeba_dir = Path(args.celeba_dir)

    # Step 1: Download if needed
    download_celeba(celeba_dir)

    # Step 2: Parse attributes
    print("Parsing attributes...")
    all_attrs = parse_attributes(celeba_dir)
    print(f"  Found {len(all_attrs)} images with attributes.")

    # Step 3: Find image directory
    img_dir = celeba_dir / "img_align_celeba" / "img_align_celeba"
    if not img_dir.exists():
        img_dir = celeba_dir / "img_align_celeba"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found. Checked:\n  {celeba_dir / 'img_align_celeba' / 'img_align_celeba'}\n  {celeba_dir / 'img_align_celeba'}")

    # Step 4: Filter to images that exist and have good attributes
    # Skip images with glasses, hats, heavy blur (cleaner training data)
    candidates = []
    for filename, attrs in all_attrs.items():
        if attrs.get("Eyeglasses", -1) == 1:
            continue
        if attrs.get("Wearing_Hat", -1) == 1:
            continue
        if attrs.get("Blurry", -1) == 1:
            continue
        img_path = img_dir / filename
        if img_path.exists():
            candidates.append((filename, attrs))

    print(f"  {len(candidates)} candidates after filtering (no glasses/hats/blur).")

    # Step 5: Sample subset
    count = min(args.count, len(candidates))
    selected = random.sample(candidates, count)
    print(f"  Selected {count} images.")

    # Step 6: Copy images and generate labels
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    labels = []
    for filename, attrs in selected:
        src = img_dir / filename
        dst = IMAGE_DIR / filename
        if not dst.exists():
            shutil.copy2(src, dst)

        mapped = map_attributes(attrs)
        labels.append({
            "image_name": filename,
            **mapped,
            "skin_tone": "",  # will be filled by build_dataset.py
        })

    # Write labels (overwrite existing)
    with open(LABELS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(labels)

    print(f"\nDone!")
    print(f"  Images copied to: {IMAGE_DIR}")
    print(f"  Labels saved to: {LABELS_PATH}")
    print(f"\nLabel distribution:")
    for cat in ["face_shape", "eye_type", "mouth_type", "brow_type", "nose_type"]:
        from collections import Counter
        counts = Counter(row[cat] for row in labels)
        print(f"  {cat}: {dict(sorted(counts.items()))}")

    print(f"\nNext: run 'python src/build_dataset.py' to extract features.")


if __name__ == "__main__":
    main()
