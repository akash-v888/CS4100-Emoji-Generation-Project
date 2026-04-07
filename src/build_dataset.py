"""
Batch feature extraction + label join.

Reads all labeled images, runs landmarking + feature extraction + skin tone,
joins with manual labels, and outputs a single CSV ready for training.

Usage:
    python src/build_dataset.py
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from landmarking import FaceProcessor
from features import compute_all_features
from skin_tone import estimate_skin_tone

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = {
    "image_dir": str(PROJECT_ROOT / "data" / "raw"),
    "labels_csv": str(PROJECT_ROOT / "data" / "labels" / "manual_labels.csv"),
    "model": str(PROJECT_ROOT / "models" / "face_landmarker.task"),
    "output": str(PROJECT_ROOT / "data" / "processed" / "features_labeled.csv"),
}


def load_labels(labels_csv: str) -> dict[str, dict]:
    rows = {}
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            rows[row["image_name"]] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build labeled feature dataset.")
    parser.add_argument("--image_dir", default=DEFAULTS["image_dir"])
    parser.add_argument("--labels_csv", default=DEFAULTS["labels_csv"])
    parser.add_argument("--model", default=DEFAULTS["model"])
    parser.add_argument("--output", default=DEFAULTS["output"])
    args = parser.parse_args()

    labels = load_labels(args.labels_csv)
    if not labels:
        print("No labels found. Run labeling_tool.py first.")
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get feature names from a dummy run to build header
    feature_names = list(compute_all_features(np.zeros((478, 2))).keys())
    label_cols = ["face_shape", "eye_type", "mouth_type", "brow_type", "nose_type", "skin_tone"]
    header = ["image_name"] + feature_names + label_cols

    results = []
    skipped = 0

    with FaceProcessor(model_path=args.model) as processor:
        for image_name, label_row in sorted(labels.items()):
            image_path = Path(args.image_dir) / image_name
            if not image_path.exists():
                print(f"  Missing image: {image_name}, skipping")
                skipped += 1
                continue

            try:
                result = processor.process_image(image_path=str(image_path))
            except Exception as e:
                print(f"  No face in {image_name}: {e}, skipping")
                skipped += 1
                continue

            features = compute_all_features(result.landmarks_pixels)

            original_bgr = cv2.imread(str(image_path))
            skin = estimate_skin_tone(original_bgr, result.landmarks_pixels, include_forehead=False)

            row = {"image_name": image_name}
            row.update(features)
            row.update({col: label_row.get(col, "") for col in label_cols})
            # Override skin_tone with detected value
            row["skin_tone"] = skin.matched_tone_id
            results.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(results)

    print(f"Dataset saved: {output_path}")
    print(f"  {len(results)} images processed, {skipped} skipped")


if __name__ == "__main__":
    main()
