"""
CLI labeling tool for face images.

Displays the face image with landmarks, then prompts for categorical labels
matching the extracted OpenMoji asset variants.

Usage:
    python src/labeling_tool.py --image_dir data/raw --model models/face_landmarker.task
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

LABELS_PATH = Path(__file__).resolve().parent.parent / "data" / "labels" / "manual_labels.csv"

CATEGORIES = {
    "face_shape": ["oval", "round", "square", "heart", "oblong"],
    "eye_type": ["round", "squint", "wink", "wide", "closed", "excited"],
    "mouth_type": ["smile", "frown", "open", "closed", "straight", "wide", "kiss", "smirk", "tongue_out", "sad"],
    "brow_type": ["flat", "raised", "arched", "angled", "sad", "mad"],
    "nose_type": ["small", "medium", "wide", "pointed"],
}

HEADER = ["image_name", "face_shape", "eye_type", "mouth_type", "brow_type", "nose_type", "skin_tone"]


def draw_landmarks(image_bgr: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    vis = image_bgr.copy()
    for x, y in landmarks.astype(int):
        cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)
    return vis


def prompt_category(name: str, options: list[str]) -> str:
    print(f"\n  {name}:")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        choice = input(f"  Choose {name} (1-{len(options)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print(f"  Invalid choice. Enter a number 1-{len(options)}.")


def get_labeled_images() -> set[str]:
    if not LABELS_PATH.exists():
        return set()
    with open(LABELS_PATH) as f:
        reader = csv.DictReader(f)
        return {row["image_name"] for row in reader}


def append_label(row: dict) -> None:
    write_header = not LABELS_PATH.exists() or LABELS_PATH.stat().st_size == 0
    with open(LABELS_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Label face images for emoji generation training.")
    parser.add_argument("--image_dir", type=str, default="data/raw", help="Directory with face images.")
    parser.add_argument("--model", type=str, default="models/face_landmarker.task", help="MediaPipe model path.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    image_files = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    already_labeled = get_labeled_images()
    remaining = [f for f in image_files if f.name not in already_labeled]

    if not remaining:
        print("All images already labeled!")
        return

    print(f"Found {len(remaining)} unlabeled images ({len(already_labeled)} already done).\n")

    with FaceProcessor(model_path=args.model) as processor:
        for i, img_path in enumerate(remaining):
            print(f"\n{'='*60}")
            print(f"Image {i+1}/{len(remaining)}: {img_path.name}")
            print(f"{'='*60}")

            try:
                result = processor.process_image(image_path=str(img_path))
            except Exception as e:
                print(f"  Skipping (no face detected): {e}")
                continue

            # Show image with landmarks
            vis = draw_landmarks(cv2.imread(str(img_path)), result.landmarks_pixels)
            cv2.imshow("Label this face (press any key)", vis)
            cv2.waitKey(1)

            # Compute features for display
            features = compute_all_features(result.landmarks_pixels)
            print("\n  Key features:")
            print(f"    face_ratio={features['face_ratio']:.3f}  "
                  f"eye_aspect_L={features['eye_aspect_ratio_left']:.3f}  "
                  f"eye_aspect_R={features['eye_aspect_ratio_right']:.3f}")
            print(f"    mouth_width={features['mouth_width_ratio']:.3f}  "
                  f"mouth_open={features['mouth_open_ratio']:.3f}  "
                  f"nose_width={features['nose_width_ratio']:.3f}")

            # Skin tone
            original_bgr = cv2.imread(str(img_path))
            skin = estimate_skin_tone(original_bgr, result.landmarks_pixels, include_forehead=False)
            print(f"    skin_tone={skin.matched_tone_id} (RGB {skin.matched_tone_rgb})")

            # Prompt for labels
            labels = {}
            for cat, options in CATEGORIES.items():
                labels[cat] = prompt_category(cat, options)

            row = {
                "image_name": img_path.name,
                **labels,
                "skin_tone": skin.matched_tone_id,
            }
            append_label(row)
            print(f"\n  Saved: {row}")

            action = input("\n  [Enter] next, [q] quit: ").strip().lower()
            if action == "q":
                break

    cv2.destroyAllWindows()
    print(f"\nLabels saved to: {LABELS_PATH}")


if __name__ == "__main__":
    main()
