"""
End-to-end emoji generation: photo → face detection → features → classify → compose.

Usage:
    python src/pipeline.py --image data/raw/sample.jpg
    python src/pipeline.py --image data/raw/sample.jpg --output outputs/emoji.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import joblib
from PIL import Image

from landmarking import FaceProcessor
from features import compute_all_features
from skin_tone import estimate_skin_tone
from compose import EmojiComposer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURE_COLS = [
    "face_ratio", "jaw_width_ratio", "forehead_width_ratio", "chin_length_ratio",
    "eye_aspect_ratio_left", "eye_aspect_ratio_right", "eye_width_ratio", "inter_eye_distance_ratio",
    "left_brow_height_ratio", "right_brow_height_ratio", "left_brow_angle", "right_brow_angle",
    "nose_width_ratio", "nose_length_ratio", "mouth_width_ratio", "mouth_open_ratio",
]

CATEGORY_COLS = ["face_shape", "eye_type", "mouth_type", "brow_type", "nose_type"]


def generate_emoji(
    image_path: str,
    model_dir: str = str(PROJECT_ROOT / "models" / "trained"),
    assets_dir: str = str(PROJECT_ROOT / "outputs" / "openmoji_parts" / "reviewed"),
    landmarker_path: str = str(PROJECT_ROOT / "models" / "face_landmarker.task"),
) -> tuple[Image.Image, dict]:
    """
    Full pipeline: image → emoji.

    Returns (emoji_image, info_dict) where info_dict has features, predictions, skin tone.
    """
    model_dir = Path(model_dir)

    # 1. Face detection + landmarks
    with FaceProcessor(model_path=landmarker_path) as processor:
        face_result = processor.process_image(image_path=image_path)

    # 2. Feature extraction
    features = compute_all_features(face_result.landmarks_pixels)
    X = np.array([[features[col] for col in FEATURE_COLS]])

    # 3. Skin tone
    image_bgr = cv2.imread(image_path)
    skin = estimate_skin_tone(image_bgr, face_result.landmarks_pixels, include_forehead=False)

    # 4. Load classifiers and predict
    label_encoders = joblib.load(model_dir / "label_encoders.pkl")
    predictions = {}
    for category in CATEGORY_COLS:
        clf = joblib.load(model_dir / f"{category}.pkl")
        le = label_encoders[category]
        y_pred = clf.predict(X)[0]
        predictions[category] = le.inverse_transform([y_pred])[0]

    # 5. Compose emoji — use actual sampled skin color for best match
    composer = EmojiComposer(assets_dir)
    emoji_img = composer.compose(predictions, skin.mean_rgb)

    info = {
        "features": features,
        "predictions": predictions,
        "skin_tone_id": skin.matched_tone_id,
        "skin_tone_rgb": skin.mean_rgb,
    }

    return emoji_img, info


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an emoji from a face photo.")
    parser.add_argument("--image", required=True, help="Path to input face image.")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "emoji.png"))
    args = parser.parse_args()

    emoji_img, info = generate_emoji(args.image)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    emoji_img.save(args.output)

    print(f"Emoji saved: {args.output}")
    print(f"\nPredictions:")
    for k, v in info["predictions"].items():
        print(f"  {k}: {v}")
    print(f"\nSkin tone: {info['skin_tone_id']} (RGB {info['skin_tone_rgb']})")


if __name__ == "__main__":
    main()
