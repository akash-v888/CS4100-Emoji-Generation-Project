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


def detect_hair(
    image_bgr: np.ndarray,
    landmarks: np.ndarray,
    skin_rgb: tuple[int, int, int],
) -> tuple[tuple[int, int, int] | None, str]:
    """
    Detect hair color and style from the image.
    Returns (hair_color_rgb or None, "short" | "long" | "none").
    """
    h, w = image_bgr.shape[:2]
    forehead = landmarks[10]
    left_face = landmarks[234]
    right_face = landmarks[454]
    face_width = np.linalg.norm(left_face - right_face)

    # Sample well above forehead (landmark 10 is mid-forehead, not hairline)
    cx = int(forehead[0])
    sample_bottom = max(0, int(forehead[1] - face_width * 0.20))
    sample_top = max(0, int(forehead[1] - face_width * 0.40))
    half_w = int(face_width * 0.20)
    x1, x2 = max(0, cx - half_w), min(w, cx + half_w)

    if sample_bottom <= sample_top or x2 <= x1:
        return None, "none"

    hair_region = image_bgr[sample_top:sample_bottom, x1:x2]
    if hair_region.size == 0:
        return None, "none"

    median_bgr = np.median(hair_region.reshape(-1, 3), axis=0)
    hair_rgb = (int(median_bgr[2]), int(median_bgr[1]), int(median_bgr[0]))

    skin_arr = np.array(skin_rgb, dtype=float)
    hair_arr = np.array(hair_rgb, dtype=float)
    if np.linalg.norm(skin_arr - hair_arr) < 35:
        return None, "none"

    # Check for long hair: both sides below jaw must have hair-colored pixels
    chin = landmarks[152]
    jaw_left, jaw_right = landmarks[172], landmarks[397]
    jaw_y = int(max(jaw_left[1], jaw_right[1]))
    side_bottom = min(h, int(chin[1] + face_width * 0.25))

    sides_with_hair = 0
    for sx1, sx2 in [(max(0, int(jaw_left[0] - face_width * 0.30)),
                       max(0, int(jaw_left[0] - face_width * 0.15))),
                      (min(w, int(jaw_right[0] + face_width * 0.15)),
                       min(w, int(jaw_right[0] + face_width * 0.30)))]:
        if sx2 > sx1 and side_bottom > jaw_y:
            side_region = image_bgr[jaw_y:side_bottom, sx1:sx2]
            if side_region.size > 0:
                side_mean = side_region.mean(axis=(0, 1))
                side_rgb = np.array([side_mean[2], side_mean[1], side_mean[0]])
                if np.linalg.norm(side_rgb - hair_arr) < np.linalg.norm(side_rgb - skin_arr) * 0.6:
                    sides_with_hair += 1

    return hair_rgb, "long" if sides_with_hair == 2 else "short"


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

    # Override eye_type with direct thresholds (classifier is weak due to noisy labels)
    avg_eye_aspect = (features["eye_aspect_ratio_left"] + features["eye_aspect_ratio_right"]) / 2
    if avg_eye_aspect < 0.27:
        predictions["eye_type"] = "squint"
    elif avg_eye_aspect > 0.40:
        predictions["eye_type"] = "wide"
    else:
        predictions["eye_type"] = "round"

    # 5. Detect hair
    hair_color, hair_style = detect_hair(image_bgr, face_result.landmarks_pixels, skin.mean_rgb)
    predictions["hair_style"] = hair_style
    predictions["hair_color_rgb"] = hair_color

    # 6. Lighten sampled skin color — photos have shadows that make cheeks
    # darker than the person actually appears. Boost brightness for emoji fill.
    skin_rgb = tuple(min(255, int(c * 1.25)) for c in skin.mean_rgb)

    # 7. Compose emoji
    composer = EmojiComposer(assets_dir)
    emoji_img = composer.compose(predictions, skin_rgb)

    info = {
        "features": features,
        "predictions": predictions,
        "skin_tone_id": skin.matched_tone_id,
        "skin_tone_rgb": skin_rgb,
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
