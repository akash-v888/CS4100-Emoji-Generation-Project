from pathlib import Path
import cv2

from landmarking import FaceProcessor
from features import compute_basic_features
from skin_tone import estimate_skin_tone, draw_skin_tone_debug


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"
IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "sample.jpg"
DEBUG_PATH = PROJECT_ROOT / "outputs" / "debug_face.jpg"
CROP_PATH = PROJECT_ROOT / "outputs" / "face_crop.jpg"
SKIN_DEBUG_PATH = PROJECT_ROOT / "outputs" / "skin_tone_debug.jpg"


def main() -> None:
    DEBUG_PATH.parent.mkdir(exist_ok=True)

    with FaceProcessor(model_path=str(MODEL_PATH)) as processor:
        result = processor.process_image(
            image_path=str(IMAGE_PATH),
            save_debug_path=str(DEBUG_PATH),
        )

    cv2.imwrite(str(CROP_PATH), result.face_crop)

    features = compute_basic_features(result.landmarks_pixels)

    original_bgr = cv2.imread(str(IMAGE_PATH))
    skin_result = estimate_skin_tone(
        image_bgr=original_bgr,
        landmarks_pixels=result.landmarks_pixels,
        include_forehead=False,
    )

    skin_debug = draw_skin_tone_debug(original_bgr, skin_result)
    cv2.imwrite(str(SKIN_DEBUG_PATH), skin_debug)

    print("\nExtracted features:")
    for k, v in features.items():
        print(f"{k}: {v:.4f}")

    print("\nSkin tone result:")
    print("Sampled regions:", skin_result.sampled_region_names)
    print("Mean RGB:", skin_result.mean_rgb)
    print("Matched tone ID:", skin_result.matched_tone_id)
    print("Matched tone RGB:", skin_result.matched_tone_rgb)
    print("Distance:", f"{skin_result.matched_distance:.4f}")

    print("\nImage size:", result.image_width, "x", result.image_height)
    print("BBox:", result.bbox_xyxy)
    print("Landmark array shape:", result.landmarks_pixels.shape)
    print("Saved debug image to:", DEBUG_PATH)
    print("Saved face crop to:", CROP_PATH)
    print("Saved skin-tone debug image to:", SKIN_DEBUG_PATH)


if __name__ == "__main__":
    main()