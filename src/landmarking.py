from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from features import compute_basic_features


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


@dataclass
class FaceResult:
    image_path: str
    image_width: int
    image_height: int
    bbox_xyxy: tuple[int, int, int, int]
    landmarks_normalized: np.ndarray # shape: (N, 3), values in [0,1] for x,y
    landmarks_pixels: np.ndarray # shape: (N, 2), pixel coordinates
    face_crop: np.ndarray # cropped BGR image
    debug_image: np.ndarray # BGR image with landmarks drawn


class FaceProcessor:
    """
    loads an image, detects one face, return landmarks in normalized and pixel space
    outputs a face bounding box (cropped image) from landmarks and a debug image
    """

    def __init__(
        self,
        model_path: str,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = model_path

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self) -> "FaceProcessor":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def process_image(
        self,
        image_path: str,
        save_debug_path: Optional[str] = None,
        crop_padding_ratio: float = 0.15,
    ) -> FaceResult:
        
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mp_image = mp.Image.create_from_file(str(path))
        result = self.landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise ValueError(f"No face detected in image: {image_path}")

        face_landmarks = result.face_landmarks[0]

        rgb = mp_image.numpy_view()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        h, w = bgr.shape[:2]

        landmarks_normalized = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_landmarks],
            dtype=np.float32,
        )

        landmarks_pixels = np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks],
            dtype=np.int32,
        )

        bbox_xyxy = self._compute_bbox(landmarks_pixels, w, h, crop_padding_ratio)
        face_crop = self._crop_image(bgr, bbox_xyxy)
        debug_image = self._draw_landmarks(bgr.copy(), landmarks_pixels, bbox_xyxy)

        if save_debug_path is not None:
            save_path = Path(save_debug_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), debug_image)

        return FaceResult(
            image_path=str(path),
            image_width=w,
            image_height=h,
            bbox_xyxy=bbox_xyxy,
            landmarks_normalized=landmarks_normalized,
            landmarks_pixels=landmarks_pixels,
            face_crop=face_crop,
            debug_image=debug_image,
        )

    @staticmethod
    def _compute_bbox(
        landmarks_pixels: np.ndarray,
        image_width: int,
        image_height: int,
        crop_padding_ratio: float = 0.15,
    ) -> tuple[int, int, int, int]:
        x_coords = landmarks_pixels[:, 0]
        y_coords = landmarks_pixels[:, 1]

        x1 = int(np.min(x_coords))
        y1 = int(np.min(y_coords))
        x2 = int(np.max(x_coords))
        y2 = int(np.max(y_coords))

        box_w = x2 - x1
        box_h = y2 - y1

        pad_x = int(box_w * crop_padding_ratio)
        pad_y = int(box_h * crop_padding_ratio)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_width - 1, x2 + pad_x)
        y2 = min(image_height - 1, y2 + pad_y)

        return x1, y1, x2, y2

    @staticmethod
    def _crop_image(image_bgr: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        return image_bgr[y1:y2, x1:x2].copy()

    @staticmethod
    def _draw_landmarks(
        image_bgr: np.ndarray,
        landmarks_pixels: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
    ) -> np.ndarray:
        for (x, y) in landmarks_pixels:
            cv2.circle(image_bgr, (x, y), 1, (0, 255, 0), -1)

        x1, y1, x2, y2 = bbox_xyxy
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (255, 0, 0), 2)

        return image_bgr