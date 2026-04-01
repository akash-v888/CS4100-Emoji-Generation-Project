from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np

# TODO: Replace these with the exact colors used by face shape assets
DEFAULT_SKIN_TONE_PALETTE = [
    {"id": "tone_1", "rgb": (255, 224, 189)},
    {"id": "tone_2", "rgb": (241, 194, 125)},
    {"id": "tone_3", "rgb": (224, 172, 105)},
    {"id": "tone_4", "rgb": (198, 134, 66)},
    {"id": "tone_5", "rgb": (141, 85, 36)},
]


@dataclass
class SkinToneResult:
    sampled_region_names: list[str]
    sampled_centers_xy: list[tuple[int, int]]
    patch_radius: int
    mean_bgr: tuple[int, int, int]
    mean_rgb: tuple[int, int, int]
    mean_lab: tuple[float, float, float]
    matched_tone_id: str
    matched_tone_rgb: tuple[int, int, int]
    matched_distance: float


def _bgr_to_lab(color_bgr: Iterable[float]) -> np.ndarray:
    arr = np.array([[list(color_bgr)]], dtype=np.uint8)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2Lab)[0, 0].astype(np.float32)
    return lab


def _rgb_to_lab(color_rgb: Iterable[int]) -> np.ndarray:
    rgb = np.array(list(color_rgb), dtype=np.uint8)
    bgr = rgb[::-1]
    return _bgr_to_lab(bgr)


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _make_circular_mask(
    image_shape: tuple[int, int, int],
    center_xy: tuple[int, int],
    radius: int,
) -> np.ndarray:
    h, w = image_shape[:2]
    cx, cy = center_xy

    y_grid, x_grid = np.ogrid[:h, :w]
    dist_sq = (x_grid - cx) ** 2 + (y_grid - cy) ** 2
    mask = dist_sq <= radius ** 2
    return mask


def _sample_region_mean_bgr(
    image_bgr: np.ndarray,
    center_xy: tuple[int, int],
    radius: int,
) -> Optional[np.ndarray]:
    h, w = image_bgr.shape[:2]
    cx, cy = center_xy

    if cx < 0 or cy < 0 or cx >= w or cy >= h:
        return None

    mask = _make_circular_mask(image_bgr.shape, center_xy, radius)
    pixels = image_bgr[mask]

    if pixels.size == 0:
        return None

    mean_bgr = pixels.mean(axis=0)
    return mean_bgr.astype(np.float32)


def estimate_skin_tone(
    image_bgr: np.ndarray,
    landmarks_pixels: np.ndarray,
    palette: Optional[list[dict]] = None,
    include_forehead: bool = False,
    patch_radius_ratio: float = 0.045,
) -> SkinToneResult:
    """
    estimates skin tone from cheek regions and matches it to the closest emoji tone
    """
    if palette is None:
        palette = DEFAULT_SKIN_TONE_PALETTE

    if landmarks_pixels.ndim != 2 or landmarks_pixels.shape[1] != 2:
        raise ValueError("landmarks_pixels must have shape (N, 2)")

    # face width from mediapipe points
    left_face = landmarks_pixels[234]
    right_face = landmarks_pixels[454]
    face_width = np.linalg.norm(left_face - right_face)

    patch_radius = max(4, int(face_width * patch_radius_ratio))

    region_points = [
        ("left_cheek", tuple(map(int, landmarks_pixels[205]))),
        ("right_cheek", tuple(map(int, landmarks_pixels[425]))),
    ]

    if include_forehead:
        region_points.append(("forehead", tuple(map(int, landmarks_pixels[9]))))

    sampled_colors = []
    sampled_names = []
    sampled_centers = []

    for region_name, center_xy in region_points:
        mean_bgr = _sample_region_mean_bgr(image_bgr, center_xy, patch_radius)
        if mean_bgr is not None:
            sampled_colors.append(mean_bgr)
            sampled_names.append(region_name)
            sampled_centers.append(center_xy)

    if not sampled_colors:
        raise ValueError("Could not sample any valid skin-tone regions.")

    mean_bgr = np.mean(np.stack(sampled_colors, axis=0), axis=0)
    mean_bgr_uint8 = tuple(int(round(x)) for x in mean_bgr.tolist())

    mean_rgb = mean_bgr_uint8[::-1]
    mean_lab = _bgr_to_lab(mean_bgr_uint8)

    best_match = None
    best_distance = float("inf")

    for tone in palette:
        tone_rgb = tuple(tone["rgb"])
        tone_lab = _rgb_to_lab(tone_rgb)
        dist = _euclidean_distance(mean_lab, tone_lab)

        if dist < best_distance:
            best_distance = dist
            best_match = tone

    if best_match is None:
        raise RuntimeError("Failed to match skin tone.")

    return SkinToneResult(
        sampled_region_names=sampled_names,
        sampled_centers_xy=sampled_centers,
        patch_radius=patch_radius,
        mean_bgr=mean_bgr_uint8,
        mean_rgb=mean_rgb,
        mean_lab=(float(mean_lab[0]), float(mean_lab[1]), float(mean_lab[2])),
        matched_tone_id=str(best_match["id"]),
        matched_tone_rgb=tuple(best_match["rgb"]),
        matched_distance=best_distance,
    )


def draw_skin_tone_debug(
    image_bgr: np.ndarray,
    result: SkinToneResult,
) -> np.ndarray:
    """
    draws sample region and color for debug
    """
    debug = image_bgr.copy()

    for region_name, center_xy in zip(result.sampled_region_names, result.sampled_centers_xy):
        cv2.circle(debug, center_xy, result.patch_radius, (0, 255, 255), 2)
        cv2.putText(
            debug,
            region_name,
            (center_xy[0] + 6, center_xy[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(debug, (20, 20), (100, 80), result.mean_bgr, -1)
    cv2.rectangle(debug, (20, 20), (100, 80), (255, 255, 255), 2)
    cv2.putText(
        debug,
        "sampled",
        (20, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    matched_bgr = result.matched_tone_rgb[::-1]
    cv2.rectangle(debug, (120, 20), (200, 80), matched_bgr, -1)
    cv2.rectangle(debug, (120, 20), (200, 80), (255, 255, 255), 2)
    cv2.putText(
        debug,
        result.matched_tone_id,
        (120, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return debug