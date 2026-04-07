from __future__ import annotations
import numpy as np


def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def compute_basic_features(landmarks: np.ndarray) -> dict:
    """
    landmarks: numpy array of shape (478, 2)
    returns a dictionary of simple geometric features
    """

    forehead = landmarks[10]
    chin = landmarks[152]
    left_face = landmarks[234]
    right_face = landmarks[454]

    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    mouth_top = landmarks[13]
    mouth_bottom = landmarks[14]

    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]

    face_width = distance(left_face, right_face)
    face_height = distance(forehead, chin)
    mouth_width = distance(mouth_left, mouth_right)
    mouth_open = distance(mouth_top, mouth_bottom)
    left_eye_open = distance(left_eye_top, left_eye_bottom)
    right_eye_open = distance(right_eye_top, right_eye_bottom)

    return {
        "face_width": float(face_width),
        "face_height": float(face_height),
        "face_ratio": float(face_width / (face_height + 1e-8)),
        "mouth_width": float(mouth_width),
        "mouth_open": float(mouth_open),
        "mouth_width_ratio": float(mouth_width / (face_width + 1e-8)),
        "mouth_open_ratio": float(mouth_open / (face_height + 1e-8)),
        "left_eye_open": float(left_eye_open),
        "right_eye_open": float(right_eye_open),
    }


def compute_all_features(landmarks: np.ndarray) -> dict:
    """
    landmarks: numpy array of shape (478, 2)
    returns ~25 scale-invariant ratio features (all floats, no absolute pixel values)
    """
    eps = 1e-8

    # ── Reference measurements ────────────────────────────────────────────
    forehead = landmarks[10]
    chin = landmarks[152]
    left_face = landmarks[234]
    right_face = landmarks[454]

    face_width = distance(left_face, right_face)
    face_height = distance(forehead, chin)

    # ── Face shape ────────────────────────────────────────────────────────
    face_ratio = face_width / (face_height + eps)

    jaw_left = landmarks[172]
    jaw_right = landmarks[397]
    jaw_width_ratio = distance(jaw_left, jaw_right) / (face_width + eps)

    forehead_left = landmarks[54]
    forehead_right = landmarks[284]
    forehead_width_ratio = distance(forehead_left, forehead_right) / (face_width + eps)

    mouth_bottom = landmarks[14]
    chin_length_ratio = distance(mouth_bottom, chin) / (face_height + eps)

    # ── Eye features ──────────────────────────────────────────────────────
    # Left eye
    left_eye_inner = landmarks[133]
    left_eye_outer = landmarks[33]
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    left_eye_width = distance(left_eye_inner, left_eye_outer)
    left_eye_height = distance(left_eye_top, left_eye_bottom)
    eye_aspect_ratio_left = left_eye_height / (left_eye_width + eps)

    # Right eye
    right_eye_inner = landmarks[362]
    right_eye_outer = landmarks[263]
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    right_eye_width = distance(right_eye_inner, right_eye_outer)
    right_eye_height = distance(right_eye_top, right_eye_bottom)
    eye_aspect_ratio_right = right_eye_height / (right_eye_width + eps)

    avg_eye_width = (left_eye_width + right_eye_width) / 2
    eye_width_ratio = avg_eye_width / (face_width + eps)

    inter_eye_distance_ratio = distance(left_eye_inner, right_eye_inner) / (face_width + eps)

    # ── Eyebrow features ──────────────────────────────────────────────────
    # Left eyebrow
    left_brow_inner = landmarks[107]
    left_brow_outer = landmarks[70]
    left_brow_mid = landmarks[66]
    left_eye_center = (left_eye_top + left_eye_bottom) / 2
    left_brow_height_ratio = distance(left_brow_mid, left_eye_center) / (face_height + eps)
    left_brow_angle = float(np.arctan2(
        left_brow_outer[1] - left_brow_inner[1],
        left_brow_outer[0] - left_brow_inner[0],
    ))

    # Right eyebrow
    right_brow_inner = landmarks[336]
    right_brow_outer = landmarks[300]
    right_brow_mid = landmarks[296]
    right_eye_center = (right_eye_top + right_eye_bottom) / 2
    right_brow_height_ratio = distance(right_brow_mid, right_eye_center) / (face_height + eps)
    right_brow_angle = float(np.arctan2(
        right_brow_outer[1] - right_brow_inner[1],
        right_brow_outer[0] - right_brow_inner[0],
    ))

    # ── Nose features ─────────────────────────────────────────────────────
    nose_left = landmarks[48]
    nose_right = landmarks[278]
    nose_bridge = landmarks[6]
    nose_tip = landmarks[1]
    nose_width_ratio = distance(nose_left, nose_right) / (face_width + eps)
    nose_length_ratio = distance(nose_bridge, nose_tip) / (face_height + eps)

    # ── Mouth features ────────────────────────────────────────────────────
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    mouth_top = landmarks[13]
    mouth_width = distance(mouth_left, mouth_right)
    mouth_open = distance(mouth_top, landmarks[14])

    mouth_width_ratio = mouth_width / (face_width + eps)
    mouth_open_ratio = mouth_open / (face_height + eps)

    return {
        # Face shape (4)
        "face_ratio": float(face_ratio),
        "jaw_width_ratio": float(jaw_width_ratio),
        "forehead_width_ratio": float(forehead_width_ratio),
        "chin_length_ratio": float(chin_length_ratio),
        # Eyes (4)
        "eye_aspect_ratio_left": float(eye_aspect_ratio_left),
        "eye_aspect_ratio_right": float(eye_aspect_ratio_right),
        "eye_width_ratio": float(eye_width_ratio),
        "inter_eye_distance_ratio": float(inter_eye_distance_ratio),
        # Eyebrows (4)
        "left_brow_height_ratio": float(left_brow_height_ratio),
        "right_brow_height_ratio": float(right_brow_height_ratio),
        "left_brow_angle": left_brow_angle,
        "right_brow_angle": right_brow_angle,
        # Nose (2)
        "nose_width_ratio": float(nose_width_ratio),
        "nose_length_ratio": float(nose_length_ratio),
        # Mouth (2)
        "mouth_width_ratio": float(mouth_width_ratio),
        "mouth_open_ratio": float(mouth_open_ratio),
    }