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