"""Helper functions.

Author: Travis Driver
"""

import math
from pathlib import Path
from typing import List

import cv2
import torch
import numpy as np


def extract_patches(img: np.ndarray, keypoints: List[cv2.KeyPoint], N: int = 32, mag_factor: float = 16):
    """Rectifies patches around keypoints, and returns patches tensor."""
    patches = []
    for _kp in keypoints:
        x, y = _kp.pt
        s = _kp.size
        a = 0

        s = mag_factor * s / N
        cos = math.cos(a * math.pi / 180.0)
        sin = math.sin(a * math.pi / 180.0)

        M = np.matrix(
            [
                [+s * cos, -s * sin, (-s * cos + s * sin) * N / 2.0 + x],
                [+s * sin, +s * cos, (-s * sin - s * cos) * N / 2.0 + y],
            ]
        )

        patch = cv2.warpAffine(img, M, (N, N), flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS)
        patches.append(patch)

    patches = torch.from_numpy(np.asarray(patches)).float()
    patches = torch.unsqueeze(patches, 1)
    return patches


def cv_keypoint_to_arrays(keypoints: List[cv2.KeyPoint]):
    """Decomposes OpenCV KeyPoint into arrays containing keypoint coordinates, scores, scales, and orientations."""
    if len(keypoints) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    assert isinstance(keypoints[0], cv2.KeyPoint)
    coordinates, scales, orientations, scores = [], [], [], []
    for kp in keypoints:
        coordinates.append(kp.pt)
        scales.append(kp.size)
        orientations.append(kp.angle)
        scores.append(kp.response)

    return np.array(coordinates), np.array(scales), np.array(orientations), np.array(scores)


def read_image(self, fpath) -> np.ndarray:
    """Reads grayscale image from file."""
    if not Path(fpath).exists():
        raise FileExistsError(f"No image found at {fpath}")
    img = cv2.cvtColor(cv2.imread(fpath), cv2.COLOR_BGR2GRAY)[..., np.newaxis]  # shape (h,w,1)
    return img
