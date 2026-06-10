"""
Blending utilities for face compositing.

The original poisson_blend() relied on pyamg (not needed for average-face
generation) and has been removed. If you need Poisson blending later, the
OpenCV cv2.seamlessClone() function is a drop-in replacement that does not
require extra dependencies.
"""
from __future__ import annotations

import cv2
import numpy as np


def mask_from_points(size, points: np.ndarray) -> np.ndarray:
    """Binary mask over the convex hull of the face landmarks.

    :param size: (height, width) of the output mask
    :param points: (m, 2) landmark points
    """
    radius = 10  # erosion kernel size
    kernel = np.ones((radius, radius), np.uint8)

    mask = np.zeros(size, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
    mask = cv2.erode(mask, kernel)
    return mask


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a grayscale mask (0-255) to a 3-channel image."""
    masked_img = np.copy(img)
    for c in range(3):
        masked_img[..., c] = img[..., c] * (mask / 255)
    return masked_img


def weighted_average(img1: np.ndarray, img2: np.ndarray, percent: float = 0.5):
    """Weighted average of two images. ``percent`` is the weight on ``img1``."""
    if percent <= 0:
        return img2
    if percent >= 1:
        return img1
    return cv2.addWeighted(img1, percent, img2, 1 - percent, 0)


def alpha_feathering(src_img, dest_img, img_mask, blur_radius: int = 15):
    """Feather the mask edge and alpha-blend src onto dest."""
    mask = cv2.blur(img_mask, (blur_radius, blur_radius))
    mask = mask / 255.0

    result_img = np.empty(src_img.shape, np.uint8)
    for i in range(3):
        result_img[..., i] = src_img[..., i] * mask + dest_img[..., i] * (1 - mask)
    return result_img
