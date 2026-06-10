"""
Align face and image sizes.

Crop and resize face images so that all faces occupy roughly the same region
in a common output frame, which is required before averaging.
"""
from __future__ import annotations

import cv2
import numpy as np


def positive_cap(num: int):
    """Cap a number to ensure positivity.

    :returns: (overflow, capped_number), both >= 0
    """
    if num < 0:
        return 0, abs(num)
    return num, 0


def roi_coordinates(rect, size, scale):
    """Align the face rectangle to the center of the target frame.

    :param rect: (x, y, w, h) bounding rect of the face in the scaled image
    :param size: (height, width) of the target output frame
    :param scale: scaling factor applied to the source image
    :returns: (roi_x, roi_y, border_x, border_y), all >= 0
    """
    rectx, recty, rectw, recth = rect
    new_height, new_width = size
    mid_x = int((rectx + rectw / 2) * scale)
    mid_y = int((recty + recth / 2) * scale)
    roi_x = mid_x - int(new_width / 2)
    roi_y = mid_y - int(new_height / 2)

    roi_x, border_x = positive_cap(roi_x)
    roi_y, border_y = positive_cap(roi_y)
    return roi_x, roi_y, border_x, border_y


def scaling_factor(rect, size) -> float:
    """Scale factor that makes the face occupy ~80% of the target frame."""
    new_height, new_width = size
    rect_h, rect_w = rect[2:]
    height_ratio = rect_h / new_height
    width_ratio = rect_w / new_width

    if height_ratio > width_ratio:
        return (0.8 * new_height) / rect_h
    return (0.8 * new_width) / rect_w


def resize_image(img: np.ndarray, scale: float) -> np.ndarray:
    """Resize an image by the given scale factor."""
    cur_height, cur_width = img.shape[:2]
    new_h = int(scale * cur_height)
    new_w = int(scale * cur_width)
    return cv2.resize(img, (new_w, new_h))


def resize_align(img: np.ndarray, points: np.ndarray, size):
    """Resize image + points so the face is centered in the target ``size``.

    :param img: source image
    :param points: (m, 2) landmarks
    :param size: (height, width) target dimensions
    :returns: (cropped_image, adjusted_points)
    """
    new_height, new_width = size

    # Resize image based on the face bounding rectangle
    rect = cv2.boundingRect(np.array([points], np.int32))
    scale = scaling_factor(rect, size)
    img = resize_image(img, scale)

    # Center the face in the target frame
    cur_height, cur_width = img.shape[:2]
    roi_x, roi_y, border_x, border_y = roi_coordinates(rect, size, scale)
    roi_h = np.min([new_height - border_y, cur_height - roi_y])
    roi_w = np.min([new_width - border_x, cur_width - roi_x])

    # Crop to the supplied size
    crop = np.zeros((new_height, new_width, 3), img.dtype)
    crop[border_y:border_y + roi_h, border_x:border_x + roi_w] = (
        img[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    )

    # Scale and align face points to the crop
    points[:, 0] = (points[:, 0] * scale) + (border_x - roi_x)
    points[:, 1] = (points[:, 1] * scale) + (border_y - roi_y)

    return crop, points
