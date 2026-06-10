"""
Facial landmark geometry utilities.

This replaces the old JSON-based locator.py. Landmarks are now obtained
directly from a MediaPipeLandmarkExtractor instance rather than loaded
from pre-computed JSON files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np


def boundary_points(points: np.ndarray) -> list:
    """Produce 2 additional boundary points at the top corners of the face.

    Helps the warping stage cover the forehead region, which is otherwise
    outside the MediaPipe landmark convex hull.

    :param points: (m, 2) array of (x, y) landmarks
    :returns: list of 2 extra corner points
    """
    x, y, w, h = cv2.boundingRect(np.array([points], np.int32))
    buffer_percent = 0.1
    spacerw = int(w * buffer_percent)
    spacerh = int(h * buffer_percent)
    return [
        [x + spacerw, y + spacerh],
        [x + w - spacerw, y + spacerh],
    ]


def face_points(
    image_path: Union[str, Path],
    extractor,
    add_boundary_points: bool = True,
) -> np.ndarray:
    """Extract facial landmarks using the provided MediaPipe extractor.

    :param image_path: path to an image file
    :param extractor: a :class:`MediaPipeLandmarkExtractor` instance
    :param add_boundary_points: whether to append 2 top-corner boundary points
    :returns: (N, 2) int32 array, or an empty array if no face was detected
    """
    points = extractor.extract(image_path)
    if points is None or len(points) == 0:
        return np.array([])
    if add_boundary_points:
        points = np.vstack([points, boundary_points(points)])
    return points


def average_points(point_set: np.ndarray) -> np.ndarray:
    """Average a set of face points from multiple images.

    :param point_set: (n, m, 2) array of face points from n images
    """
    return np.mean(point_set, 0).astype(np.int32)


def weighted_average_points(
    start_points: np.ndarray,
    end_points: np.ndarray,
    percent: float = 0.5,
) -> np.ndarray:
    """Weighted average of two sets of landmark points.

    :param percent: weight on ``start_points`` (0 = all end, 1 = all start)
    """
    if percent <= 0:
        return end_points
    if percent >= 1:
        return start_points
    return np.asarray(
        start_points * percent + end_points * (1 - percent), np.int32
    )
