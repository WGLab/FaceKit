"""
Average face generation using MediaPipe landmarks.

This replaces the old two-step workflow:
  1. extract_mediapipe_landmarks.py --input imgs/ --output lm/
  2. averager.py --images imgs/ --landmarks lm/

Now done in a single call: ``average_faces(list_imgpaths("imgs/"), ...)``.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Union

import cv2
import numpy as np
import matplotlib.image as mpimg

from facekit.core.morph import aligner, blender, locator, warper
from facekit.core.morph.landmarks import MediaPipeLandmarkExtractor


def list_imgpaths(imgfolder: Union[str, Path]) -> Iterable[str]:
    """Yield image paths in a folder (.jpg, .jpeg, .png), sorted."""
    exts = (".jpg", ".jpeg", ".png")
    for fname in sorted(os.listdir(imgfolder)):
        if fname.lower().endswith(exts):
            yield os.path.join(imgfolder, fname)


def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp mask: useful when the averaged face looks soft."""
    blurred = cv2.GaussianBlur(img, (0, 0), 2.5)
    return cv2.addWeighted(img, 1.4, blurred, -0.4, 0)


def _load_image_points(path, size, extractor):
    """Load an image, extract landmarks, resize and align to ``size``."""
    img = cv2.imread(str(path))
    if img is None:
        print(f"[FaceKit] Could not read image: {path}")
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = locator.face_points(path, extractor)
    if len(points) == 0:
        print(f"[FaceKit] No face detected: {path}")
        return None, None
    return aligner.resize_align(img, points, size)


def average_faces(
    imgpaths: Iterable[Union[str, Path]],
    dest_filename: Optional[Union[str, Path]] = None,
    width: int = 500,
    height: int = 600,
    alpha: bool = False,
    blur_edges: bool = False,
    out_filename: Union[str, Path] = "result.png",
    plot: bool = False,
    extractor: Optional[MediaPipeLandmarkExtractor] = None,
    model_path: Optional[Union[str, Path]] = None,
) -> np.ndarray:
    """Generate an average face from a collection of face images.

    :param imgpaths: iterable of image file paths
    :param dest_filename: optional destination image to overlay the avg onto
    :param width: output width in pixels
    :param height: output height in pixels
    :param alpha: save with transparent background outside the face mask
    :param blur_edges: blur the edge of the face mask
    :param out_filename: output path for the generated average face
    :param plot: also display the result with matplotlib
    :param extractor: pre-initialized MediaPipe extractor (optional, for reuse)
    :param model_path: MediaPipe model path (ignored if ``extractor`` is set)
    :returns: the averaged image as an RGB (or RGBA) numpy array
    """
    size = (height, width)
    if extractor is None:
        extractor = MediaPipeLandmarkExtractor(model_path=model_path)

    images = []
    point_set = []
    for path in imgpaths:
        img, points = _load_image_points(path, size, extractor)
        if img is not None:
            images.append(img)
            point_set.append(points)

    if len(images) == 0:
        raise FileNotFoundError(
            "No valid face images found. Supported formats: .jpg, .jpeg, .png"
        )

    if dest_filename is not None:
        dest_img, dest_points = _load_image_points(dest_filename, size, extractor)
        if dest_img is None or dest_points is None:
            raise ValueError(
                f"No face detected in destination image: {dest_filename}"
            )
    else:
        dest_img = np.zeros(images[0].shape, np.uint8)
        dest_points = locator.average_points(np.array(point_set))

    num_images = len(images)
    result_images = np.zeros(images[0].shape, np.float32)
    for i in range(num_images):
        result_images += warper.warp_image(
            images[i], point_set[i], dest_points, size, np.float32
        )

    result_image = np.uint8(result_images / num_images)
    face_indexes = np.nonzero(result_image)
    dest_img[face_indexes] = result_image[face_indexes]

    mask = blender.mask_from_points(size, dest_points)
    if blur_edges:
        mask = cv2.blur(mask, (10, 10))
    if alpha:
        dest_img = np.dstack((dest_img, mask))

    out_path = Path(out_filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(str(out_path), dest_img)
    print(f"[FaceKit] Saved average face ({num_images} inputs) -> {out_path}")

    if plot:
        import matplotlib.pyplot as plt
        plt.axis("off")
        plt.imshow(dest_img)
        plt.show()

    return dest_img
