"""
MediaPipe Face Landmarker wrapper.

Visualization follows the official MediaPipe Face Landmarker Tasks API
notebook directly (mediapipe.tasks.python.vision.drawing_utils).
No mp.solutions dependency.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_MODEL_PATH = Path.home() / ".cache" / "facekit" / "face_landmarker.task"


def ensure_model(model_path: Optional[Union[str, Path]] = None) -> Path:
    model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if model_path.exists():
        return model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FaceKit] Downloading MediaPipe face landmarker model -> {model_path}")
    urllib.request.urlretrieve(DEFAULT_MODEL_URL, model_path)
    print("[FaceKit] Model downloaded")
    return model_path


class MediaPipeLandmarkExtractor:
    """478-landmark face extractor using MediaPipe Face Landmarker Tasks API."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        num_faces: int = 1,
        output_blendshapes: bool = False,
        output_transformation_matrixes: bool = False,
    ):
        model_path = ensure_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=output_transformation_matrixes,
            num_faces=num_faces,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)
        self.output_blendshapes = output_blendshapes
        self.output_transformation_matrixes = output_transformation_matrixes

    def extract(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        image = mp.Image.create_from_file(str(image_path))
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return None
        h, w = image.height, image.width
        return np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in result.face_landmarks[0]],
            dtype=np.int32,
        )

    def extract_from_array(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        if image_rgb.dtype != np.uint8:
            image_rgb = image_rgb.astype(np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError(f"Expected HxWx3 RGB image, got shape {image_rgb.shape}")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        h, w = image_rgb.shape[:2]
        return np.array(
            [[int(lm.x * w), int(lm.y * h)] for lm in result.face_landmarks[0]],
            dtype=np.int32,
        )

    def extract_full(self, image_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        image = mp.Image.create_from_file(str(image_path))
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return None

        h, w = image.height, image.width
        faces = []
        for idx, face_landmarks in enumerate(result.face_landmarks):
            face: Dict[str, Any] = {
                "landmarks_2d": [[int(lm.x * w), int(lm.y * h)] for lm in face_landmarks],
                "landmarks_3d": [
                    [float(lm.x), float(lm.y), float(lm.z)] for lm in face_landmarks
                ],
            }
            if self.output_blendshapes and result.face_blendshapes:
                face["blendshapes"] = {
                    bs.category_name: float(bs.score)
                    for bs in result.face_blendshapes[idx]
                }
            if (
                self.output_transformation_matrixes
                and result.facial_transformation_matrixes
            ):
                mat = result.facial_transformation_matrixes[idx]
                face["transformation_matrix"] = np.asarray(mat).tolist()
            faces.append(face)

        return {
            "image": {"filename": Path(image_path).name, "width": w, "height": h},
            "num_faces": len(faces),
            "faces": faces,
        }


# ---------- Visualization (direct port of the official Tasks API notebook) ---

def draw_landmarks_on_image(rgb_image: np.ndarray, detection_result) -> np.ndarray:
    """Draw the MediaPipe face mesh on an RGB image.

    Direct port of ``draw_landmarks_on_image`` from the official MediaPipe
    Face Landmarker Tasks API notebook. Uses ``mediapipe.tasks.python.vision``
    drawing utilities; no ``mp.solutions`` dependency.

    :param rgb_image: HxWx3 uint8 RGB image
    :param detection_result: the full ``FaceLandmarkerResult`` returned by
        ``detector.detect(image)`` (we need it whole, not just landmarks,
        in case MediaPipe's drawer uses other fields internally)
    :returns: annotated RGB image (new array)
    """
    from mediapipe.tasks.python.vision import drawing_utils, drawing_styles

    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]

        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_contours_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )
        drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS,
            landmark_drawing_spec=None,
            connection_drawing_spec=drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    return annotated_image


# Back-compat alias (the previous command calls ``draw_face_landmarks``).
# Accepts either a ``detection_result`` or a bare ``face_landmarks_list``
# depending on caller.
def draw_face_landmarks(
    rgb_image: np.ndarray,
    face_landmarks_or_result,
    style: str = "mesh",  # kept for API compatibility; always mesh
) -> np.ndarray:
    """Thin wrapper around ``draw_landmarks_on_image`` for backward compat."""
    # Detect whether we got the full result (has ``.face_landmarks``) or
    # just the landmarks list already.
    if hasattr(face_landmarks_or_result, "face_landmarks"):
        return draw_landmarks_on_image(rgb_image, face_landmarks_or_result)

    # Wrap a bare face_landmarks list into a minimal object that
    # ``draw_landmarks_on_image`` can consume.
    class _Wrap:
        pass
    w = _Wrap()
    w.face_landmarks = face_landmarks_or_result
    return draw_landmarks_on_image(rgb_image, w)