"""
Geometric feature extractor for facial phenotyping.

Ported from ``mm_fusion_top10/preprocess_scripts/facekit_feature_extractor.py``.
The 9 ``_feat_*`` methods preserve the source numerics byte-for-byte; only
structural plumbing (loading, two entry points, optional frontal check) was
adjusted to fit FaceKit conventions.

Pipeline
--------
1. Detect 478 landmarks with MediaPipe Tasks API (entry point ``extract``)
   OR receive pre-computed landmarks (entry point ``extract_from_landmarks``).
2. Optionally reject non-frontal frames by yaw/pitch/roll extracted from the
   facial transformation matrix (thresholds default 15/15/10 deg).
3. Canonicalize landmarks: roll-correct, center on mid inner-canthi,
   normalize distances by BIZYG (bizygomatic face width).
4. Compute every feature as a unitless ratio, angle, or normalized polygon
   area. Bilateral features yield ``_r``, ``_l``, ``_mean``, ``_asym``.

Convention
----------
- Image coordinates: x goes right, y goes DOWN.
- "left"/"right" follow MediaPipe (subject's side).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ==========================================================================
# Landmark index constants (MediaPipe Face Mesh / Face Landmarker v2)
# ==========================================================================

# --- eyes ---
R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOT = 33, 133, 159, 145
L_EYE_OUTER, L_EYE_INNER, L_EYE_TOP, L_EYE_BOT = 263, 362, 386, 374
R_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155,
                 133, 173, 157, 158, 159, 160, 161, 246]
L_EYE_CONTOUR = [263, 249, 390, 373, 374, 380, 381, 382,
                 362, 398, 384, 385, 386, 387, 388, 466]

# --- iris (FaceMesh refine / FaceLandmarker v2 give 478 landmarks) ---
R_IRIS_CENTER = 468
L_IRIS_CENTER = 473

# --- eyebrows (each strand ordered head-to-tail along the brow) ---
R_BROW_UPPER = [70, 63, 105, 66, 107]
R_BROW_LOWER = [46, 53, 52, 65, 55]
L_BROW_UPPER = [300, 293, 334, 296, 336]
L_BROW_LOWER = [276, 283, 282, 295, 285]

# --- nose ---
NASION       = 168
NOSE_TIP     = 1
SUBNASALE    = 2
R_ALA        = 98
L_ALA        = 327
R_ALAR_CREST = 49
L_ALAR_CREST = 279
R_BRIDGE_MID = 196
L_BRIDGE_MID = 419

# --- lips ---
UPPER_LIP_TOP    = 0
UPPER_LIP_INNER  = 13
LOWER_LIP_INNER  = 14
LOWER_LIP_BOT    = 17
UPPER_LIP_PEAK_R = 37
UPPER_LIP_PEAK_L = 267
R_COMMISSURE     = 61
L_COMMISSURE     = 291

OUTER_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
             409, 270, 269, 267, 0, 37, 39, 40, 185]
INNER_LIP = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
             415, 310, 311, 312, 13, 82, 81, 80, 191]

# --- chin / jaw ---
MENTON     = 152
R_CHIN     = 176
L_CHIN     = 400
R_GONION   = 172
L_GONION   = 397

# --- face oval / forehead ---
TRICHION   = 10
GLABELLA   = 9
R_ZYGO     = 234
L_ZYGO     = 454
R_FOREHEAD_UPPER = 103
L_FOREHEAD_UPPER = 332
R_FOREHEAD_MID   = 54
L_FOREHEAD_MID   = 284

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# --- cheek / maxilla ---
R_CHEEK   = 116
L_CHEEK   = 345
R_MAXILLA = 205
L_MAXILLA = 425


# ==========================================================================
# Geometry utilities
# ==========================================================================

def _dist(p, q):
    return float(np.linalg.norm(np.asarray(p) - np.asarray(q)))


def _angle_at(p_center, p1, p2):
    """Angle (deg) at p_center in the triangle p1-p_center-p2."""
    v1 = np.asarray(p1) - np.asarray(p_center)
    v2 = np.asarray(p2) - np.asarray(p_center)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def _polygon_area_ordered(pts):
    """Shoelace area; pts must be in cyclic order."""
    pts = np.asarray(pts)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _hull_area(pts):
    """Convex-hull area via polar-angle sort + shoelace."""
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return 0.0
    c = pts.mean(axis=0)
    order = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))
    return _polygon_area_ordered(pts[order])


def _euler_from_matrix(M):
    """Return (yaw, pitch, roll) in degrees from MediaPipe's 4x4 face
    transformation matrix. Small-angle XYZ Tait-Bryan decomposition."""
    R = np.asarray(M)[:3, :3]
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = np.arctan2(R[1, 0], R[0, 0])
    else:
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw   = np.arctan2(-R[2, 0], sy)
        roll  = 0.0
    return float(np.degrees(yaw)), float(np.degrees(pitch)), float(np.degrees(roll))


# ==========================================================================
# Main class
# ==========================================================================

class GeometricFeatureExtractor:
    """MediaPipe Face Landmarker + geometric feature extraction.

    Two entry points:
      * ``extract(image_path, frontal_check=False)`` — runs MediaPipe on an
        image file.
      * ``extract_from_landmarks(landmarks_2d_px, transformation_matrix=None,
        frontal_check=False)`` — accepts already-detected landmarks in
        **pixel coordinates** (shape ``(478, 2)`` or ``(478, 3)``; only x/y
        are used). Lets callers skip MediaPipe entirely (e.g., for batch
        processing of pre-extracted JSONL landmarks).
    """

    # Canonical feature-column order. Populated lazily once via
    # ``_init_feature_columns`` (a single dummy run of ``_compute_all`` on
    # synthetic non-degenerate landmarks). Anchors the CSV schema so the
    # feature-column order is identical across runs even when every row
    # takes the early-return ``frontal_ok=False`` path.
    FEATURE_COLUMNS: Optional[list] = None

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        yaw_thresh: float = 15.0,
        pitch_thresh: float = 15.0,
        roll_thresh: float = 10.0,
    ):
        # Lazy-load MediaPipe so users of ``extract_from_landmarks`` don't
        # need a model file on disk.
        self._model_path = model_path
        self._detector = None
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.roll_thresh = roll_thresh

    @classmethod
    def _init_feature_columns(cls) -> list:
        """Populate (once) and return the canonical feature-column list.

        Runs ``_compute_all`` on a deterministic synthetic landmark set so
        that the dispatch order in ``_compute_all`` becomes the canonical
        column order. Synthetic values may be numerically meaningless; only
        the dict keys are used.

        Plugin columns registered via ``facekit.api.register_feature`` are
        NOT discovered here; they must be appended via
        ``extend_feature_columns`` BEFORE the first call into the extractor.
        """
        if cls.FEATURE_COLUMNS is not None:
            return cls.FEATURE_COLUMNS
        rng = np.random.default_rng(0)
        # 1000-pixel uniform spread is non-degenerate enough that all
        # divisions, polyfits, and convex-hull steps produce finite values.
        synthetic = rng.uniform(0.0, 1000.0, size=(478, 2))
        # Use a bare instance (no detector loaded) — _canonicalize and
        # _compute_all are pure numerics. Bypass plugin dispatch here so
        # _init_feature_columns stays a pure base-125 probe even if
        # plugins are already registered (extend_feature_columns appends
        # plugin cols after this base list is built).
        probe = cls.__new__(cls)
        lm_c, scales = probe._canonicalize(synthetic)
        # _compute_all may dispatch plugins; we want the base-125 keys only.
        feats = {}
        feats.update(probe._feat_eyebrow(lm_c, scales))
        feats.update(probe._feat_eye(lm_c, scales))
        feats.update(probe._feat_nose(lm_c, scales))
        feats.update(probe._feat_philtrum(lm_c, scales))
        feats.update(probe._feat_mouth_lip(lm_c, scales))
        feats.update(probe._feat_chin_jaw(lm_c, scales))
        feats.update(probe._feat_forehead(lm_c, scales))
        feats.update(probe._feat_face_global(lm_c, scales))
        feats.update(probe._feat_midface_cheek(lm_c, scales))
        cls.FEATURE_COLUMNS = list(feats.keys())
        return cls.FEATURE_COLUMNS

    @classmethod
    def extend_feature_columns(cls, new_cols: List[str]) -> None:
        """Append plugin csv_columns to ``FEATURE_COLUMNS``.

        Must be called AFTER plugin registration and BEFORE the first
        extraction. If the column list is already locked with a
        DIFFERENT extension, raises ``RuntimeError``. Idempotent when
        called repeatedly with the same suffix.
        """
        if not new_cols:
            # Lock the base column list if no plugins; cheap no-op if
            # already locked.
            if cls.FEATURE_COLUMNS is None:
                cls._init_feature_columns()
            return
        if cls.FEATURE_COLUMNS is None:
            cls._init_feature_columns()
        current_tail = cls.FEATURE_COLUMNS[len(cls.FEATURE_COLUMNS) - len(new_cols):]
        if current_tail == list(new_cols):
            return  # idempotent
        # Detect a partial / different extension.
        already = set(cls.FEATURE_COLUMNS)
        overlap = [c for c in new_cols if c in already]
        new_after_lock = [c for c in new_cols if c not in already]
        if overlap:
            raise RuntimeError(
                f"extend_feature_columns: FEATURE_COLUMNS was already locked "
                f"with a different plugin set. Already-locked plugin columns: "
                f"{overlap}; columns trying to be added now: {new_after_lock}. "
                f"This typically means a plugin was registered AFTER the "
                f"column schema was locked. Register all plugin features "
                f"BEFORE the first call to extract or to extend_feature_columns."
            )
        cls.FEATURE_COLUMNS = list(cls.FEATURE_COLUMNS) + list(new_cols)

    def _ensure_detector(self):
        if self._detector is not None:
            return
        from facekit.core.morph.landmarks import ensure_model
        import mediapipe as mp  # noqa: F401  (used implicitly by tasks API)
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = ensure_model(self._model_path)
        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=True,
            num_faces=1,
        )
        self._detector = mp_vision.FaceLandmarker.create_from_options(options)

    # --------------------------------------------------------------
    # Public entry point: image path
    # --------------------------------------------------------------
    def extract(
        self,
        image_path: Union[str, Path],
        frontal_check: bool = False,
    ) -> Optional[dict]:
        """Run detection + feature extraction on one image.

        Returns a flat dict of floats. If no face is detected, returns
        ``None``. If ``frontal_check`` is on and the pose is out of range,
        returns a minimal dict with pose + ``frontal_ok=False``.
        """
        self._ensure_detector()
        import mediapipe as mp

        image = mp.Image.create_from_file(str(image_path))
        result = self._detector.detect(image)
        if not result.face_landmarks:
            return None

        rgb = image.numpy_view()
        h, w = rgb.shape[:2]
        lm_px = np.array(
            [[p.x * w, p.y * h] for p in result.face_landmarks[0]],
            dtype=np.float64,
        )
        matrix = np.asarray(result.facial_transformation_matrixes[0])
        return self._compute(lm_px, matrix, frontal_check=frontal_check)

    # --------------------------------------------------------------
    # Public entry point: pre-computed landmarks
    # --------------------------------------------------------------
    def extract_from_landmarks(
        self,
        landmarks_2d_px: np.ndarray,
        transformation_matrix: Optional[np.ndarray] = None,
        frontal_check: bool = False,
    ) -> Optional[dict]:
        """Compute features from pre-detected landmarks in pixel coordinates.

        :param landmarks_2d_px: array of shape ``(478, 2)`` or ``(478, 3)``;
            x/y must be in pixel units (image-frame coordinates, NOT
            MediaPipe normalized [0,1] coords). If you have normalized
            coords, multiply by ``image_size`` before calling.
        :param transformation_matrix: optional 4x4 facial transformation
            matrix from MediaPipe. Required if ``frontal_check=True``.
        :param frontal_check: if True, returns a minimal dict with
            ``frontal_ok=False`` when pose is out of range. Requires the
            transformation matrix.
        """
        lm_arr = np.asarray(landmarks_2d_px, dtype=np.float64)
        if lm_arr.ndim != 2 or lm_arr.shape[0] < 478 or lm_arr.shape[1] < 2:
            raise ValueError(
                f"landmarks_2d_px must have shape (>=478, >=2), got {lm_arr.shape}"
            )
        # Drop any z component; the geometry math is 2D only.
        lm_px = lm_arr[:, :2]

        if frontal_check and transformation_matrix is None:
            raise ValueError(
                "frontal_check=True requires transformation_matrix; pass the "
                "MediaPipe 4x4 facial transformation matrix or disable the check."
            )

        matrix = (
            np.asarray(transformation_matrix)
            if transformation_matrix is not None
            else None
        )
        return self._compute(lm_px, matrix, frontal_check=frontal_check)

    # --------------------------------------------------------------
    # Internal: shared compute path
    # --------------------------------------------------------------
    def _compute(
        self,
        lm_px: np.ndarray,
        matrix: Optional[np.ndarray],
        frontal_check: bool,
    ) -> dict:
        feats: dict = {}

        if matrix is not None:
            yaw, pitch, roll = _euler_from_matrix(matrix)
            feats.update(pose_yaw=yaw, pose_pitch=pitch, pose_roll=roll)
        else:
            # matrix unavailable -> pose and frontal_ok both unknown
            feats.update(
                pose_yaw=float("nan"),
                pose_pitch=float("nan"),
                pose_roll=float("nan"),
            )

        if frontal_check:
            yaw = feats["pose_yaw"]
            pitch = feats["pose_pitch"]
            roll = feats["pose_roll"]
            if (abs(yaw) > self.yaw_thresh or
                    abs(pitch) > self.pitch_thresh or
                    abs(roll) > self.roll_thresh):
                feats["frontal_ok"] = False
                return feats
            feats["frontal_ok"] = True
        elif matrix is not None:
            feats["frontal_ok"] = True
        else:
            # No matrix and check disabled: tri-state "unknown".
            feats["frontal_ok"] = float("nan")

        lm_c, scales = self._canonicalize(lm_px)
        feats.update(self._compute_all(lm_c, scales))
        return feats

    # --------------------------------------------------------------
    # Canonicalization: roll-correct, center, build scales
    # --------------------------------------------------------------
    def _canonicalize(self, lm):
        r_inner, l_inner = lm[R_EYE_INNER], lm[L_EYE_INNER]
        theta = np.arctan2(l_inner[1] - r_inner[1], l_inner[0] - r_inner[0])
        c, s = np.cos(-theta), np.sin(-theta)
        R_mat = np.array([[c, -s], [s, c]])
        mid = 0.5 * (r_inner + l_inner)
        lm_c = (lm - mid) @ R_mat.T

        scales = dict(
            bizyg     = _dist(lm_c[R_ZYGO], lm_c[L_ZYGO]),
            face_h    = abs(lm_c[TRICHION, 1] - lm_c[MENTON, 1]),
            midface_h = abs(lm_c[NASION, 1] - lm_c[SUBNASALE, 1]),
            mouth_w   = _dist(lm_c[R_COMMISSURE], lm_c[L_COMMISSURE]),
        )
        return lm_c, scales

    # --------------------------------------------------------------
    # Feature dispatch
    # --------------------------------------------------------------
    def _compute_all(self, lm, s):
        f = {}
        f.update(self._feat_eyebrow(lm, s))
        f.update(self._feat_eye(lm, s))
        f.update(self._feat_nose(lm, s))
        f.update(self._feat_philtrum(lm, s))
        f.update(self._feat_mouth_lip(lm, s))
        f.update(self._feat_chin_jaw(lm, s))
        f.update(self._feat_forehead(lm, s))
        f.update(self._feat_face_global(lm, s))
        f.update(self._feat_midface_cheek(lm, s))
        # Plugin features (registered via facekit.api.register_feature).
        # Local import to avoid circular dep: api.py imports extractor for
        # conflict-checking against base columns.
        from facekit.api import get_registered
        for spec in get_registered().values():
            try:
                f.update(spec.func(lm, s))
            except Exception as e:
                logger.warning("plugin %s raised: %s", spec.group, e)
                for col in spec.csv_columns:
                    f.setdefault(col, float("nan"))
        return f

    # --------------------------------------------------------------
    # 1. Eyebrow features
    # --------------------------------------------------------------
    def _feat_eyebrow(self, lm, s):
        f, bz = {}, s['bizyg']

        f['eb_thickness_r'] = _hull_area(lm[R_BROW_UPPER + R_BROW_LOWER]) / (bz ** 2)
        f['eb_thickness_l'] = _hull_area(lm[L_BROW_UPPER + L_BROW_LOWER]) / (bz ** 2)
        f['eb_thickness_mean'] = 0.5 * (f['eb_thickness_r'] + f['eb_thickness_l'])
        f['eb_thickness_asym'] = (abs(f['eb_thickness_r'] - f['eb_thickness_l'])
                                  / (f['eb_thickness_mean'] + 1e-12))

        r_medial = max(lm[R_BROW_UPPER + R_BROW_LOWER], key=lambda p: p[0])
        l_medial = min(lm[L_BROW_UPPER + L_BROW_LOWER], key=lambda p: p[0])
        f['eb_inter_distance'] = _dist(r_medial, l_medial) / bz

        for side, ups in [('r', R_BROW_UPPER), ('l', L_BROW_UPPER)]:
            pts = lm[ups]
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            peak_y = pts[:, 1].min()
            end_y  = 0.5 * (pts[0, 1] + pts[-1, 1])
            brow_len = _dist(pts[0], pts[-1])
            f[f'eb_arch_{side}'] = (end_y - peak_y) / (brow_len + 1e-12)
        f['eb_arch_mean'] = 0.5 * (f['eb_arch_r'] + f['eb_arch_l'])
        f['eb_arch_asym'] = (abs(f['eb_arch_r'] - f['eb_arch_l'])
                             / (abs(f['eb_arch_mean']) + 1e-12))

        mfh = s['midface_h']
        for side, ups, top in [('r', R_BROW_UPPER, R_EYE_TOP),
                               ('l', L_BROW_UPPER, L_EYE_TOP)]:
            brow_c = lm[ups].mean(axis=0)
            f[f'eb_position_{side}'] = abs(brow_c[1] - lm[top][1]) / (mfh + 1e-12)
        f['eb_position_mean'] = 0.5 * (f['eb_position_r'] + f['eb_position_l'])
        f['eb_position_asym'] = (abs(f['eb_position_r'] - f['eb_position_l'])
                                 / (f['eb_position_mean'] + 1e-12))

        for side, ups in [('r', R_BROW_UPPER), ('l', L_BROW_UPPER)]:
            pts = lm[ups]
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            if side == 'r':
                seg = pts[-3:]
            else:
                seg = pts[:3]
            dx = seg[-1, 0] - seg[0, 0]
            dy = seg[-1, 1] - seg[0, 1]
            slope = dy / (dx + 1e-12)
            flare = -slope if side == 'r' else slope
            f[f'eb_medial_flare_{side}'] = flare

        for side, ups, lows in [('r', R_BROW_UPPER, R_BROW_LOWER),
                                ('l', L_BROW_UPPER, L_BROW_LOWER)]:
            pts = lm[ups + lows]
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n = len(pts) // 3
            seg = pts[:n] if side == 'r' else pts[-n:]
            f[f'eb_lateral_thickness_{side}'] = _hull_area(seg) / (bz ** 2)

        f['eb_medial_flare_mean'] = 0.5 * (f['eb_medial_flare_r'] + f['eb_medial_flare_l'])
        f['eb_lateral_thickness_mean'] = 0.5 * (f['eb_lateral_thickness_r']
                                                 + f['eb_lateral_thickness_l'])
        f['eb_lateral_thickness_asym'] = (abs(f['eb_lateral_thickness_r']
                                              - f['eb_lateral_thickness_l'])
                                          / (f['eb_lateral_thickness_mean'] + 1e-12))
        return f

    # --------------------------------------------------------------
    # 2. Eye / periocular features
    # --------------------------------------------------------------
    def _feat_eye(self, lm, s):
        f, bz = {}, s['bizyg']

        for side, outer, inner, top, bot in [
            ('r', R_EYE_OUTER, R_EYE_INNER, R_EYE_TOP, R_EYE_BOT),
            ('l', L_EYE_OUTER, L_EYE_INNER, L_EYE_TOP, L_EYE_BOT)]:
            f[f'eye_fissure_length_{side}'] = _dist(lm[outer], lm[inner]) / bz
            f[f'eye_fissure_height_{side}'] = _dist(lm[top],   lm[bot])   / bz
        f['eye_fissure_length_mean'] = 0.5 * (f['eye_fissure_length_r'] + f['eye_fissure_length_l'])
        f['eye_fissure_height_mean'] = 0.5 * (f['eye_fissure_height_r'] + f['eye_fissure_height_l'])
        f['eye_fissure_length_asym'] = (abs(f['eye_fissure_length_r'] - f['eye_fissure_length_l'])
                                        / (f['eye_fissure_length_mean'] + 1e-12))
        f['eye_fissure_height_asym'] = (abs(f['eye_fissure_height_r'] - f['eye_fissure_height_l'])
                                        / (f['eye_fissure_height_mean'] + 1e-12))

        for side, outer, inner in [('r', R_EYE_OUTER, R_EYE_INNER),
                                   ('l', L_EYE_OUTER, L_EYE_INNER)]:
            dx = lm[outer, 0] - lm[inner, 0]
            dy = lm[outer, 1] - lm[inner, 1]
            f[f'eye_fissure_slant_{side}'] = float(np.degrees(np.arctan2(-dy, abs(dx))))
        f['eye_fissure_slant_mean'] = 0.5 * (f['eye_fissure_slant_r'] + f['eye_fissure_slant_l'])
        f['eye_fissure_slant_asym'] = abs(f['eye_fissure_slant_r'] - f['eye_fissure_slant_l'])

        f['eye_fissure_aspect_r'] = f['eye_fissure_height_r'] / (f['eye_fissure_length_r'] + 1e-12)
        f['eye_fissure_aspect_l'] = f['eye_fissure_height_l'] / (f['eye_fissure_length_l'] + 1e-12)
        f['eye_fissure_aspect_mean'] = 0.5 * (f['eye_fissure_aspect_r'] + f['eye_fissure_aspect_l'])

        f['inter_canthal_distance']   = _dist(lm[R_EYE_INNER], lm[L_EYE_INNER])      / bz
        f['inter_pupillary_distance'] = _dist(lm[R_IRIS_CENTER], lm[L_IRIS_CENTER])  / bz
        f['outer_canthal_distance']   = _dist(lm[R_EYE_OUTER], lm[L_EYE_OUTER])      / bz
        f['canthal_to_pupillary_ratio'] = (f['inter_canthal_distance']
                                           / (f['inter_pupillary_distance'] + 1e-12))

        for side, inner, up_spoke, lo_spoke in [
            ('r', R_EYE_INNER, 173, 155),
            ('l', L_EYE_INNER, 398, 382)]:
            f[f'eye_epicanthus_angle_{side}'] = _angle_at(lm[inner], lm[up_spoke], lm[lo_spoke])
        f['eye_epicanthus_angle_mean'] = 0.5 * (f['eye_epicanthus_angle_r']
                                                + f['eye_epicanthus_angle_l'])
        f['eye_epicanthus_angle_asym'] = (abs(f['eye_epicanthus_angle_r']
                                              - f['eye_epicanthus_angle_l'])
                                          / (f['eye_epicanthus_angle_mean'] + 1e-12))

        for side, top, iris, bot in [
            ('r', R_EYE_TOP, R_IRIS_CENTER, R_EYE_BOT),
            ('l', L_EYE_TOP, L_IRIS_CENTER, L_EYE_BOT)]:
            fiss_h = _dist(lm[top], lm[bot])
            f[f'eye_upper_lid_to_iris_{side}'] = (lm[iris, 1] - lm[top, 1]) / (fiss_h + 1e-12)
            f[f'eye_lower_lid_to_iris_{side}'] = (lm[bot, 1] - lm[iris, 1]) / (fiss_h + 1e-12)

        f['eye_upper_lid_to_iris_mean'] = 0.5 * (f['eye_upper_lid_to_iris_r']
                                                  + f['eye_upper_lid_to_iris_l'])
        f['eye_upper_lid_to_iris_asym'] = (abs(f['eye_upper_lid_to_iris_r']
                                               - f['eye_upper_lid_to_iris_l'])
                                           / (f['eye_upper_lid_to_iris_mean'] + 1e-12))
        f['eye_lower_lid_to_iris_mean'] = 0.5 * (f['eye_lower_lid_to_iris_r']
                                                  + f['eye_lower_lid_to_iris_l'])
        f['eye_lower_lid_to_iris_asym'] = (abs(f['eye_lower_lid_to_iris_r']
                                               - f['eye_lower_lid_to_iris_l'])
                                           / (f['eye_lower_lid_to_iris_mean'] + 1e-12))

        f['eye_area_r'] = _hull_area(lm[R_EYE_CONTOUR]) / (bz ** 2)
        f['eye_area_l'] = _hull_area(lm[L_EYE_CONTOUR]) / (bz ** 2)
        f['eye_area_mean'] = 0.5 * (f['eye_area_r'] + f['eye_area_l'])
        f['eye_area_asym'] = (abs(f['eye_area_r'] - f['eye_area_l'])
                              / (f['eye_area_mean'] + 1e-12))

        for side in ('r', 'l'):
            fl = f[f'eye_fissure_length_{side}'] * bz
            f[f'eye_fissure_fill_{side}'] = (f[f'eye_area_{side}'] * bz ** 2
                                             / (fl ** 2 + 1e-12))
        f['eye_fissure_fill_mean'] = 0.5 * (f['eye_fissure_fill_r'] + f['eye_fissure_fill_l'])

        for side, outer, inner, iris in [
            ('r', R_EYE_OUTER, R_EYE_INNER, R_IRIS_CENTER),
            ('l', L_EYE_OUTER, L_EYE_INNER, L_IRIS_CENTER)]:
            fissure_center = 0.5 * (lm[outer] + lm[inner])
            fissure_len = _dist(lm[outer], lm[inner])
            f[f'iris_offset_x_{side}'] = (lm[iris, 0] - fissure_center[0]) / (fissure_len + 1e-12)
            f[f'iris_offset_y_{side}'] = (lm[iris, 1] - fissure_center[1]) / (fissure_len + 1e-12)
        f['gaze_asym_x'] = f['iris_offset_x_r'] - f['iris_offset_x_l']
        f['gaze_asym_y'] = f['iris_offset_y_r'] - f['iris_offset_y_l']
        f['gaze_asym_norm'] = float(np.hypot(f['gaze_asym_x'], f['gaze_asym_y']))
        return f

    # --------------------------------------------------------------
    # 3. Nose features
    # --------------------------------------------------------------
    def _feat_nose(self, lm, s):
        f, bz, fh, mfh = {}, s['bizyg'], s['face_h'], s['midface_h']

        f['nose_length'] = _dist(lm[NASION], lm[SUBNASALE]) / fh
        f['nose_bridge_length'] = _dist(lm[NASION], lm[NOSE_TIP]) / (mfh + 1e-12)
        f['nose_bridge_width'] = _dist(lm[R_BRIDGE_MID], lm[L_BRIDGE_MID]) / bz
        f['nose_tip_width'] = _dist(lm[R_ALAR_CREST], lm[L_ALAR_CREST]) / bz
        f['nose_base_width'] = _dist(lm[R_ALA], lm[L_ALA]) / bz

        f['nose_tip_to_bridge_ratio'] = f['nose_tip_width'] / (f['nose_bridge_width'] + 1e-12)
        mid_y = lm[NOSE_TIP, 1]
        lat_y = 0.5 * (lm[45, 1] + lm[275, 1])
        f['nose_tip_midline_dent'] = (mid_y - lat_y) / bz

        f['nose_ala_height_r'] = abs(lm[R_ALAR_CREST, 1] - lm[R_ALA, 1]) / bz
        f['nose_ala_height_l'] = abs(lm[L_ALAR_CREST, 1] - lm[L_ALA, 1]) / bz
        f['nose_ala_height_mean'] = 0.5 * (f['nose_ala_height_r'] + f['nose_ala_height_l'])
        f['nose_ala_height_asym'] = (abs(f['nose_ala_height_r'] - f['nose_ala_height_l'])
                                     / (f['nose_ala_height_mean'] + 1e-12))

        nostril_poly = [R_ALA, R_ALAR_CREST, NOSE_TIP, SUBNASALE,
                        L_ALAR_CREST, L_ALA]
        f['nostril_region_area'] = _hull_area(lm[nostril_poly]) / (bz ** 2)

        f['nasolabial_angle'] = _angle_at(lm[SUBNASALE], lm[NOSE_TIP], lm[UPPER_LIP_TOP])
        f['nose_tip_elevation'] = (lm[SUBNASALE, 1] - lm[NOSE_TIP, 1]) / bz
        f['ala_flare_angle'] = _angle_at(lm[SUBNASALE], lm[R_ALA], lm[L_ALA])

        alar_base_y = 0.5 * (lm[R_ALA, 1] + lm[L_ALA, 1])
        f['columella_length']    = abs(lm[SUBNASALE, 1] - lm[NOSE_TIP, 1]) / (mfh + 1e-12)
        f['columella_hang_below_ala'] = (lm[SUBNASALE, 1] - alar_base_y) / (mfh + 1e-12)

        nose_poly = [NASION, R_ALA, NOSE_TIP, L_ALA]
        face_area = _hull_area(lm[FACE_OVAL])
        f['nose_to_face_area_ratio'] = _hull_area(lm[nose_poly]) / (face_area + 1e-12)
        return f

    # --------------------------------------------------------------
    # 4. Philtrum features
    # --------------------------------------------------------------
    def _feat_philtrum(self, lm, s):
        f, bz, mfh = {}, s['bizyg'], s['midface_h']
        f['philtrum_length'] = _dist(lm[SUBNASALE], lm[UPPER_LIP_TOP]) / mfh
        f['philtrum_width']  = _dist(lm[UPPER_LIP_PEAK_R], lm[UPPER_LIP_PEAK_L]) / (s['mouth_w'] + 1e-12)
        return f

    # --------------------------------------------------------------
    # 5. Mouth / lip features
    # --------------------------------------------------------------
    def _feat_mouth_lip(self, lm, s):
        f, bz, mw = {}, s['bizyg'], s['mouth_w']

        f['mouth_width'] = mw / bz
        f['mouth_opening'] = _dist(lm[UPPER_LIP_INNER], lm[LOWER_LIP_INNER]) / (mw + 1e-12)

        upper_vh = abs(lm[UPPER_LIP_INNER, 1] - lm[UPPER_LIP_TOP, 1])
        lower_vh = abs(lm[LOWER_LIP_BOT, 1]  - lm[LOWER_LIP_INNER, 1])
        f['upper_vermilion_height']  = upper_vh / (mw + 1e-12)
        f['lower_vermilion_height']  = lower_vh / (mw + 1e-12)
        f['vermilion_total']         = (upper_vh + lower_vh) / (mw + 1e-12)

        peaks_y = 0.5 * (lm[UPPER_LIP_PEAK_R, 1] + lm[UPPER_LIP_PEAK_L, 1])
        f['cupid_bow_drop'] = (lm[UPPER_LIP_TOP, 1] - peaks_y) / (mw + 1e-12)

        commissure_y = 0.5 * (lm[R_COMMISSURE, 1] + lm[L_COMMISSURE, 1])
        f['mouth_corner_drop'] = (commissure_y - lm[UPPER_LIP_TOP, 1]) / (mw + 1e-12)

        f['mouth_triangularity'] = (commissure_y - peaks_y) / (mw + 1e-12)
        peak_span = _dist(lm[UPPER_LIP_PEAK_R], lm[UPPER_LIP_PEAK_L])
        f['mouth_tenting'] = ((commissure_y - peaks_y)
                              / (peak_span + 1e-12))
        upper_outer = [185, 40, 39, 37, 0, 267, 269, 270, 409]
        pts = lm[upper_outer]
        x, y = pts[:, 0], pts[:, 1]
        coeffs = np.polyfit(x, y, 2)
        yhat = np.polyval(coeffs, x)
        f['mouth_upper_curvature_c2'] = float(coeffs[0]) * bz
        f['mouth_upper_fit_residual'] = float(np.std(y - yhat)) / (mw + 1e-12)

        upper_outer_idx = [185, 40, 39, 37, 0, 267, 269, 270, 409]
        upper_inner_idx = [191, 80, 81, 82, 13, 312, 311, 310, 415]
        f['upper_lip_eversion'] = (float(lm[upper_inner_idx, 1].mean()
                                         - lm[upper_outer_idx, 1].mean())
                                   / (mw + 1e-12))
        lower_outer_idx = [146, 91, 181, 84, 17, 314, 405, 321, 375]
        lower_inner_idx = [95, 88, 178, 87, 14, 317, 402, 318, 324]
        f['lower_lip_eversion'] = (float(lm[lower_outer_idx, 1].mean()
                                         - lm[lower_inner_idx, 1].mean())
                                   / (mw + 1e-12))
        outer_area = _hull_area(lm[OUTER_LIP])
        inner_area = _hull_area(lm[INNER_LIP])
        f['lip_outer_to_inner_area'] = outer_area / (inner_area + 1e-12)

        midline_x = [lm[UPPER_LIP_TOP, 0], lm[UPPER_LIP_INNER, 0],
                     lm[LOWER_LIP_INNER, 0], lm[LOWER_LIP_BOT, 0]]
        f['lip_midline_x_std'] = float(np.std(midline_x)) / bz
        f['cupid_bow_peak_asym'] = (abs(lm[UPPER_LIP_PEAK_R, 1] - lm[UPPER_LIP_PEAK_L, 1])
                                    / (mw + 1e-12))
        return f

    # --------------------------------------------------------------
    # 6. Chin / jaw features
    # --------------------------------------------------------------
    def _feat_chin_jaw(self, lm, s):
        f, bz, fh = {}, s['bizyg'], s['face_h']

        f['chin_height'] = abs(lm[MENTON, 1] - lm[LOWER_LIP_BOT, 1]) / fh
        f['chin_width'] = _dist(lm[R_CHIN], lm[L_CHIN]) / bz
        f['chin_pointedness_angle'] = _angle_at(lm[MENTON], lm[R_CHIN], lm[L_CHIN])
        f['jaw_bigonial_width'] = _dist(lm[R_GONION], lm[L_GONION]) / bz
        return f

    # --------------------------------------------------------------
    # 7. Forehead / hairline features
    # --------------------------------------------------------------
    def _feat_forehead(self, lm, s):
        f, bz, fh = {}, s['bizyg'], s['face_h']

        f['forehead_height'] = abs(lm[TRICHION, 1] - lm[GLABELLA, 1]) / fh
        f['forehead_width_upper'] = _dist(lm[R_FOREHEAD_UPPER], lm[L_FOREHEAD_UPPER]) / bz
        f['forehead_width_mid']   = _dist(lm[R_FOREHEAD_MID],   lm[L_FOREHEAD_MID])   / bz

        f['hairline_height'] = (lm[GLABELLA, 1] - lm[TRICHION, 1]) / fh

        f['forehead_taper_ratio'] = (f['forehead_width_upper']
                                     / (f['forehead_width_mid'] + 1e-12))
        return f

    # --------------------------------------------------------------
    # 8. Face global features
    # --------------------------------------------------------------
    def _feat_face_global(self, lm, s):
        f = {}
        bz, fh = s['bizyg'], s['face_h']

        f['face_aspect_ratio'] = bz / (fh + 1e-12)

        oval = lm[FACE_OVAL]
        area = _polygon_area_ordered(oval)
        perim = float(np.sum(np.linalg.norm(np.diff(oval, axis=0, append=oval[:1]), axis=1)))
        f['face_roundness']    = 4 * np.pi * area / (perim ** 2 + 1e-12)
        jaw_w = _dist(lm[R_GONION], lm[L_GONION])
        fhd_w = _dist(lm[R_FOREHEAD_UPPER], lm[L_FOREHEAD_UPPER])
        f['face_width_uniformity'] = 0.5 * (jaw_w + fhd_w) / (bz + 1e-12)
        f['jaw_to_forehead_ratio'] = jaw_w / (fhd_w + 1e-12)
        chn_w = _dist(lm[R_CHIN], lm[L_CHIN])
        f['face_triangularity'] = (fhd_w - chn_w) / (fhd_w + 1e-12)

        mirror_pairs = [
            (R_EYE_OUTER, L_EYE_OUTER), (R_EYE_INNER, L_EYE_INNER),
            (R_EYE_TOP,   L_EYE_TOP),   (R_EYE_BOT,   L_EYE_BOT),
            (R_ALA,       L_ALA),       (R_ALAR_CREST, L_ALAR_CREST),
            (R_COMMISSURE, L_COMMISSURE),
            (R_ZYGO,      L_ZYGO),      (R_GONION,    L_GONION),
            (R_FOREHEAD_UPPER, L_FOREHEAD_UPPER),
            (R_CHIN,      L_CHIN),      (R_CHEEK,     L_CHEEK),
            (R_MAXILLA,   L_MAXILLA),
        ]
        err = []
        for a, b in mirror_pairs:
            pa, pb = lm[a], lm[b].copy()
            pb[0] = -pb[0]
            err.append(np.linalg.norm(pa - pb))
        f['face_asymmetry'] = float(np.mean(err)) / bz
        return f

    # --------------------------------------------------------------
    # 9. Midface / cheek features
    # --------------------------------------------------------------
    def _feat_midface_cheek(self, lm, s):
        f, bz = {}, s['bizyg']

        f['midface_width'] = _dist(lm[R_MAXILLA], lm[L_MAXILLA]) / bz

        oval = lm[FACE_OVAL]
        for side, cheek_idx, dir_sign in [('r', R_CHEEK, -1), ('l', L_CHEEK, +1)]:
            cy = lm[cheek_idx, 1]
            cx = lm[cheek_idx, 0]
            side_pts = oval[np.sign(oval[:, 0]) == dir_sign]
            diffs = np.abs(side_pts[:, 1] - cy)
            order = np.argsort(diffs)[:2]
            p1, p2 = side_pts[order]
            if abs(p2[1] - p1[1]) < 1e-6:
                xe = 0.5 * (p1[0] + p2[0])
            else:
                t = (cy - p1[1]) / (p2[1] - p1[1])
                xe = p1[0] + t * (p2[0] - p1[0])
            f[f'malar_bulge_{side}'] = dir_sign * (xe - cx) / bz

        f['malar_bulge_mean'] = 0.5 * (f['malar_bulge_r'] + f['malar_bulge_l'])
        f['malar_bulge_asym'] = (abs(f['malar_bulge_r'] - f['malar_bulge_l'])
                                 / (abs(f['malar_bulge_mean']) + 1e-12))

        for side, cheek_idx in [('r', R_CHEEK), ('l', L_CHEEK)]:
            cy = lm[cheek_idx, 1]
            sgn = -1 if side == 'r' else +1
            side_pts = oval[np.sign(oval[:, 0]) == sgn]
            nearby = side_pts[np.argsort(np.abs(side_pts[:, 1] - cy))[:4]]
            poly_pts = np.vstack([nearby, lm[cheek_idx][None, :]])
            f[f'cheek_area_{side}'] = _hull_area(poly_pts) / (bz ** 2)
        f['cheek_area_mean'] = 0.5 * (f['cheek_area_r'] + f['cheek_area_l'])
        f['cheek_area_asym'] = (abs(f['cheek_area_r'] - f['cheek_area_l'])
                                / (f['cheek_area_mean'] + 1e-12))
        return f
