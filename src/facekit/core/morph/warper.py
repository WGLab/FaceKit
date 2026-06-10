"""
Triangular affine warping using Delaunay triangulation.

Core algorithm unchanged from the original FaceMorpher; only Python 2
compatibility shims and a broken test_local() function (which relied on
the removed scipy.ndimage.imread API) have been cleaned up.
"""
from __future__ import annotations

import numpy as np
import scipy.spatial as spatial


def bilinear_interpolate(img: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Interpolate over every image channel.

    https://en.wikipedia.org/wiki/Bilinear_interpolation

    :param img: up to 3-channel image
    :param coords: (2, m) array; row 0 = x coords, row 1 = y coords
    :returns: interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 neighbouring pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0 + 1]
    q12 = img[y0 + 1, x0]
    q22 = img[y0 + 1, x0 + 1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def grid_coordinates(points: np.ndarray) -> np.ndarray:
    """Grid (x, y) coordinates within the ROI of the supplied points."""
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1
    return np.asarray(
        [(x, y) for y in range(ymin, ymax) for x in range(xmin, xmax)],
        np.uint32,
    )


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """Warp each Delaunay triangle from ``src_img`` into ``result_img``."""
    roi_coords = grid_coordinates(dst_points)
    # -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(
            tri_affines[simplex_index],
            np.vstack((coords.T, np.ones(num_coords))),
        )
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)


def triangular_affine_matrices(vertices, src_points, dest_points):
    """Yield 2x3 affine matrix for each triangle (dest -> src)."""
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image(
    src_img: np.ndarray,
    src_points: np.ndarray,
    dest_points: np.ndarray,
    dest_shape,
    dtype=np.uint8,
) -> np.ndarray:
    """Warp ``src_img`` so that its ``src_points`` align to ``dest_points``."""
    num_chans = 3
    src_img = src_img[:, :, :3]

    rows, cols = dest_shape[:2]
    result_img = np.zeros((rows, cols, num_chans), dtype)

    delaunay = spatial.Delaunay(dest_points)
    tri_affines = np.asarray(
        list(triangular_affine_matrices(delaunay.simplices, src_points, dest_points))
    )

    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)
    return result_img
