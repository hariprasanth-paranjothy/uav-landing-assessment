"""Free-space landing-site search via OpenCV distance transform."""

import cv2
import numpy as np
from PIL import Image


def build_obstruction_mask(image: Image.Image, detections: list) -> np.ndarray:
    W, H = image.size
    mask = np.zeros((H, W), dtype=bool)
    for d in detections:
        if "mask" in d:
            mask |= d["mask"]
    return mask


def interior_mask(image: Image.Image, edge_margin_frac: float) -> np.ndarray:
    """A mask of pixels at least `margin_px` from any image boundary, so
    candidates aren't chosen right at the frame edge."""
    W, H = image.size
    margin = int(edge_margin_frac * min(W, H))
    interior = np.zeros((H, W), dtype=bool)
    interior[margin:H - margin, margin:W - margin] = True
    return interior


def find_local_maxima(dist_map: np.ndarray, min_distance: int, top_k: int = 5) -> list:
    """Find up to top_k local maxima in the distance map, separated by at
    least `min_distance` pixels, via greedy non-max suppression."""
    H, W = dist_map.shape
    flat_idx = np.argsort(dist_map, axis=None)[::-1]  # high -> low
    picks = []
    taken = np.zeros_like(dist_map, dtype=bool)
    for idx in flat_idx:
        if len(picks) >= top_k:
            break
        y, x = divmod(idx, W)
        if dist_map[y, x] < 1.0:  # distance == 0 means inside an obstruction
            break
        if taken[y, x]:
            continue
        picks.append((int(x), int(y), float(dist_map[y, x])))
        y0, y1 = max(0, y - min_distance), min(H, y + min_distance + 1)
        x0, x1 = max(0, x - min_distance), min(W, x + min_distance + 1)
        taken[y0:y1, x0:x1] = True
    return picks  # list of (x, y, free_disk_radius_px)


def search_landing_candidates(image: Image.Image, detections: list, vehicle_radius_frac: float,
                               edge_margin_frac: float, top_k: int = 5) -> dict:
    """Returns ranked landing candidates plus the distance map and
    obstruction mask used to compute them (for visualization/scoring)."""
    W, H = image.size
    obstruction_mask = build_obstruction_mask(image, detections)
    interior = interior_mask(image, edge_margin_frac)

    free = (~obstruction_mask) & interior
    free_u8 = free.astype(np.uint8)

    dist = cv2.distanceTransform(free_u8, cv2.DIST_L2, 5)

    vehicle_r = int(vehicle_radius_frac * min(W, H))
    candidates = find_local_maxima(dist, min_distance=max(vehicle_r, 20), top_k=top_k)

    return {
        "obstruction_mask": obstruction_mask,
        "distance_map": dist,
        "candidates": candidates,
        "vehicle_radius_px": vehicle_r,
    }
