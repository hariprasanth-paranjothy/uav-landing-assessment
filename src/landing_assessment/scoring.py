"""Multi-factor weighted risk scoring for candidate landing points."""

import numpy as np
from PIL import Image

HAZARD_WEIGHTS = {
    "person": 1.00, "child": 1.00, "human": 1.00, "pedestrian": 1.00,
    "animal": 1.00, "dog": 1.00, "cat": 1.00,
    "car": 0.90, "vehicle": 0.90, "motorcycle": 0.90, "bicycle": 0.70,
    "water": 0.95, "pond": 0.95, "pool": 0.95,
    "wire": 0.95, "power line": 0.95, "cable": 0.95,
    "tree": 0.65, "pole": 0.70, "lamp": 0.70,
    "building": 0.85, "rooftop": 0.92, "roof": 0.92, "wall": 0.55, "fence": 0.55,
    "tile": 0.92, "slate": 0.92,
    "tarp": 0.75, "canopy": 0.75,
    "bench": 0.55, "table": 0.55, "chair": 0.55,
    "rock": 0.45, "debris": 0.45, "trash": 0.40,
    "plant": 0.35, "bush": 0.35, "manhole": 0.30, "planter": 0.40,
}
DEFAULT_HAZARD_WEIGHT = 0.50

# Used both for the dominant_surface score and for ranking candidates by
# their nearest-ranked surface.
SURFACE_SUITABILITY = {
    "concrete": 1.00, "asphalt": 1.00, "paved": 1.00,
    "grass": 0.85, "lawn": 0.85, "field": 0.80, "dirt": 0.70, "soil": 0.70,
    "rooftop": 0.50, "sand": 0.55, "gravel": 0.40,
    "cobblestone": 0.45, "tile": 0.50,
    "mulch": 0.20,
    "snow": 0.40, "ice": 0.20, "water": 0.00,
    "mixed": 0.55, "unknown": 0.40,
}


def _hazard_weight_for(label: str) -> float:
    lbl = label.lower()
    for k, w in HAZARD_WEIGHTS.items():
        if k in lbl:
            return w
    return DEFAULT_HAZARD_WEIGHT


def _free_disk_adequacy(max_r: float, req_r: float) -> float:
    if max_r < req_r:
        return 0.0
    return min(0.5 + (max_r - req_r) / max(req_r, 1.0), 1.0)


def _score_candidate(image: Image.Image, detections: list, vlm_result: dict,
                      candidate, vehicle_r: int, weights: dict) -> dict:
    """Score one candidate landing point. Returns the four sub-scores plus
    aggregate safety_01 in [0, 1]."""
    W, H = image.size
    bx, by, max_r = candidate
    check_r = max(vehicle_r * 1.8, 25)

    yy, xx = np.ogrid[:H, :W]
    nbhd = ((yy - by) ** 2 + (xx - bx) ** 2) <= check_r ** 2
    nbhd_area = int(nbhd.sum())

    obstruction_mask = np.zeros((H, W), dtype=bool)
    for d in detections:
        if "mask" in d:
            obstruction_mask |= d["mask"]

    free_in_nbhd = int((nbhd & ~obstruction_mask).sum())
    s_freespace = free_in_nbhd / max(nbhd_area, 1)

    s_freedisk = _free_disk_adequacy(max_r, vehicle_r)

    nearby = []
    for d in detections:
        if "mask" not in d:
            continue
        if (d["mask"] & nbhd).any():
            w = _hazard_weight_for(d["label"])
            nearby.append({"label": d["label"], "severity": w})
    max_hazard = max((h["severity"] for h in nearby), default=0.0)
    s_hazard = 1.0 - max_hazard

    dominant = vlm_result.get("dominant_surface", "unknown")
    s_surface = SURFACE_SUITABILITY.get(dominant, 0.4)

    safety_01 = (
        weights["freespace"] * s_freespace
        + weights["freedisk"] * s_freedisk
        + weights["hazard"] * s_hazard
        + weights["surface"] * s_surface
    )

    return {
        "candidate": (bx, by, max_r),
        "sub_scores": {
            "free_space_ratio": round(s_freespace, 3),
            "free_disk_adequacy": round(s_freedisk, 3),
            "hazard_safety": round(s_hazard, 3),
            "surface_suitability": round(s_surface, 3),
        },
        "nearby_hazards": sorted(nearby, key=lambda x: -x["severity"]),
        "max_hazard": round(max_hazard, 2),
        "safety_01": safety_01,
    }


def compute_landing_assessment(image: Image.Image, vlm_result: dict, detections: list,
                                search_result: dict, weights: dict = None) -> dict:
    """Score all candidates, pick the best, and produce the final assessment
    dict (recommendation, scores, rationale, geometry, hazards)."""
    weights = weights or {"freespace": 0.30, "freedisk": 0.35, "hazard": 0.20, "surface": 0.15}
    vehicle_r = search_result["vehicle_radius_px"]

    if not search_result["candidates"]:
        return {
            "recommendation": "UNSAFE",
            "safety_score": 0.0,
            "risk_score": 100.0,
            "rationale": "no obstruction-free disk found anywhere in the interior of the frame",
            "best_candidate": None,
            "all_candidates_scored": [],
            "sub_scores": {"free_space_ratio": 0.0, "free_disk_adequacy": 0.0,
                           "hazard_safety": 0.0, "surface_suitability": 0.0},
            "geometry": {"optimal_landing_xy": None,
                         "max_inscribed_free_radius_px": 0.0,
                         "vehicle_radius_required_px": vehicle_r,
                         "neighborhood_radius_px": 0},
            "flags": {"hard_unsafe_geometry": True, "critical_hazard_nearby": False,
                      "max_hazard_severity": 0.0},
            "nearby_hazards": [],
            "obstruction_mask": search_result["obstruction_mask"],
            "distance_map": search_result["distance_map"],
        }

    all_scored = [
        _score_candidate(image, detections, vlm_result, c, vehicle_r, weights)
        for c in search_result["candidates"]
    ]
    best = max(all_scored, key=lambda s: s["safety_01"])

    bx, by, max_r = best["candidate"]
    safety_score = round(best["safety_01"] * 100, 1)
    risk_score = round(100 - safety_score, 1)

    hard_unsafe_geometry = max_r < vehicle_r
    has_critical = any(
        any(k in h["label"].lower() for k in ("person", "child", "animal", "dog", "water", "pond"))
        for h in best["nearby_hazards"]
    )

    if hard_unsafe_geometry or safety_score < 40:
        recommendation = "UNSAFE"
    elif has_critical or safety_score < 75:
        recommendation = "CAUTION"
    else:
        recommendation = "SAFE"

    parts = []
    if hard_unsafe_geometry:
        parts.append(
            f"largest available landing disk ({max_r:.0f}px) is smaller than "
            f"required vehicle disk ({vehicle_r}px)"
        )
    if has_critical:
        crit = sorted({
            h["label"] for h in best["nearby_hazards"]
            if any(k in h["label"].lower() for k in ("person", "child", "animal", "dog", "water"))
        })
        parts.append(f"critical hazards near landing point: {', '.join(crit)}")
    if not parts:
        parts.append(
            f"optimal landing point at ({bx},{by}) with {max_r:.0f}px clear radius; "
            f"surface={vlm_result.get('dominant_surface', 'unknown')}; "
            f"max nearby hazard severity {best['max_hazard']:.2f}"
        )
    rationale = "; ".join(parts)

    return {
        "recommendation": recommendation,
        "safety_score": safety_score,
        "risk_score": risk_score,
        "rationale": rationale,
        "best_candidate": (bx, by, max_r),
        "all_candidates_scored": all_scored,
        "sub_scores": best["sub_scores"],
        "geometry": {
            "optimal_landing_xy": (bx, by),
            "max_inscribed_free_radius_px": round(max_r, 1),
            "vehicle_radius_required_px": vehicle_r,
            "neighborhood_radius_px": int(max(vehicle_r * 1.8, 25)),
        },
        "flags": {
            "hard_unsafe_geometry": bool(hard_unsafe_geometry),
            "critical_hazard_nearby": bool(has_critical),
            "max_hazard_severity": best["max_hazard"],
        },
        "nearby_hazards": best["nearby_hazards"],
        "obstruction_mask": search_result["obstruction_mask"],
        "distance_map": search_result["distance_map"],
    }
