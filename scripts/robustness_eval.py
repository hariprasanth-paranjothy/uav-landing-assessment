#!/usr/bin/env python
"""
Robustness evaluation: run the pipeline on each input image under a fixed
set of perturbations (Gaussian noise, motion blur, brightness extremes,
simulated fog, rotation) and check whether the final recommendation is
stable relative to the unperturbed original.

Example
-------
    python scripts/robustness_eval.py img1.jpg img2.jpg --output-dir results/robustness
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from landing_assessment import Config, LandingAssessmentPipeline  # noqa: E402


def aug_gaussian_noise(img, sigma=18):
    arr = np.array(img, dtype=np.float32)
    return Image.fromarray(np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8))


def aug_motion_blur(img, k=15):
    arr = np.array(img)
    kern = np.zeros((k, k))
    kern[k // 2, :] = 1.0 / k
    return Image.fromarray(cv2.filter2D(arr, -1, kern))


def aug_low_brightness(img):
    return ImageEnhance.Brightness(img).enhance(0.45)


def aug_overexposure(img):
    return ImageEnhance.Brightness(img).enhance(1.9)


def aug_fog(img, intensity=0.45):
    arr = np.array(img, dtype=np.float32)
    fog = np.ones_like(arr) * 255
    return Image.fromarray(np.clip(arr * (1 - intensity) + fog * intensity, 0, 255).astype(np.uint8))


def aug_rotation(img, angle=15):
    arr = np.array(img)
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return Image.fromarray(cv2.warpAffine(arr, M, (w, h), borderValue=(255, 255, 255)))


AUGS = [
    ("Original", None),
    ("Gaussian Noise", aug_gaussian_noise),
    ("Motion Blur", aug_motion_blur),
    ("Low Brightness", aug_low_brightness),
    ("Overexposure", aug_overexposure),
    ("Simulated Fog", aug_fog),
    ("15deg Rotation", aug_rotation),
]


def _load(path, size=896):
    img = Image.open(path).convert("RGB")
    img.thumbnail((size, size))
    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("images", nargs="+", help="Image file(s) to stress-test")
    parser.add_argument("--output-dir", default="results/robustness", help="Where to save augmented images/results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline = LandingAssessmentPipeline(Config())

    rows = []
    for img_path in args.images:
        img_path = Path(img_path)
        print(f"\n== {img_path.name} " + "=" * 50)
        orig_img = _load(img_path)
        ref_rec = None

        for aug_name, aug_fn in AUGS:
            if aug_fn is None:
                test_path = img_path
            else:
                test_img = aug_fn(orig_img)
                test_path = out_dir / f"{img_path.stem}_{aug_name.replace(' ', '_')}.jpg"
                test_img.save(test_path, quality=92)

            try:
                r = pipeline.analyze(test_path, save_dir=None, show=False, verbose=False)
                a = r["assessment"]
                if aug_name == "Original":
                    ref_rec = a["recommendation"]
                    match = ""
                else:
                    match = " MATCH" if a["recommendation"] == ref_rec else " DIFFERS"
                rows.append({
                    "image": img_path.name, "augmentation": aug_name,
                    "recommendation": a["recommendation"], "safety_score": a["safety_score"],
                    "risk_score": a["risk_score"], "n_detections": len(r["detections"]),
                    "matches_original": (match == " MATCH") if aug_name != "Original" else None,
                })
                print(f"  {aug_name:18s} -> {a['recommendation']:8s}  safety={a['safety_score']:5.1f}{match}")
            except Exception as e:
                rows.append({
                    "image": img_path.name, "augmentation": aug_name,
                    "recommendation": "ERROR", "safety_score": -1, "risk_score": -1,
                    "n_detections": 0, "matches_original": False,
                })
                print(f"  {aug_name:18s} -> ERROR: {e}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "robustness_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults written to {csv_path}")
    print(df.to_string(index=False))

    print("\nAgreement vs Original (per augmentation type):")
    for aug_name, _ in AUGS[1:]:
        orig_recs = df[df["augmentation"] == "Original"]["recommendation"].values
        aug_recs = df[df["augmentation"] == aug_name]["recommendation"].values
        n = min(len(orig_recs), len(aug_recs))
        if n == 0:
            continue
        agree = sum(o == a for o, a in zip(orig_recs[:n], aug_recs[:n]))
        pct = agree / n * 100
        bar = "#" * agree + "." * (n - agree)
        print(f"  {aug_name:18s} {bar}  {agree}/{n}  ({pct:.0f}%)")


if __name__ == "__main__":
    main()
