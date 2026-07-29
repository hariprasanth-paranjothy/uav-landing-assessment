#!/usr/bin/env python
"""
Run the landing assessment pipeline on a single image, a directory of
images, or a specific list of images.

Examples
--------
    # Single image, show + save
    python scripts/run_analysis.py path/to/image.jpg --output-dir results

    # All images in a directory, headless (no plt.show), save everything
    python scripts/run_analysis.py path/to/image_dir/ --output-dir results --no-show

    # A specific subset of images
    python scripts/run_analysis.py img1.jpg img2.jpg img3.jpg --output-dir results
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from landing_assessment import Config, LandingAssessmentPipeline  # noqa: E402

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_image_paths(inputs: list) -> list:
    paths = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in VALID_EXT))
        elif p.is_file():
            paths.append(p)
        else:
            print(f"WARNING: not found, skipping: {p}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", help="Image file(s) and/or directory of images")
    parser.add_argument("--output-dir", default="results", help="Where to save assessment PNGs/JSONs")
    parser.add_argument("--no-show", action="store_true", help="Don't call plt.show() (headless/CI use)")
    parser.add_argument("--box-threshold", type=float, default=None, help="Override Grounding DINO box threshold")
    parser.add_argument("--text-threshold", type=float, default=None, help="Override Grounding DINO text threshold")
    args = parser.parse_args()

    config = Config()
    if args.box_threshold is not None:
        config.box_threshold = args.box_threshold
    if args.text_threshold is not None:
        config.text_threshold = args.text_threshold

    image_paths = collect_image_paths(args.inputs)
    if not image_paths:
        print("No valid images found. Exiting.")
        sys.exit(1)

    print(f"Found {len(image_paths)} image(s) to analyze.\n")

    pipeline = LandingAssessmentPipeline(config)

    rows = []
    for p in image_paths:
        print("=" * 70)
        result = pipeline.analyze(p, show=not args.no_show, save_dir=args.output_dir, verbose=True)
        a = result["assessment"]
        rows.append({
            "image": Path(result["image_path"]).name,
            "recommendation": a["recommendation"],
            "safety_score": a["safety_score"],
            "risk_score": a["risk_score"],
            "free_space_ratio": a["sub_scores"]["free_space_ratio"],
            "free_disk_adequacy": a["sub_scores"]["free_disk_adequacy"],
            "hazard_safety": a["sub_scores"]["hazard_safety"],
            "surface_suitability": a["sub_scores"]["surface_suitability"],
            "max_free_radius_px": a["geometry"]["max_inscribed_free_radius_px"],
            "landing_xy": a["geometry"]["optimal_landing_xy"],
            "vlm_intuition": result["vlm_result"]["overall_landing_intuition"],
            "vlm_surface": result["vlm_result"]["dominant_surface"],
        })
        print()

    if len(rows) > 1:
        df = pd.DataFrame(rows)
        out_csv = Path(args.output_dir) / "summary.csv"
        df.to_csv(out_csv, index=False)
        print(f"Summary written to {out_csv}\n")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
