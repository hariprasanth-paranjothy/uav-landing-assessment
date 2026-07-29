"""End-to-end pipeline: image in, landing assessment out."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .config import Config
from .detection import ObstructionDetector
from .landing_search import search_landing_candidates
from .scoring import compute_landing_assessment
from .segmentation import ObstructionSegmenter
from .visualization import visualize_landing_assessment
from .vlm_reasoning import SceneReasoner


class LandingAssessmentPipeline:
    """Loads all three models once and exposes a single `analyze()` call.

    Example
    -------
    >>> from landing_assessment import Config, LandingAssessmentPipeline
    >>> pipeline = LandingAssessmentPipeline(Config())
    >>> result = pipeline.analyze("path/to/image.jpg", save_dir="results")
    >>> result["assessment"]["recommendation"]
    'SAFE'
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()
        print(f"Loading models on device: {self.config.device}")

        print(f"  [1/3] Scene reasoning VLM:  {self.config.vlm_model_id}")
        self.reasoner = SceneReasoner(self.config.vlm_model_id, self.config.device)

        print(f"  [2/3] Open-vocab detector:  {self.config.grounding_model_id}")
        self.detector = ObstructionDetector(
            self.config.grounding_model_id,
            self.config.box_threshold,
            self.config.text_threshold,
            self.config.device,
        )

        print(f"  [3/3] Segmenter:            {self.config.sam_model_id}")
        self.segmenter = ObstructionSegmenter(self.config.sam_model_id, self.config.device)

        print("Models loaded.")

    @property
    def _weights(self) -> dict:
        return {
            "freespace": self.config.w_freespace,
            "freedisk": self.config.w_freedisk,
            "hazard": self.config.w_hazard,
            "surface": self.config.w_surface,
        }

    def analyze(self, image_path, show: bool = True, save_dir: str = None, verbose: bool = True) -> dict:
        image_path = Path(image_path)
        if verbose:
            print(f"[1/5] Loading {image_path.name} ...")
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((self.config.max_image_size, self.config.max_image_size))

        if verbose:
            print("[2/5] VLM scene reasoning ...")
        vlm_out = self.reasoner.analyze(image, max_retries=self.config.max_vlm_retries)
        if verbose:
            print(f"      surface={vlm_out['dominant_surface']}, "
                  f"ranking={vlm_out['landing_surface_ranking']}, "
                  f"obstructions={[o['label'] for o in vlm_out['obstructions']]}")

        if verbose:
            print("[3/5] Open-vocabulary detection + segmentation ...")
        prompts = self.detector.build_prompts(vlm_out)
        dets = self.detector.detect(image, prompts)
        seg_dets = self.segmenter.segment(image, dets) if dets else []
        seg_dets = self.segmenter.filter_oversized(seg_dets, image)
        if verbose:
            print(f"      {len(seg_dets)} obstructions after filtering")

        if verbose:
            print("[4/5] Landing site search ...")
        search = search_landing_candidates(
            image, seg_dets,
            vehicle_radius_frac=self.config.vehicle_radius_frac,
            edge_margin_frac=self.config.edge_margin_frac,
            top_k=5,
        )
        if verbose:
            print(f"      {len(search['candidates'])} candidate landing points")

        if verbose:
            print("[5/5] Risk scoring ...")
        assessment = compute_landing_assessment(image, vlm_out, seg_dets, search, weights=self._weights)
        if verbose:
            print(f"      RECOMMENDATION: {assessment['recommendation']}  "
                  f"(safety={assessment['safety_score']:.0f}, risk={assessment['risk_score']:.0f})")

        save_path = None
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(save_dir / f"{image_path.stem}_assessment.png")
        if show or save_path:
            visualize_landing_assessment(
                image, seg_dets, assessment,
                weights=self._weights,
                factor_of_safety=self.config.factor_of_safety,
                title=f"Landing assessment — {image_path.name}",
                save_path=save_path,
            )

        if save_dir:
            clean_a = {k: v for k, v in assessment.items() if not isinstance(v, np.ndarray)}
            clean_a.pop("all_candidates_scored", None)
            clean_d = [{k: v for k, v in d.items() if k != "mask"} for d in seg_dets]
            sidecar = {"image": image_path.name, "vlm": vlm_out,
                       "detections": clean_d, "assessment": clean_a}
            with open(save_dir / f"{image_path.stem}_assessment.json", "w") as f:
                json.dump(sidecar, f, indent=2, default=str)
            if verbose:
                print(f"      saved: {save_path}")

        return {"image_path": str(image_path), "vlm_result": vlm_out,
                "detections": seg_dets, "assessment": assessment}
