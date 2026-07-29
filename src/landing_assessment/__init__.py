"""
landing_assessment
===================

Vision-language-model-driven landing site risk assessment for UAVs
(quadcopters) using downward-facing aerial imagery.

Pipeline stages
---------------
1. SceneReasoner        - Qwen2-VL scene understanding (obstructions, surface)
2. ObstructionDetector  - Grounding DINO open-vocabulary detection
3. ObstructionSegmenter - SAM box-prompted segmentation
4. landing site search  - OpenCV distance-transform free-space search
5. risk scoring         - multi-factor weighted safety score
6. visualization        - four-panel assessment dashboard

The top-level entry point is `LandingAssessmentPipeline`.
"""

from .config import Config
from .pipeline import LandingAssessmentPipeline

__all__ = ["Config", "LandingAssessmentPipeline"]
__version__ = "0.1.0"
