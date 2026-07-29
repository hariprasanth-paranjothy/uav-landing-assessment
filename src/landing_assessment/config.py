"""Pipeline configuration."""

from dataclasses import dataclass

import torch


@dataclass
class Config:
    """Central configuration for the landing assessment pipeline.

    All values can be overridden via CLI flags in scripts/analyze.py,
    or by constructing `Config(...)` directly when using the package
    as a library.
    """

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # Model checkpoints (HuggingFace hub IDs)
    vlm_model_id: str = "Qwen/Qwen2-VL-7B-Instruct"
    grounding_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_id: str = "facebook/sam-vit-large"

    # Grounding DINO detection thresholds
    box_threshold: float = 0.27
    text_threshold: float = 0.22

    # Landing geometry
    vehicle_radius_frac: float = 0.10   # fraction of min(W, H)
    edge_margin_frac: float = 0.05      # keep-out margin from image border

    # Risk-score weights (must sum to 1.0)
    w_freespace: float = 0.30
    w_freedisk: float = 0.35
    w_hazard: float = 0.20
    w_surface: float = 0.15

    # Safety factor applied to the vehicle footprint when visualizing
    factor_of_safety: float = 1.2

    # Inference
    max_image_size: int = 896
    max_vlm_retries: int = 2
