# UAV Landing Site Assessment

A foundation-model perception pipeline that looks at a single downward-facing
drone image and decides **where** a quadcopter can land, **whether** it's
safe to, and **why**.

It chains a vision-language model, an open-vocabulary detector, and a
promptable segmenter into one geometric risk-scoring pipeline — no
task-specific training or fine-tuning required.

```
Image → Qwen2-VL (scene reasoning) → Grounding DINO (detection)
      → SAM (segmentation) → distance-transform search → risk scoring → verdict
```

## Why this exists

Most "landing site detection" demos either (a) train a dedicated segmentation
model on a fixed obstacle taxonomy, or (b) hard-code a list of hazard classes.
Both break the moment a scene contains something outside that taxonomy. This
pipeline instead uses a VLM to *reason* about what's actually in the frame —
in open vocabulary — then hands that reasoning to a detector and segmenter to
get pixel-accurate obstruction masks, and finally reduces the whole thing to
simple, auditable geometry: "is there a big enough obstruction-free disk,
away from hazards, on a good surface?"

## Pipeline stages

| Stage | Model | Role |
|---|---|---|
| 1. Scene reasoning | `Qwen/Qwen2-VL-7B-Instruct` | Names visible obstructions, criticality-rates them, and classifies the dominant ground surface, as structured JSON |
| 2. Open-vocab detection | `IDEA-Research/grounding-dino-tiny` | Localizes every obstruction the VLM (plus a fixed hazard vocabulary) named, as boxes |
| 3. Segmentation | `facebook/sam-vit-large` | Turns each box into a pixel-accurate mask |
| 4. Site search | OpenCV `distanceTransform` | Finds the largest obstruction-free disks in the frame |
| 5. Risk scoring | weighted multi-factor score | Combines free-space ratio, free-disk adequacy, hazard severity, and surface suitability into a 0–100 safety score |
| 6. Visualization | matplotlib | Renders a four-panel dashboard: original → detections → landing geometry → score breakdown |

The final recommendation is one of `SAFE`, `CAUTION`, `UNSAFE`, with a
plain-English rationale and a full breakdown of sub-scores.

## Installation

```bash
git clone https://github.com/hariprasanth-paranjothy/uav-landing-assessment.git
cd uav-landing-assessment
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
pip install -e .
```

Requires a CUDA GPU with ≥16GB VRAM to run all three models comfortably
(Qwen2-VL-7B in fp16 is the dominant cost). CPU execution works but is slow
and only recommended for smoke-testing the geometry/scoring stages.

## Quick start

```bash
# Single image — shows the dashboard and saves it + a JSON sidecar
python scripts/run_analysis.py path/to/image.jpg --output-dir results

# A whole directory of images, headless (for servers/CI)
python scripts/run_analysis.py path/to/image_dir/ --output-dir results --no-show

# Robustness check — same image under noise/blur/fog/rotation/exposure shifts
python scripts/robustness_eval.py path/to/image.jpg --output-dir results/robustness
```

Or use it as a library:

```python
from landing_assessment import Config, LandingAssessmentPipeline

pipeline = LandingAssessmentPipeline(Config())
result = pipeline.analyze("path/to/image.jpg", save_dir="results")

print(result["assessment"]["recommendation"])   # SAFE / CAUTION / UNSAFE
print(result["assessment"]["safety_score"])      # 0-100
print(result["assessment"]["rationale"])
```

## Repository structure

```
uav-landing-assessment/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── src/
│   └── landing_assessment/
│       ├── __init__.py
│       ├── config.py            # Config dataclass (models, thresholds, weights)
│       ├── vlm_reasoning.py      # Qwen2-VL scene reasoning + JSON parsing/validation
│       ├── detection.py          # Grounding DINO open-vocabulary detection
│       ├── segmentation.py       # SAM box-prompted segmentation
│       ├── landing_search.py     # Distance-transform free-space search
│       ├── scoring.py            # Multi-factor risk scoring
│       ├── visualization.py      # Four-panel assessment dashboard
│       └── pipeline.py           # LandingAssessmentPipeline orchestrator
├── scripts/
│   ├── run_analysis.py           # CLI: single image / directory / batch
│   └── robustness_eval.py        # CLI: augmentation robustness stress test
└── assets/                       # sample dashboard screenshots
```

## Configuration

All thresholds and weights live in `Config` (`src/landing_assessment/config.py`)
and can be overridden either by constructing `Config(...)` directly or via
CLI flags on `scripts/run_analysis.py`:

| Field | Default | Meaning |
|---|---|---|
| `box_threshold` / `text_threshold` | `0.27` / `0.22` | Grounding DINO detection confidence thresholds |
| `vehicle_radius_frac` | `0.10` | Required clear-landing radius, as a fraction of `min(width, height)` |
| `edge_margin_frac` | `0.05` | Keep-out margin from the image border |
| `w_freespace` / `w_freedisk` / `w_hazard` / `w_surface` | `0.30` / `0.35` / `0.20` / `0.15` | Risk-score component weights (sum to 1.0) |
| `factor_of_safety` | `1.2` | Safety margin applied to the vehicle footprint when visualizing |

## Dataset

Development and evaluation images come from the
[Semantic Drone Dataset](https://www.kaggle.com/datasets/bulentsiyah/semantic-drone-dataset)
(nadir aerial imagery of urban scenes). The pipeline works on any
downward-facing RGB image — the dataset is only needed for batch evaluation.

```bash
kaggle datasets download -d bulentsiyah/semantic-drone-dataset -p data/ --unzip
```


## Sample Outputs

The figures below show representative landing-site assessments generated by the
pipeline on different aerial scenes. Each dashboard visualizes the complete
decision-making process, including:

- Original aerial image
- Open-vocabulary obstacle detection (Grounding DINO)
- Pixel-accurate segmentation (SAM)
- Distance-transform free-space analysis
- Optimal landing-zone selection
- Safety score, recommendation, and explanation

<p align="center">
  <img src="assets/Im1.webp" width="100%" alt="Landing Assessment Example 1">
</p>

<p align="center">
  <img src="assets/Im2.webp" width="100%" alt="Landing Assessment Example 2">
</p>

<p align="center">
  <img src="assets/Im3.webp" width="100%" alt="Landing Assessment Example 3">
</p>

<p align="center">
  <img src="assets/Im4.webp" width="100%" alt="Landing Assessment Example 4">
</p>

Each assessment is produced automatically by the foundation-model perception
pipeline without task-specific training, combining scene reasoning, open-vocabulary
detection, segmentation, geometric free-space analysis, and multi-factor risk
scoring to determine whether the UAV can land safely.

## Design notes

- **Open-vocabulary over fixed taxonomy.** The VLM's obstruction list feeds
  Grounding DINO's prompts directly, so the detector isn't limited to a
  pre-defined class list — it's supplemented with a fixed fallback hazard
  vocabulary as a safety net.
- **Hard overrides beat soft scores.** A landing point is forced to `UNSAFE`
  if the largest free disk is smaller than the vehicle's footprint,
  regardless of how the weighted score comes out. Similarly, any critical
  hazard (person, animal, water) within the landing neighborhood forces at
  least `CAUTION`.
- **Degenerate-generation detection.** The VLM call detects repetition
  collapse (a short n-gram repeating past a threshold) and retries instead of
  silently returning garbage JSON.
- **Robustness evaluation is a first-class script**, not an afterthought —
  `robustness_eval.py` checks whether the final recommendation is stable
  under noise, blur, exposure shifts, fog, and rotation.

## Limitations

- Runs one image at a time; no temporal/video consistency modeling.
- Risk weights were hand-tuned on the Semantic Drone Dataset's urban scenes;
  they're a reasonable starting point, not calibrated against real flight
  data.
- Qwen2-VL-7B in fp16 dominates both latency and VRAM; a smaller VLM (or
  batching multiple candidate crops) would be the natural next optimization.

## License

MIT — see [LICENSE](LICENSE).
