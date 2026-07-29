"""Open-vocabulary obstruction detection via Grounding DINO."""

from PIL import Image
import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

from .vlm_reasoning import FALLBACK_HAZARD_PROMPTS, VALID_SURFACES


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def _split_concat_labels(label: str, valid_singles: set) -> str:
    """Grounding DINO sometimes returns multi-word concatenations; take the
    last word that matches a known valid prompt."""
    for w in reversed(label.split()):
        if w in valid_singles:
            return w
    return label


def _per_class_nms(detections: list, iou_thr: float = 0.5) -> list:
    out = []
    for cls in {d["label"] for d in detections}:
        items = sorted(
            [d for d in detections if d["label"] == cls],
            key=lambda x: -x["score"],
        )
        kept = []
        for d in items:
            if all(_iou(d["box"], k["box"]) < iou_thr for k in kept):
                kept.append(d)
        out.extend(kept)
    return out


class ObstructionDetector:
    """Wraps Grounding DINO for open-vocabulary, prompt-driven detection."""

    def __init__(self, model_id: str, box_threshold: float, text_threshold: float, device: str = "cuda"):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()

    @staticmethod
    def build_prompts(vlm_result: dict) -> list:
        """Combine obstructions the VLM saw with a fixed fallback hazard
        vocabulary, so detection isn't limited to what the VLM named."""
        expansions = {
            "vegetation": ["tree", "bush", "plant"],
            "structure": ["building", "wall", "rooftop"],
            "furniture": ["bench", "table", "chair"],
            "wire": ["wire", "power line", "cable"],
        }
        prompts = set()
        for o in vlm_result.get("obstructions", []):
            lbl = o["label"]
            if lbl in expansions:
                prompts.update(expansions[lbl])
            else:
                prompts.add(lbl)
        prompts.update(FALLBACK_HAZARD_PROMPTS)
        prompts = {p for p in prompts if p and p not in VALID_SURFACES}
        return sorted(prompts)

    @staticmethod
    def _format_query(prompts: list) -> str:
        return " . ".join(p.lower() for p in prompts) + " ."

    def detect(self, image: Image.Image, prompts: list, box_threshold: float = None,
               text_threshold: float = None, nms_iou: float = 0.5) -> list:
        box_threshold = box_threshold if box_threshold is not None else self.box_threshold
        text_threshold = text_threshold if text_threshold is not None else self.text_threshold
        query = self._format_query(prompts)
        inputs = self.proc(images=image, text=query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        target_size = torch.tensor([image.size[::-1]])
        results = self.proc.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=box_threshold, text_threshold=text_threshold,
            target_sizes=target_size,
        )[0]
        valid = set(prompts)
        detections = []
        for s, l, b in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "label": _split_concat_labels(l.strip(), valid),
                "score": float(s),
                "box": [float(c) for c in b.tolist()],
            })
        detections = _per_class_nms(detections, iou_thr=nms_iou)
        detections.sort(key=lambda x: -x["score"])
        return detections
