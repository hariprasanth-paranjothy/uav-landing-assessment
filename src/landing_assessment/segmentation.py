"""Box-prompted obstruction segmentation via SAM."""

from PIL import Image
import torch
from transformers import SamModel, SamProcessor


class ObstructionSegmenter:
    """Wraps SAM to turn Grounding DINO boxes into pixel masks."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.proc = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id).to(device).eval()

    def segment(self, image: Image.Image, detections: list) -> list:
        if not detections:
            return detections
        boxes = [[d["box"] for d in detections]]
        inputs = self.proc(image, input_boxes=boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, multimask_output=False)
        masks = self.proc.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0].squeeze(1).numpy().astype(bool)

        enriched = []
        for det, mask in zip(detections, masks):
            d = dict(det)
            d["mask"] = mask
            d["mask_area_px"] = int(mask.sum())
            enriched.append(d)
        return enriched

    @staticmethod
    def filter_oversized(detections: list, image: Image.Image, max_area_frac: float = 0.55) -> list:
        """Drop masks that cover an implausibly large fraction of the frame
        (usually a mis-segmented background region)."""
        W, H = image.size
        total = W * H
        out, dropped = [], []
        for d in detections:
            if "mask" not in d:
                out.append(d)
                continue
            if d["mask_area_px"] / total > max_area_frac:
                dropped.append((d["label"], d["mask_area_px"] / total))
            else:
                out.append(d)
        if dropped:
            for lbl, frac in dropped:
                print(f"    dropped oversized: {lbl} ({frac * 100:.0f}% of image)")
        return out
