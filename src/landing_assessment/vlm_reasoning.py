"""Scene reasoning via Qwen2-VL: identify obstructions and ground surface."""

import json
import re
from collections import Counter

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

VALID_SURFACES = {
    "paved", "concrete", "asphalt", "rooftop", "grass", "lawn", "field",
    "dirt", "soil", "gravel", "sand", "snow", "ice", "water", "mixed",
    "cobblestone", "tile", "mulch", "unknown",
}
VALID_INTUITIONS = {"safe", "caution", "unsafe"}
HUMAN_LABELS = {"person", "child", "infant", "human", "pedestrian"}

SYSTEM_PROMPT = """You are a perception module on a quadcopter that is about to land.
You see a downward-facing image. List ONLY the obstructions actually visible,
classify the ground surface, and rank surface types by landing suitability.

WHAT COUNTS AS AN OBSTRUCTION
A discrete object on the ground that the quadcopter could collide with: people,
animals, vehicles, trees, poles, wires, benches, rocks, debris, water, walls,
buildings, furniture, planters.

WHAT IS NOT AN OBSTRUCTION
- The ground surface itself (concrete, grass, asphalt) — goes in dominant_surface.
- The drone's own shadow.
- Shadows cast by objects.

CRITICAL RULES
- Use singular generic nouns: "person" not "people".
- ONE entry per object TYPE. If there are 5 people, exactly ONE entry "person".
- DO NOT repeat the same label or entry. Each label appears AT MOST ONCE.
- HUMANS ARE ALWAYS criticality 5.
- criticality scale: 5=catastrophic (humans, animals, water, wires);
  4=severe (cars, motorcycles, large rocks);
  3=moderate (benches, trees, walls, buildings);
  2=minor (debris, manhole covers, planters);
  1=trivial.
- overall_landing_intuition: EXACTLY one of safe, caution, unsafe.
- dominant_surface: REQUIRED. Look at the largest visible ground area and pick
  ONE specific surface: concrete, asphalt, paved, grass, lawn, field, dirt,
  soil, gravel, sand, snow, water, cobblestone, tile, rooftop, mulch, mixed.
  Use "unknown" ONLY as last resort if image is too dark or blurry to tell —
  this should almost never happen.
- landing_surface_ranking: list visible surface types BEST first, max 4.
  ALWAYS include at least the dominant_surface in this list.

EXAMPLE 1 (paved plaza with pedestrians, bench, planter, gravel patch):
{"obstructions":[{"label":"person","category":"person","criticality":5,"reason":"Multiple pedestrians."},{"label":"bench","category":"furniture","criticality":3,"reason":"Seating fixture."},{"label":"planter","category":"vegetation","criticality":2,"reason":"Raised plant bed."}],"dominant_surface":"concrete","landing_surface_ranking":["concrete","gravel"],"scene_summary":"Paved plaza with pedestrians, bench, planter.","overall_landing_intuition":"unsafe"}

EXAMPLE 2 (open grass field with one bicycle parked at the corner):
{"obstructions":[{"label":"bicycle","category":"vehicle","criticality":3,"reason":"Stationary bike at edge."}],"dominant_surface":"grass","landing_surface_ranking":["grass"],"scene_summary":"Open grass field with bicycle at corner.","overall_landing_intuition":"safe"}

EXAMPLE 3 (cobblestone courtyard with planter, no people):
{"obstructions":[{"label":"planter","category":"vegetation","criticality":2,"reason":"Plant bed."}],"dominant_surface":"cobblestone","landing_surface_ranking":["cobblestone","concrete"],"scene_summary":"Cobblestone courtyard with one planter.","overall_landing_intuition":"caution"}

Now analyze the actual image. Output ONLY the JSON. Do NOT repeat entries.
The dominant_surface MUST be a specific surface, not 'unknown'."""


def _extract_json(text: str) -> dict:
    """Pull the first balanced {...} object out of raw model text."""
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    if start == -1:
        raise ValueError("No '{' in VLM output.")
    depth, end, in_str, esc = 0, -1, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
    if end == -1:
        raise ValueError("Unmatched braces.")
    candidate = text[start:end]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
        cleaned = re.sub(r",\s*,", ",", cleaned)
        return json.loads(cleaned)


def _is_degenerate(token_ids: torch.Tensor, window: int = 30, repeat_thresh: int = 5) -> bool:
    """Detect degenerate repetition: a short n-gram repeating many times in
    the tail of generation, meaning decoding has collapsed and should be
    retried rather than trusted."""
    ids = token_ids.tolist()
    if len(ids) < window * 2:
        return False
    tail = ids[-window:]
    grams = [tuple(tail[i:i + 3]) for i in range(len(tail) - 2)]
    if not grams:
        return False
    most_common = Counter(grams).most_common(1)[0]
    return most_common[1] >= repeat_thresh


def normalize_vlm_output(d: dict) -> dict:
    """Clean, dedupe, and validate the raw JSON the VLM produced."""
    obs = d.get("obstructions", [])
    if not isinstance(obs, list):
        obs = []
    cleaned_obs = []
    for o in obs:
        if not isinstance(o, dict):
            continue
        lbl = str(o.get("label", "")).strip().lower()
        if lbl.endswith("ies"):
            lbl = lbl[:-3] + "y"
        elif lbl.endswith("ses"):
            lbl = lbl[:-2]
        elif lbl.endswith("s") and not lbl.endswith("ss"):
            lbl = lbl[:-1]
        if not lbl or lbl in {"other", "none", ""}:
            continue
        try:
            crit = max(1, min(5, int(o.get("criticality", 3))))
        except (TypeError, ValueError):
            crit = 3
        if any(h in lbl for h in HUMAN_LABELS):
            crit = 5
        cleaned_obs.append({
            "label": lbl,
            "category": str(o.get("category", "other")).strip().lower(),
            "criticality": crit,
            "reason": str(o.get("reason", "")).strip(),
        })

    seen, unique = set(), []
    for o in cleaned_obs:
        if o["label"] not in seen:
            seen.add(o["label"])
            unique.append(o)
    d["obstructions"] = unique

    surf = str(d.get("dominant_surface", "unknown")).strip().lower()
    d["dominant_surface"] = surf if surf in VALID_SURFACES else "unknown"

    ranking = d.get("landing_surface_ranking", [])
    if not isinstance(ranking, list):
        ranking = []
    cleaned_ranking, seen_r = [], set()
    for s in ranking:
        if not isinstance(s, str):
            continue
        s = s.strip().lower()
        if s in VALID_SURFACES and s not in seen_r:
            cleaned_ranking.append(s)
            seen_r.add(s)
    if not cleaned_ranking and d["dominant_surface"] != "unknown":
        cleaned_ranking = [d["dominant_surface"]]
    d["landing_surface_ranking"] = cleaned_ranking[:4]

    intuit = str(d.get("overall_landing_intuition", "")).lower()
    d["overall_landing_intuition"] = next(
        (w for w in VALID_INTUITIONS if w in intuit), "caution"
    )
    if (
        any(any(h in o["label"] for h in HUMAN_LABELS) for o in unique)
        and d["overall_landing_intuition"] == "safe"
    ):
        d["overall_landing_intuition"] = "unsafe"

    d["scene_summary"] = str(d.get("scene_summary", "")).strip()
    return d


class SceneReasoner:
    """Wraps a Qwen2-VL model to produce structured obstruction/surface
    JSON for a single downward-facing image."""

    def __init__(self, model_id: str, device: str = "cuda"):
        self.device = device
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
        ).eval()
        if device != "cuda":
            self.model.to(device)

    def analyze(self, image: Image.Image, debug: bool = False, max_retries: int = 2) -> dict:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Analyze this image for quadcopter landing. Output JSON only. Do NOT repeat entries."},
            ]},
        ]
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = process_vision_info(messages)
        inputs = self.proc(
            text=[text], images=img_in, videos=vid_in,
            padding=True, return_tensors="pt",
        ).to(self.device)

        last_err, last_raw = None, None
        for attempt in range(max_retries + 1):
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=600,
                    do_sample=False,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=4,
                    temperature=None, top_p=None, top_k=None,
                )
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
            gen_ids = trimmed[0]

            if _is_degenerate(gen_ids):
                last_err = "degenerate output (repetition loop)"
                if debug:
                    print(f"--- Attempt {attempt + 1}: DEGENERATE — retrying ---")
                continue

            raw = self.proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
            last_raw = raw
            if debug:
                print(f"--- Attempt {attempt + 1} ---\n{raw}\n--- end ---")
            try:
                return normalize_vlm_output(_extract_json(raw))
            except (ValueError, json.JSONDecodeError) as e:
                last_err = e
                continue

        print(f"=== Failed VLM output ===\n{(last_raw or '')[:1500]}\n===")
        raise RuntimeError(f"VLM JSON parse failed after {max_retries + 1} attempts: {last_err}")


FALLBACK_HAZARD_PROMPTS = [
    "person", "child", "animal", "dog",
    "car", "motorcycle", "bicycle",
    "tree", "pole", "lamp post",
    "fence", "wall", "building",
    "rooftop", "tile roof", "slate roof",
    "bench", "table",
    "water", "pond",
    "rock", "debris",
    "tarp", "canopy", "planter",
]
