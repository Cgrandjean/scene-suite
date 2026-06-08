"""Thin wrapper around NVIDIA LocateAnything-3B.

Loads the model once (it's stateful / expensive to load) and exposes the three
localization tasks the model supports, plus a parser that turns its ``<box>``
token output into pixel-coordinate boxes.

Usage matches the official model card (https://huggingface.co/nvidia/LocateAnything-3B):
the model ships custom code, so the processor/model are loaded with
``trust_remote_code=True`` and driven through the model's own ``generate``.
"""

from __future__ import annotations

import os
import re

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

DEFAULT_MODEL = os.environ.get("LOCATE_MODEL", "nvidia/LocateAnything-3B")

# Prompt templates (verbatim from the model card).
_TMPL = {
    "detect": "Locate all the instances that matches the following description: {q}.",
    "ground": "Locate all the instances that match the following description: {q}.",
    "point": "Point to: {q}.",
}
# Boxes come back normalised to [0, 1000] as <box><x1><y1><x2><y2></box>.
_BOX_RE = re.compile(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>")


def _pick_device(device: str) -> str:
    """Resolve 'auto' to cuda -> mps (Apple Silicon) -> cpu."""
    if device and device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class LocateAnythingWorker:
    """Stateful worker: load the model once, serve many perception queries."""

    def __init__(self, model_path: str = DEFAULT_MODEL, device: str = "auto",
                 dtype: "torch.dtype | None" = None):
        self.device = _pick_device(device)
        # bf16 only really pays off on CUDA; MPS/CPU are safer (and often only work) in fp32.
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_path, torch_dtype=self.dtype, trust_remote_code=True,
        ).to(self.device).eval()

    @torch.no_grad()
    def predict(self, image: Image.Image, question: str, *, generation_mode: str = "hybrid",
                max_new_tokens: int = 2048, temperature: float = 0.7,
                verbose: bool = False) -> str:
        """Run one query against one image; return the raw text answer."""
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]}]
        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt").to(self.device)

        response = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.dtype),
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_grid_hws=inputs.get("image_grid_hws", None),
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )
        return response[0] if isinstance(response, tuple) else response

    # --- task helpers -------------------------------------------------------
    def detect(self, image: Image.Image, categories: list[str], **kw) -> str:
        return self.predict(image, _TMPL["detect"].format(q="</c>".join(categories)), **kw)

    def ground(self, image: Image.Image, phrase: str, **kw) -> str:
        return self.predict(image, _TMPL["ground"].format(q=phrase), **kw)

    def point(self, image: Image.Image, phrase: str, **kw) -> str:
        return self.predict(image, _TMPL["point"].format(q=phrase), **kw)

    @staticmethod
    def parse_boxes(answer: str, width: int, height: int, label: str | None = None) -> list[dict]:
        """Turn the model's <box> tokens into pixel-coordinate boxes."""
        out = []
        for m in _BOX_RE.finditer(answer):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            box = {
                "x1": round(x1 / 1000 * width, 1), "y1": round(y1 / 1000 * height, 1),
                "x2": round(x2 / 1000 * width, 1), "y2": round(y2 / 1000 * height, 1),
            }
            if label is not None:
                box["label"] = label
            out.append(box)
        return out

    def detect_labeled(self, image: Image.Image, categories: list[str], **kw) -> list[dict]:
        """Detection with reliable per-box labels: query each category separately
        (so every returned box is unambiguously that class) and merge."""
        w, h = image.size
        boxes: list[dict] = []
        for cat in categories:
            boxes += self.parse_boxes(self.detect(image, [cat], **kw), w, h, label=cat)
        return boxes
