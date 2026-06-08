"""CLI: open-vocabulary object detection on an image or a video with LocateAnything-3B.

Examples:
  # image: detect categories -> annotated PNG + JSON of boxes
  python -m locate_studio.detect --image street.jpg --classes "person,car,traffic light"

  # video: detect every 5th frame -> annotated MP4 + per-frame JSON
  python -m locate_studio.detect --video clip.mp4 --classes "person,dog" --stride 5

  # referring grounding (one phrase instead of class list)
  python -m locate_studio.detect --image room.jpg --query "the lamp on the left"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .worker import DEFAULT_MODEL, LocateAnythingWorker

# Distinct BGR colors cycled per label.
_PALETTE = [(56, 56, 255), (56, 255, 56), (255, 144, 30), (56, 255, 255),
            (255, 56, 200), (0, 165, 255), (200, 200, 0), (128, 0, 255)]


def _color(label: str) -> tuple:
    return _PALETTE[hash(label) % len(_PALETTE)]


def _draw(frame_bgr: np.ndarray, boxes: list[dict]) -> np.ndarray:
    for b in boxes:
        x1, y1, x2, y2 = (int(b[k]) for k in ("x1", "y1", "x2", "y2"))
        label = b.get("label", "obj")
        c = _color(label)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), c, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame_bgr, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), c, -1)
        cv2.putText(frame_bgr, label, (x1 + 1, max(9, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return frame_bgr


def _query_one(worker, pil_img, args) -> list[dict]:
    """Run the chosen task on one PIL image, return labeled pixel boxes."""
    kw = dict(generation_mode=args.generation_mode, max_new_tokens=args.max_new_tokens,
              temperature=args.temperature)
    if args.classes:
        cats = [c.strip() for c in args.classes.split(",") if c.strip()]
        return worker.detect_labeled(pil_img, cats, **kw)
    w, h = pil_img.size
    answer = (worker.point(pil_img, args.query, **kw) if args.point
              else worker.ground(pil_img, args.query, **kw))
    return worker.parse_boxes(answer, w, h, label=args.query)


def run_image(worker, args) -> int:
    src = Path(args.image).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"--image not found: {src}")
    img = Image.open(src).convert("RGB")
    boxes = _query_one(worker, img, args)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / f"{src.stem}_annotated.png"), _draw(frame, boxes))
    (out_dir / f"{src.stem}.json").write_text(json.dumps(boxes, indent=2))
    print(f"[locate] {len(boxes)} box(es) -> {out_dir}/{src.stem}_annotated.png  (+ {src.stem}.json)")
    return 0


def run_video(worker, args) -> int:
    src = Path(args.video).expanduser().resolve()
    if not src.exists():
        raise SystemExit(f"--video not found: {src}")
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{src.stem}_annotated.mp4"
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
                             max(1.0, fps / args.stride), (w, h))

    per_frame, idx, done = [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % args.stride == 0:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            boxes = _query_one(worker, pil, args)
            writer.write(_draw(frame, boxes))
            per_frame.append({"frame": idx, "boxes": boxes})
            done += 1
            if done % 10 == 0:
                print(f"[locate]   {done} frames done (idx {idx})...", flush=True)
            if args.max_frames and done >= args.max_frames:
                break
        idx += 1
    cap.release()
    writer.release()
    (out_dir / f"{src.stem}.json").write_text(json.dumps(per_frame, indent=2))
    print(f"[locate] {done} frames -> {out_mp4}  (+ {src.stem}.json)")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Object detection / grounding on an image or video (LocateAnything-3B)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an input image")
    src.add_argument("--video", help="path to an input video (detected frame-by-frame)")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--classes", help='comma-separated open-vocab categories, e.g. "person,car,dog"')
    what.add_argument("--query", help='a single referring phrase, e.g. "the man in red" (grounding)')
    p.add_argument("--point", action="store_true", help="with --query: point at it instead of boxing it")
    p.add_argument("--out", default="outputs/locate", help="output directory")
    p.add_argument("--stride", type=int, default=5, help="[video] run detection every Nth frame")
    p.add_argument("--max-frames", type=int, default=0, help="[video] stop after this many processed frames (0 = all)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    p.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (auto: cuda->mps->cpu)")
    p.add_argument("--generation-mode", default="hybrid", help="model decoding mode (hybrid|...) ")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    print(f"[locate] loading {args.model} on {args.device} ...", file=sys.stderr)
    worker = LocateAnythingWorker(args.model, device=args.device)
    return run_image(worker, args) if args.image else run_video(worker, args)


if __name__ == "__main__":
    raise SystemExit(main())
