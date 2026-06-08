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

from .worker import DEFAULT_MODEL, LocateAnythingWorker, _pick_device

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
    """Run the chosen task on one PIL image; boxes are returned in ORIGINAL pixels.
    If --max-size is set we shrink what the MODEL sees (fewer vision tokens -> smaller
    tensors, dodging Metal's 4GB/tensor cap and going faster), but parse boxes against the
    original size since the model's coords are resolution-independent ([0,1000])."""
    kw = dict(generation_mode=args.generation_mode, max_new_tokens=args.max_new_tokens,
              temperature=args.temperature)
    ow, oh = pil_img.size
    img = pil_img
    if args.max_size and max(ow, oh) > args.max_size:
        s = args.max_size / max(ow, oh)
        img = pil_img.resize((max(1, round(ow * s)), max(1, round(oh * s))), Image.LANCZOS)
    if args.classes:
        cats = [c.strip() for c in args.classes.split(",") if c.strip()]
        if args.fast:
            return worker.parse_labeled_boxes(worker.detect(img, cats, **kw), ow, oh)
        out = []
        for c in cats:
            out += worker.parse_boxes(worker.detect(img, [c], **kw), ow, oh, label=c)
        return out
    ans = (worker.point(img, args.query, **kw) if args.point
           else worker.ground(img, args.query, **kw))
    return worker.parse_boxes(ans, ow, oh, label=args.query)


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

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = out_dir / f"{src.stem}_annotated.mp4"
    # Write H.264 via imageio-ffmpeg. cv2's mp4v output decodes as a green mess in QuickTime
    # and isn't finalized cleanly on interrupt; H.264 plays everywhere. Every frame is written
    # at the source fps (smooth); detection runs every --stride frames, boxes held in between.
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise SystemExit("video output needs imageio: pip install imageio imageio-ffmpeg")
    writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264",
                                quality=8, macro_block_size=None)

    boxes, per_frame, idx, n_det = [], [], 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride == 0:
                pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                boxes = _query_one(worker, pil, args)
                per_frame.append({"frame": idx, "boxes": boxes})
                n_det += 1
                if n_det % 10 == 0:
                    print(f"[locate]   detected on {n_det} frames (idx {idx})...", flush=True)
            writer.append_data(cv2.cvtColor(_draw(frame, boxes), cv2.COLOR_BGR2RGB))
            idx += 1
            if args.max_frames and idx >= args.max_frames:
                break
    finally:
        cap.release()
        writer.close()   # finalize the mp4 (moov atom) even on early exit / interrupt
    (out_dir / f"{src.stem}.json").write_text(json.dumps(per_frame, indent=2))
    print(f"[locate] {idx} frames written, detection on {n_det} -> {out_mp4}  (+ {src.stem}.json)")
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
    p.add_argument("--fast", action="store_true",
                   help="detect all --classes in ONE call per frame (~Nx faster, key for video on MPS/CPU; labels best-effort)")
    p.add_argument("--out", default="outputs/locate", help="output directory")
    p.add_argument("--stride", type=int, default=5, help="[video] detect every Nth frame (boxes held in between); 1 = every frame")
    p.add_argument("--max-frames", type=int, default=0, help="[video] stop after reading this many input frames (0 = whole clip)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF repo id or local path")
    p.add_argument("--device", default="auto", help="auto|cuda|mps|cpu (auto: cuda->mps->cpu)")
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"],
                   help="precision (auto: bf16 cuda / fp16 mps / fp32 cpu)")
    p.add_argument("--max-size", type=int, default=0,
                   help="downscale longest side to N px before detection (0=off; auto 768 on mps for Metal's 4GB/tensor limit)")
    p.add_argument("--generation-mode", default="hybrid", help="model decoding mode (hybrid|...) ")
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    dev = _pick_device(args.device)
    # Metal caps a single tensor at 2**32 bytes (4GB); fp16 (auto) + a downscale keep the
    # vision tensors under it. Auto-pick a safe input cap on MPS if the user didn't set one.
    if args.max_size == 0 and dev == "mps":
        args.max_size = 768
        print(f"[locate] mps -> fp16 + input capped to {args.max_size}px (Metal 4GB/tensor limit; "
              f"lower --max-size if it still aborts, or use --device cpu)", file=sys.stderr)
    print(f"[locate] loading {args.model} on {dev} ...", file=sys.stderr)
    worker = LocateAnythingWorker(args.model, device=args.device, dtype=args.dtype)
    return run_image(worker, args) if args.image else run_video(worker, args)


if __name__ == "__main__":
    raise SystemExit(main())
