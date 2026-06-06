"""`chain` mode: run several segments in sequence, feeding the LAST frame of one
as the start image of the next, then normalise + concatenate them with ffmpeg.

A spec is a JSON list of steps. Example (travel in, then let it come alive):
  [
    {"mode": "travel",  "motion": "dolly", "frames": 161, "distance": 2,
     "prompt": "a quiet street, still", "use_dmd": true},
    {"mode": "animate", "prompt": "the street comes to life, gentle traffic", "frames": 81}
  ]

`travel` keys: motion, frames, + shaping (radius/degrees/elevation_deg/distance/
look_dist/width/height/fov_deg/ease/keys), + prompt/pose_scale/guidance/fps/seed/use_dmd.
`animate` keys: prompt, size, frames, steps, seed, offload, convert_dtype, t5_cpu.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from . import trajectory
from .backends import lyra, wan

_TRAJ_KEYS = ("width", "height", "fov_deg", "ease", "radius", "degrees",
              "elevation_deg", "distance", "look_dist", "keys")


def _ffmpeg(args, *, dry_run):
    cmd = ["ffmpeg", "-y", *args]
    if dry_run:
        print("  ffmpeg " + " ".join(str(a) for a in args))
        return 0
    return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode


def _parse_size(size: str):
    for sep in ("*", "x", "X", ","):
        if sep in size:
            w, h = size.split(sep)
            return int(w), int(h)
    raise ValueError(f"bad size {size!r}; use e.g. 1280*720")


def extract_last_frame(video, out_png, *, dry_run=False):
    return _ffmpeg(["-sseof", "-1", "-i", str(video), "-update", "1", "-q:v", "2", str(out_png)],
                   dry_run=dry_run)


def normalize(video, out, w, h, fps, *, dry_run=False):
    vf = (f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
          f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,fps={fps}")
    return _ffmpeg(["-i", str(video), "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-an", str(out)], dry_run=dry_run)


def concat(videos, out, *, dry_run=False):
    listfile = Path(out).with_suffix(".concat.txt")
    if not dry_run:
        listfile.write_text("".join(f"file '{Path(v).resolve()}'\n" for v in videos))
    else:
        print(f"  # concat {[str(v) for v in videos]} -> {out}")
    return _ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)],
                   dry_run=dry_run)


def _run_step(settings, mode, params, input_image, seg_dir, *, dry_run):
    params = dict(params)
    if mode == "animate":
        out = seg_dir / "animate.mp4"
        rc, video = wan.animate(settings, str(input_image), save_file=str(out),
                                dry_run=dry_run, **params)
        return rc, (video or out)
    if mode == "travel":
        motion = params.pop("motion", "dolly")
        frames = params.pop("frames", 161)
        traj_kwargs = {k: params.pop(k) for k in list(params) if k in _TRAJ_KEYS}
        npz = seg_dir / "trajectory.npz"
        if dry_run:
            print(f"  # build trajectory {motion} frames={frames} -> {npz}")
        else:
            trajectory.build_and_save(motion, npz, frames, **traj_kwargs)
        return lyra.travel(settings, str(input_image), str(npz), num_frames=frames,
                           output_path=str(seg_dir), dry_run=dry_run, **params)
    raise SystemExit(f"unknown chain mode {mode!r} (use 'travel' or 'animate')")


def run_chain(settings, start_image, steps, out_path, *, size="1280*720", fps=16,
              work_dir=None, dry_run=False):
    if not steps:
        raise SystemExit("chain spec is empty")
    w, h = _parse_size(size)
    out_path = Path(out_path)
    work_dir = Path(work_dir) if work_dir else out_path.parent / (out_path.stem + "_segments")
    if not dry_run:
        work_dir.mkdir(parents=True, exist_ok=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    current_image = Path(start_image)
    normed = []
    for i, step in enumerate(steps):
        mode = step.get("mode")
        params = {k: v for k, v in step.items() if k != "mode"}
        seg_dir = work_dir / f"seg{i}_{mode}"
        if not dry_run:
            seg_dir.mkdir(parents=True, exist_ok=True)
        print(f"[scene-studio] chain step {i}: {mode}", file=sys.stderr)
        rc, video = _run_step(settings, mode, params, current_image, seg_dir, dry_run=dry_run)
        if rc != 0:
            raise SystemExit(f"chain step {i} ({mode}) failed with code {rc}")
        norm = seg_dir / "norm.mp4"
        normalize(video, norm, w, h, fps, dry_run=dry_run)
        normed.append(norm)
        if i + 1 < len(steps):
            nxt = seg_dir / "last.png"
            extract_last_frame(video, nxt, dry_run=dry_run)
            current_image = nxt

    concat(normed, out_path, dry_run=dry_run)
    print(f"[scene-studio] chain -> {out_path}")
    return 0
