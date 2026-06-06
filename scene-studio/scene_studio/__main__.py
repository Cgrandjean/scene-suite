"""Unified CLI:  python -m scene_studio {animate|travel|chain} ...

  animate  -> Wan 2.2 i2v        (scene comes alive, camera ~fixed)
  travel   -> Lyra 2.0           (camera moves through a frozen 3D world)
  chain    -> sequence + stitch  (e.g. travel then animate)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import chain, trajectory
from .backends import lyra, wan
from .config import Settings
from .trajectory import MOTIONS


def _cmd_animate(settings: Settings, a) -> int:
    rc, _ = wan.animate(
        settings, a.image, prompt=a.prompt, size=a.size, frames=a.frames, steps=a.steps,
        seed=a.seed, offload=not a.no_offload, convert_dtype=not a.no_convert_dtype,
        t5_cpu=a.t5_cpu, save_file=a.save_file, dry_run=a.dry_run,
    )
    return rc


def _cmd_travel(settings: Settings, a) -> int:
    home = settings.require("lyra").home
    out_dir = Path(a.output_path)
    npz = (out_dir if out_dir.is_absolute() else home / out_dir) / "trajectory.npz"

    kwargs = dict(width=a.width, height=a.height, fov_deg=a.fov,
                  ease=not a.linear, look_dist=a.look_dist)
    if a.motion == "orbit":
        kwargs.update(radius=a.radius, degrees=a.degrees, elevation_deg=a.elevation_deg)
    elif a.motion in ("dolly", "truck", "pedestal"):
        kwargs.update(distance=a.distance)
    elif a.motion in ("pan", "tilt"):
        kwargs.update(degrees=a.degrees)
    elif a.motion == "keyframes":
        kwargs.update(keys=json.loads(Path(a.keys).read_text()))

    if a.dry_run:
        print(f"[scene-studio] would build {a.motion} trajectory ({a.frames} frames) -> {npz}")
    else:
        trajectory.build_and_save(a.motion, npz, a.frames, **kwargs)

    rc, _ = lyra.travel(
        settings, a.image, str(npz), prompt=a.prompt, num_frames=a.frames,
        pose_scale=a.pose_scale, guidance=a.guidance, fps=a.fps, seed=a.seed,
        use_dmd=a.use_dmd, output_path=a.output_path, dry_run=a.dry_run,
    )
    return rc


def _cmd_chain(settings: Settings, a) -> int:
    steps = json.loads(Path(a.spec).read_text())
    if not isinstance(steps, list):
        raise SystemExit("chain spec must be a JSON list of steps")
    return chain.run_chain(
        settings, a.image, steps, a.out, size=a.size, fps=a.fps,
        work_dir=a.work_dir, dry_run=a.dry_run,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scene_studio",
                                description="One image -> video, three ways")
    sub = p.add_subparsers(dest="mode", required=True)

    # animate (Wan)
    an = sub.add_parser("animate", help="Wan 2.2 i2v: the scene comes alive (camera ~fixed)")
    an.add_argument("--image", required=True, help="start image")
    an.add_argument("--prompt", help="how the scene continues")
    an.add_argument("--size", default="1280*720", help='quote it: "1280*720" (720P) or "832*480"')
    an.add_argument("--frames", type=int, help="frame count (Wan default ~81 ≈ 5s)")
    an.add_argument("--steps", type=int, help="sampling steps")
    an.add_argument("--seed", type=int, default=-1)
    an.add_argument("--save-file", dest="save_file", help="output .mp4 (default: auto-named in WAN22_HOME)")
    an.add_argument("--no-offload", action="store_true", help="needs >80GB VRAM")
    an.add_argument("--no-convert-dtype", action="store_true")
    an.add_argument("--t5-cpu", action="store_true", dest="t5_cpu")
    an.add_argument("--dry-run", action="store_true")

    # travel (Lyra)
    tr = sub.add_parser("travel", help="Lyra 2.0: camera moves through a frozen 3D world")
    tr.add_argument("--image", required=True, help="start image")
    tr.add_argument("--motion", choices=MOTIONS, default="dolly")
    tr.add_argument("--frames", type=int, default=161)
    tr.add_argument("--prompt", help="caption of the (static) scene")
    tr.add_argument("--radius", type=float, default=3.0)
    tr.add_argument("--degrees", type=float, default=30.0)
    tr.add_argument("--elevation-deg", type=float, default=0.0, dest="elevation_deg")
    tr.add_argument("--distance", type=float, default=2.0)
    tr.add_argument("--look-dist", type=float, default=trajectory.DEFAULT_LOOK_DIST, dest="look_dist")
    tr.add_argument("--width", type=int, default=1280)
    tr.add_argument("--height", type=int, default=720)
    tr.add_argument("--fov", type=float, default=76.0)
    tr.add_argument("--linear", action="store_true", help="constant speed (default eases)")
    tr.add_argument("--keys", help="keyframes JSON (for --motion keyframes)")
    tr.add_argument("--pose-scale", type=float, default=1.1, dest="pose_scale")
    tr.add_argument("--guidance", type=float, default=5.0)
    tr.add_argument("--fps", type=int, default=16)
    tr.add_argument("--seed", type=int, default=1)
    tr.add_argument("--use-dmd", action="store_true", dest="use_dmd", help="4-step fast mode")
    tr.add_argument("--output-path", default="outputs/travel", dest="output_path")
    tr.add_argument("--dry-run", action="store_true")

    # chain
    ch = sub.add_parser("chain", help="sequence segments (e.g. travel then animate) + stitch")
    ch.add_argument("--image", required=True, help="start image for the first segment")
    ch.add_argument("--spec", required=True, help="chain spec .json (list of steps)")
    ch.add_argument("--out", required=True, help="final stitched .mp4")
    ch.add_argument("--size", default="1280*720", help="normalise all segments to this WxH")
    ch.add_argument("--fps", type=int, default=16)
    ch.add_argument("--work-dir", dest="work_dir", help="where segments are written")
    ch.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    return {"animate": _cmd_animate, "travel": _cmd_travel, "chain": _cmd_chain}[a.mode](settings, a)


if __name__ == "__main__":
    raise SystemExit(main())
