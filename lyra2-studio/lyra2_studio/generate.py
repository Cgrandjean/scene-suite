"""Drive Lyra 2.0 inference to produce a video from a single image.

Two entry points:

* ``preset``  -> ``lyra2_zoomgs_inference``: easiest path. One image + one caption
  produces a zoom-in and zoom-out walkthrough. Good for "just make me a video".
* ``custom``  -> ``lyra2_custom_traj_inference``: full control. One image + a
  ``trajectory.npz`` (see ``lyra2_studio.trajectory``) + optional per-chunk
  ``captions.json`` follows exactly the camera path you designed.

Both run the official inference module inside ``$LYRA2_HOME`` via subprocess.
Use ``--dry-run`` to print the exact command without executing it (handy on a
machine without a GPU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings

PRESET_MODULE = "lyra_2._src.inference.lyra2_zoomgs_inference"
CUSTOM_MODULE = "lyra_2._src.inference.lyra2_custom_traj_inference"


def generate_preset(
    settings: Settings,
    image: str,
    *,
    prompt: str | None = None,
    prompt_dir: str | None = None,
    output_path: str = "outputs/zoomgs",
    experiment: str = "lyra2",
    num_frames_zoom_in: int = 81,
    num_frames_zoom_out: int = 241,
    zoom_in_strength: float = 0.5,
    zoom_out_strength: float = 1.5,
    fps: int = 16,
    seed: int = 1,
    use_dmd: bool = False,
    dry_run: bool = False,
) -> int:
    if not prompt and not prompt_dir:
        raise SystemExit("preset needs either --prompt \"...\" or --prompt-dir <folder of *.txt>")
    args = [
        "--input_image_path", image,
        "--experiment", experiment,
        "--checkpoint_dir", settings.checkpoint_dir,
        "--output_path", output_path,
        "--num_frames_zoom_in", str(num_frames_zoom_in),
        "--num_frames_zoom_out", str(num_frames_zoom_out),
        "--zoom_in_strength", str(zoom_in_strength),
        "--zoom_out_strength", str(zoom_out_strength),
        "--fps", str(fps),
        "--seed", str(seed),
    ]
    if prompt:
        args += ["--prompt", prompt]
    if prompt_dir:
        args += ["--prompt_dir", prompt_dir]
    if use_dmd:
        args.append("--use_dmd")
    rc = settings.run_module(PRESET_MODULE, args, dry_run=dry_run)
    if not dry_run and rc == 0:
        stem = Path(image).stem
        print(f"[lyra2-studio] combined video -> {output_path}/videos/{stem}.mp4")
    return rc


def generate_custom(
    settings: Settings,
    image: str,
    trajectory: str,
    *,
    captions: str | None = None,
    prompt: str | None = None,
    output_path: str = "outputs/custom_traj",
    experiment: str = "lyra2",
    num_frames: int = 161,
    pose_scale: float = 1.1,
    guidance: float = 5.0,
    fps: int = 16,
    seed: int = 1,
    use_dmd: bool = False,
    dry_run: bool = False,
) -> int:
    args = [
        "--input_image_path", image,
        "--trajectory_path", trajectory,
        "--experiment", experiment,
        "--checkpoint_dir", settings.checkpoint_dir,
        "--output_path", output_path,
        "--num_frames", str(num_frames),
        "--pose_scale", str(pose_scale),
        "--guidance", str(guidance),
        "--fps", str(fps),
        "--seed", str(seed),
    ]
    if captions:
        args += ["--captions_path", captions]
    elif prompt:
        args += ["--prompt", prompt]
    if use_dmd:
        args.append("--use_dmd")
    rc = settings.run_module(CUSTOM_MODULE, args, dry_run=dry_run)
    if not dry_run and rc == 0:
        stem = Path(image).stem
        print(f"[lyra2-studio] video -> {output_path}/{stem}.mp4")
    return rc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Lyra 2.0 video from a single image")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact inference command instead of running it")
    parser.add_argument("--use-dmd", action="store_true",
                        help="4-step DMD distillation: ~15x faster, slightly lower quality")
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--experiment", default="lyra2")
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("preset", help="image + caption -> zoom-in/zoom-out walkthrough")
    p.add_argument("--image", required=True, help="path to a single input image")
    p.add_argument("--prompt", help="caption describing the (static) scene")
    p.add_argument("--prompt-dir", help="folder of <image_stem>.txt captions instead of --prompt")
    p.add_argument("--output-path", default="outputs/zoomgs")
    p.add_argument("--num-frames-zoom-in", type=int, default=81)
    p.add_argument("--num-frames-zoom-out", type=int, default=241)
    p.add_argument("--zoom-in-strength", type=float, default=0.5)
    p.add_argument("--zoom-out-strength", type=float, default=1.5)

    p = sub.add_parser("custom", help="image + trajectory.npz -> camera path video")
    p.add_argument("--image", required=True, help="path to the first-frame image")
    p.add_argument("--trajectory", required=True, help="trajectory.npz (see lyra2_studio.trajectory)")
    p.add_argument("--captions", help="captions.json (per-chunk prompts)")
    p.add_argument("--prompt", help="single caption applied to the whole clip")
    p.add_argument("--output-path", default="outputs/custom_traj")
    p.add_argument("--num-frames", type=int, default=161)
    p.add_argument("--pose-scale", type=float, default=1.1)
    p.add_argument("--guidance", type=float, default=5.0)
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    common = dict(experiment=args.experiment, fps=args.fps, seed=args.seed,
                  use_dmd=args.use_dmd, dry_run=args.dry_run)
    if args.mode == "preset":
        return generate_preset(
            settings, args.image, prompt=args.prompt, prompt_dir=args.prompt_dir,
            output_path=args.output_path, num_frames_zoom_in=args.num_frames_zoom_in,
            num_frames_zoom_out=args.num_frames_zoom_out,
            zoom_in_strength=args.zoom_in_strength, zoom_out_strength=args.zoom_out_strength,
            **common,
        )
    return generate_custom(
        settings, args.image, args.trajectory, captions=args.captions, prompt=args.prompt,
        output_path=args.output_path, num_frames=args.num_frames,
        pose_scale=args.pose_scale, guidance=args.guidance, **common,
    )


if __name__ == "__main__":
    raise SystemExit(main())
