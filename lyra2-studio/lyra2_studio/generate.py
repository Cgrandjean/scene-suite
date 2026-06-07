"""Drive Lyra 2.0 inference to produce video(s) from a single image or a folder.

Two entry points:

* ``preset``  -> ``lyra2_zoomgs_inference``: image(s) + caption -> a zoom-in/zoom-out
  walkthrough. Easiest path ("just make me a video").
* ``custom``  -> ``lyra2_custom_traj_inference``: image(s) + a ``trajectory.npz``
  (see ``lyra2_studio.trajectory``) -> follows exactly the camera path you designed.

INPUTS may be a single image OR a folder of images: the inference batches over up to
``--num-samples`` of them. Pair a folder with ``--prompt-dir`` for per-image
``<stem>.txt`` captions, or give one ``--prompt`` applied to all.

QUALITY: drop ``--use-dmd`` (4-step distilled = fast but softer) to run the full
``--steps`` diffusion; keep the camera motion small (low ``--zoom-*-strength`` /
``--pose-scale``) for the sharpest, most stable result -- big moves expose occluded
regions the model has to hallucinate. ``--dry-run`` prints the command without running.

Engine flags (quality / batch / perf, shared by both modes) go BEFORE the subcommand;
mode-specific flags go after it, e.g.::

    python -m lyra2_studio.generate --steps 50 --guidance 6 preset --image x.png --prompt "..."

Anything not exposed here can be forwarded verbatim to the inference with
``--raw "--some_flag value"``.
"""

from __future__ import annotations

import argparse
import shlex
from datetime import datetime
from pathlib import Path

from .config import Settings

PRESET_MODULE = "lyra_2._src.inference.lyra2_zoomgs_inference"
CUSTOM_MODULE = "lyra_2._src.inference.lyra2_custom_traj_inference"


def _resolve(path: str, what: str) -> str:
    # The inference subprocess runs in $LYRA2_HOME, so resolve user paths to absolute
    # against the caller's cwd NOW. A folder is allowed (the inference batches over it).
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise SystemExit(f"{what} not found: {p}")
    return str(p)


def _check_frames(n: int, flag: str) -> None:
    # Lyra's autoregressive pipeline requires (num_frames - 1) to be a multiple of
    # tokens_per_step * frames_per_latent (= 80 for this model); 81/161/241 are the usual
    # valid values. Fail fast with a clear message instead of a cryptic deep assertion.
    if (n - 1) % 80:
        up = n + (80 - (n - 1) % 80)
        opts = [v for v in (up - 80, up) if v >= 81] or [81]
        raise SystemExit(
            f"{flag}={n} is invalid: Lyra needs (frames - 1) divisible by 80. "
            f"Nearest valid: {', '.join(map(str, opts))}  (valid set: 81, 161, 241, 321, ...)."
        )


def _engine_args(a: argparse.Namespace) -> list[str]:
    """Flags shared by both inference scripts (quality / batch / perf). Each is only
    forwarded when the user set it, so the inference keeps its own (mode-specific)
    defaults otherwise (e.g. steps=50 for preset, 35 for custom)."""
    out: list[str] = ["--seed", str(a.seed), "--fps", str(a.fps), "--experiment", a.experiment]
    if a.use_dmd:
        out.append("--use_dmd")
    if a.steps is not None:
        out += ["--num_sampling_step", str(a.steps)]
    if a.guidance is not None:
        out += ["--guidance", str(a.guidance)]
    if a.shift is not None:
        out += ["--shift", str(a.shift)]
    if a.resolution:
        out += ["--resolution", a.resolution]
    if a.num_samples is not None:
        out += ["--num_samples", str(a.num_samples)]
    if a.sample_start_idx is not None:
        out += ["--sample_start_idx", str(a.sample_start_idx)]
    if a.prompt_suffix:
        out += ["--prompt_suffix", a.prompt_suffix]
    if a.offload:
        out.append("--offload")
    if a.lora_paths:
        out += ["--lora_paths", *a.lora_paths]
    if a.lora_weights:
        out += ["--lora_weights", *[str(w) for w in a.lora_weights]]
    if a.raw:
        out += shlex.split(a.raw)
    return out


def run_preset(settings: Settings, a: argparse.Namespace) -> int:
    if not a.prompt and not a.prompt_dir:
        raise SystemExit('preset needs --prompt "..." or --prompt-dir <folder of *.txt>')
    _check_frames(a.num_frames_zoom_in, "--num-frames-zoom-in")
    _check_frames(a.num_frames_zoom_out, "--num-frames-zoom-out")
    args = [
        "--input_image_path", _resolve(a.image, "--image"),
        "--checkpoint_dir", settings.checkpoint_dir,
        "--output_path", a.output_path,
        "--num_frames_zoom_in", str(a.num_frames_zoom_in),
        "--num_frames_zoom_out", str(a.num_frames_zoom_out),
        "--zoom_in_strength", str(a.zoom_in_strength),
        "--zoom_out_strength", str(a.zoom_out_strength),
        "--zoom_in_direction", a.zoom_in_direction,
        "--zoom_out_direction", a.zoom_out_direction,
    ]
    if a.prompt:
        args += ["--prompt", a.prompt]
    if a.prompt_dir:
        args += ["--prompt_dir", _resolve(a.prompt_dir, "--prompt-dir")]
    args += _engine_args(a)
    rc = settings.run_module(PRESET_MODULE, args, dry_run=a.dry_run)
    if not a.dry_run and rc == 0:
        print(f"[lyra2-studio] preset videos -> {a.output_path}/videos/")
    return rc


def run_custom(settings: Settings, a: argparse.Namespace) -> int:
    _check_frames(a.num_frames, "--num-frames")
    args = [
        "--input_image_path", _resolve(a.image, "--image"),
        "--trajectory_path", _resolve(a.trajectory, "--trajectory"),
        "--checkpoint_dir", settings.checkpoint_dir,
        "--output_path", a.output_path,
        "--num_frames", str(a.num_frames),
        "--pose_scale", str(a.pose_scale),
    ]
    if a.captions:
        args += ["--captions_path", _resolve(a.captions, "--captions")]
    elif a.prompt:
        args += ["--prompt", a.prompt]
    if a.prompt_dir:
        args += ["--prompt_dir", _resolve(a.prompt_dir, "--prompt-dir")]
    args += _engine_args(a)
    rc = settings.run_module(CUSTOM_MODULE, args, dry_run=a.dry_run)
    if not a.dry_run and rc == 0:
        print(f"[lyra2-studio] custom videos -> {a.output_path}/")
    return rc


def _add_engine_args(parser: argparse.ArgumentParser) -> None:
    """Quality / batch / perf flags shared by both modes (placed on the top-level
    parser so they come before the subcommand)."""
    parser.add_argument("--dry-run", action="store_true",
                        help="print the exact inference command instead of running it")
    g = parser.add_argument_group("quality / sampling")
    g.add_argument("--use-dmd", action="store_true",
                   help="4-step DMD distillation: ~15x faster but softer. OMIT for full quality.")
    g.add_argument("--steps", type=int, default=None,
                   help="denoising steps (inference default 50 preset / 35 custom; ignored with --use-dmd)")
    g.add_argument("--guidance", type=float, default=None,
                   help="classifier-free guidance scale (default 5.0; higher = follows prompt more)")
    g.add_argument("--shift", type=float, default=None, help="flow-matching shift (default 5.0)")
    g.add_argument("--resolution", default=None,
                   help='output resolution "H,W" (default 480,832; e.g. 720,1280 = sharper, more VRAM)')
    g.add_argument("--seed", type=int, default=1, help="random seed (change to vary the result)")
    g.add_argument("--lora-paths", nargs="+", default=None,
                   help="LoRA .safetensors to apply (advanced; default = realism_boost + detail_enhancer)")
    g.add_argument("--lora-weights", nargs="+", type=float, default=None, help="blend weight per LoRA")
    b = parser.add_argument_group("batch (folder input) / misc")
    b.add_argument("--num-samples", type=int, default=None,
                   help="when --image is a FOLDER: how many images to process (default 10)")
    b.add_argument("--sample-start-idx", type=int, default=None, help="folder batch start index")
    b.add_argument("--prompt-suffix", default=None, help="text appended to every prompt")
    b.add_argument("--fps", type=int, default=16, help="output frames per second")
    b.add_argument("--offload", action="store_true",
                   help="offload modules to CPU between stages (saves VRAM, slower)")
    b.add_argument("--experiment", default="lyra2", help="inference experiment config name")
    b.add_argument("--raw", default=None,
                   help='forward arbitrary extra inference flags verbatim, e.g. --raw "--ground_plane_align"')
    b.add_argument("--no-timestamp", action="store_true",
                   help="don't append -YYYYMMDD-HHMMSS to --output-path (by default it's added so reruns don't overwrite)")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Lyra 2.0 video(s) from an image or a folder of images")
    _add_engine_args(parser)
    sub = parser.add_subparsers(dest="mode", required=True)

    p = sub.add_parser("preset", help="image(s) + caption -> zoom-in/zoom-out walkthrough")
    p.add_argument("--image", required=True, help="input image OR folder of images")
    p.add_argument("--prompt", help="caption describing the (static) scene")
    p.add_argument("--prompt-dir", help="folder of <image_stem>.txt captions instead of --prompt")
    p.add_argument("--output-path", default="outputs/zoomgs")
    p.add_argument("--num-frames-zoom-in", type=int, default=81, help="(frames-1) must be a multiple of 80: 81, 161, 241")
    p.add_argument("--num-frames-zoom-out", type=int, default=241, help="(frames-1) must be a multiple of 80: 81, 161, 241")
    p.add_argument("--zoom-in-strength", type=float, default=0.5, help="zoom-in motion magnitude (low = subtle/stable)")
    p.add_argument("--zoom-out-strength", type=float, default=1.5, help="zoom-out motion magnitude")
    p.add_argument("--zoom-in-direction", default="right", choices=["left", "right", "up", "down"])
    p.add_argument("--zoom-out-direction", default="left", choices=["left", "right", "up", "down"])

    p = sub.add_parser("custom", help="image(s) + trajectory.npz -> exact camera path")
    p.add_argument("--image", required=True, help="first-frame image OR folder")
    p.add_argument("--trajectory", required=True, help="trajectory.npz (see lyra2_studio.trajectory)")
    p.add_argument("--captions", help="captions.json (per-chunk prompts)")
    p.add_argument("--prompt", help="single caption applied to the whole clip")
    p.add_argument("--prompt-dir", help="folder of <image_stem>.txt captions")
    p.add_argument("--output-path", default="outputs/custom_traj")
    p.add_argument("--num-frames", type=int, default=161, help="(frames-1) must be a multiple of 80: 81, 161, 241; match the trajectory's --frames")
    p.add_argument("--pose-scale", type=float, default=1.1, help="scale the trajectory translation (low = subtle)")

    return parser


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    # Stamp the output dir so reruns don't overwrite previous results (opt out: --no-timestamp).
    if not a.no_timestamp:
        a.output_path = f"{a.output_path.rstrip('/')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if a.mode == "preset":
        return run_preset(settings, a)
    return run_custom(settings, a)


if __name__ == "__main__":
    raise SystemExit(main())
