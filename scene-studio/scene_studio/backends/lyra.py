"""`travel` backend: Lyra 2.0 custom-trajectory inference -- the camera moves
through a frozen 3D world reconstructed from the input image."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, run_in

CUSTOM_MODULE = "lyra_2._src.inference.lyra2_custom_traj_inference"
_ENV = {"PYTHONPATH": ".", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def _predicted_output(home: Path, output_path: str, stem: str) -> Path:
    base = Path(output_path)
    if not base.is_absolute():
        base = home / base
    return base / f"{stem}.mp4"


def travel(
    settings: Settings,
    image: str,
    trajectory: str,
    *,
    prompt: str | None = None,
    num_frames: int = 161,
    pose_scale: float = 1.1,
    guidance: float = 5.0,
    fps: int = 16,
    seed: int = 1,
    use_dmd: bool = False,
    output_path: str = "outputs/travel",
    dry_run: bool = False,
):
    """Run Lyra custom-trajectory i2v. Returns (returncode, predicted_video_path)."""
    b = settings.require("lyra")
    args = [
        "-m", CUSTOM_MODULE,
        "--input_image_path", image,
        "--trajectory_path", trajectory,
        "--experiment", "lyra2",
        "--checkpoint_dir", b.ckpt_dir,
        "--output_path", output_path,
        "--num_frames", str(num_frames),
        "--pose_scale", str(pose_scale),
        "--guidance", str(guidance),
        "--fps", str(fps),
        "--seed", str(seed),
    ]
    if prompt:
        args += ["--prompt", prompt]
    if use_dmd:
        args.append("--use_dmd")
    rc = run_in(b, args, extra_env=_ENV, label="Lyra travel", dry_run=dry_run)
    return rc, _predicted_output(b.home, output_path, Path(image).stem)
