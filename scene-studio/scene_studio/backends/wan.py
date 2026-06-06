"""`animate` backend: Wan 2.2 image-to-video -- the scene comes alive while the
camera stays roughly fixed."""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, run_in

_ENV = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}


def animate(
    settings: Settings,
    image: str,
    *,
    prompt: str | None = None,
    size: str = "1280*720",
    frames: int | None = None,
    steps: int | None = None,
    seed: int = -1,
    offload: bool = True,
    convert_dtype: bool = True,
    t5_cpu: bool = False,
    save_file: str | None = None,
    dry_run: bool = False,
):
    """Run Wan 2.2 i2v-A14B. Returns (returncode, video_path_or_None).

    Defaults target a single A100 80GB (offload + dtype conversion on).
    """
    b = settings.require("wan")
    args = [
        "generate.py",
        "--task", "i2v-A14B",
        "--size", size,
        "--ckpt_dir", b.ckpt_dir,
        "--image", image,
    ]
    if prompt:
        args += ["--prompt", prompt]
    if offload:
        args += ["--offload_model", "True"]
    if convert_dtype:
        args.append("--convert_model_dtype")
    if t5_cpu:
        args.append("--t5_cpu")
    if frames is not None:
        args += ["--frame_num", str(frames)]
    if steps is not None:
        args += ["--sample_steps", str(steps)]
    args += ["--base_seed", str(seed)]
    if save_file:
        args += ["--save_file", save_file]
    rc = run_in(b, args, extra_env=_ENV, label="Wan animate", dry_run=dry_run)
    return rc, (Path(save_file) if save_file else None)
