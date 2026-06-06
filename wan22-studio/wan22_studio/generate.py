"""Drive Wan 2.2 image-to-video: start from one image and continue the scene.

Wraps the official Wan2.2 ``generate.py`` ``i2v-A14B`` task. You give it a single
image (e.g. a Google Maps / Street View frame) and an optional prompt describing
how the scene should continue; it produces a short video.

Defaults target a single A100 80GB: ``--offload_model True`` + ``--convert_model_dtype``
are on (the validated 80GB recipe). On a bigger card you can pass ``--no-offload``
for speed; on a tighter one add ``--t5-cpu``.

Use ``--dry-run`` to print the exact underlying command without running it.
"""

from __future__ import annotations

import argparse

from .config import Settings

TASK = "i2v-A14B"


def generate_i2v(
    settings: Settings,
    image: str,
    *,
    prompt: str | None = None,
    task: str = TASK,
    size: str = "1280*720",
    frames: int | None = None,
    steps: int | None = None,
    guide_scale: float | None = None,
    shift: float | None = None,
    seed: int = -1,
    offload: bool = True,
    convert_dtype: bool = True,
    t5_cpu: bool = False,
    prompt_extend: bool = False,
    prompt_lang: str = "en",
    save_file: str | None = None,
    dry_run: bool = False,
) -> int:
    args = [
        "--task", task,
        "--size", size,
        "--ckpt_dir", settings.ckpt_dir,
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
    if guide_scale is not None:
        args += ["--sample_guide_scale", str(guide_scale)]
    if shift is not None:
        args += ["--sample_shift", str(shift)]
    if seed is not None:
        args += ["--base_seed", str(seed)]
    if prompt_extend:
        args += ["--use_prompt_extend", "--prompt_extend_target_lang", prompt_lang]
    if save_file:
        args += ["--save_file", save_file]

    rc = settings.run_generate(args, dry_run=dry_run)
    if not dry_run and rc == 0:
        where = save_file or f"{settings.wan_home} (auto-named {task}_*.mp4)"
        print(f"[wan22-studio] video -> {where}")
    return rc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Wan 2.2 image-to-video: continue a scene from one image"
    )
    p.add_argument("--image", required=True, help="path to the start image (your Maps/Street View frame)")
    p.add_argument("--prompt", help="optional text describing how the scene continues")
    p.add_argument("--size", default="1280*720",
                   help='resolution area, quote it: "1280*720" (720P) or "832*480" (480P)')
    p.add_argument("--task", default=TASK, help="Wan task (default i2v-A14B)")
    p.add_argument("--frames", type=int, help="frame count (Wan default ~81 ≈ 5s @ 16fps)")
    p.add_argument("--steps", type=int, help="sampling steps (more = better/slower)")
    p.add_argument("--guide-scale", type=float, dest="guide_scale", help="CFG guidance scale")
    p.add_argument("--shift", type=float, help="sampling shift")
    p.add_argument("--seed", type=int, default=-1, help="-1 = random")
    p.add_argument("--save-file", dest="save_file",
                   help="output .mp4 path (default: auto-named inside $WAN22_HOME)")
    p.add_argument("--no-offload", action="store_true",
                   help="disable model offload (only if you have >80GB VRAM)")
    p.add_argument("--no-convert-dtype", action="store_true", help="disable --convert_model_dtype")
    p.add_argument("--t5-cpu", action="store_true", dest="t5_cpu",
                   help="keep the T5 text encoder on CPU (saves VRAM)")
    p.add_argument("--prompt-extend", action="store_true", dest="prompt_extend",
                   help="expand the prompt with a local Qwen model (better quality, extra download)")
    p.add_argument("--prompt-lang", default="en", dest="prompt_lang",
                   help="target language for --prompt-extend (default en)")
    p.add_argument("--dry-run", action="store_true", help="print the command instead of running it")
    return p


def main(argv=None) -> int:
    a = _build_parser().parse_args(argv)
    settings = Settings.from_env()
    return generate_i2v(
        settings, a.image, prompt=a.prompt, task=a.task, size=a.size,
        frames=a.frames, steps=a.steps, guide_scale=a.guide_scale, shift=a.shift,
        seed=a.seed, offload=not a.no_offload, convert_dtype=not a.no_convert_dtype,
        t5_cpu=a.t5_cpu, prompt_extend=a.prompt_extend, prompt_lang=a.prompt_lang,
        save_file=a.save_file, dry_run=a.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
