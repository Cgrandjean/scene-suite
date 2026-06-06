"""Reconstruct an explorable 3D Gaussian scene from a generated walkthrough video.

Wraps ``lyra_2._src.inference.vipe_da3_gs_recon``. Given a video produced by
``generate``, it estimates camera poses + depth and fits 3D Gaussian Splats,
writing a ``.ply`` point cloud and a fly-through ``.mp4``.
"""

from __future__ import annotations

import argparse

from .config import Settings

RECON_MODULE = "lyra_2._src.inference.vipe_da3_gs_recon"


def reconstruct(settings: Settings, video_path: str, *, dry_run: bool = False) -> int:
    return settings.run_module(RECON_MODULE, ["--input_video_path", video_path], dry_run=dry_run)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct 3D Gaussians from a Lyra 2.0 video")
    parser.add_argument("--video", required=True, help="path to a generated .mp4")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    return reconstruct(Settings.from_env(), args.video, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
