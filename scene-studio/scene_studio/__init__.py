"""scene-studio: turn one image into video, three ways, from a single CLI.

Modes:
  * animate     -> Wan 2.2 i2v: the scene comes alive, camera ~fixed
  * travel      -> Lyra 2.0:     the camera moves through a frozen 3D world
  * chain       -> stitch segments (e.g. travel then animate) with ffmpeg

Each model runs in its own conda env on the GPU host; this package is a thin
orchestrator that dispatches to the right backend. (A 4th mode, `move-alive`
= camera move + living scene in one shot via Wan-Fun-Camera, is planned.)
"""

__version__ = "0.1.0"
