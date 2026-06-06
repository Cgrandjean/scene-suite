"""lyra2-studio: host NVIDIA Lyra 2.0 and generate camera-controlled videos from a single image.

This package is a thin orchestration layer around the official Lyra 2.0 inference
scripts (https://github.com/nv-tlabs/lyra). It does not bundle the model; it drives
a local checkout + downloaded weights on an NVIDIA GPU host.
"""

__version__ = "0.1.0"
