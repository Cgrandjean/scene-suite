"""locate-studio: open-vocabulary object detection on images and videos with
NVIDIA LocateAnything-3B (a vision-language grounding model).

Public API:
  * LocateAnythingWorker -- loads the model once, runs detect/ground/point queries.
  * detect (module)      -- CLI: image or video -> annotated output + JSON boxes.
"""

from .worker import LocateAnythingWorker

__all__ = ["LocateAnythingWorker"]
