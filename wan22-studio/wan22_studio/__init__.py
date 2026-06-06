"""wan22-studio: drive NVIDIA-class GPUs to run Wan 2.2 image-to-video.

A thin orchestration layer around the official Wan 2.2 inference code
(https://github.com/Wan-Video/Wan2.2). It does not bundle the model; it installs
it, downloads weights, and runs image-to-video generation -- start from one image
(e.g. a Google Maps / Street View frame) and let the scene continue as a video.
"""

__version__ = "0.1.0"
