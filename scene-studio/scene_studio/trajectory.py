"""Build a ``trajectory.npz`` camera path for Lyra 2.0 custom-trajectory inference
(the `travel` mode). Ported from the verified lyra2-studio builder.

The .npz holds: w2c (N,4,4) f32, intrinsics (N,3,3) f32, image_height/width int64.
OpenCV camera frame (x right, y down, z forward); frame 0 = identity, so every
motion starts anchored on the input image and moves away from it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

WORLD_DOWN = np.array([0.0, 1.0, 0.0])  # +y is "down", so world-up is -y
DEFAULT_LOOK_DIST = 4.0
MOTIONS = ("orbit", "dolly", "truck", "pedestal", "pan", "tilt", "keyframes")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-8 else v / n


def look_at(eye, target, world_down=WORLD_DOWN) -> np.ndarray:
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = target - eye
    z = np.array([0.0, 0.0, 1.0]) if np.linalg.norm(z) < 1e-8 else _normalize(z)
    down = world_down
    if abs(float(np.dot(down, z))) > 0.999:
        down = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = _normalize(np.cross(down, z))
    y = np.cross(z, x)
    c2w = np.eye(4)
    c2w[:3, 0] = x
    c2w[:3, 1] = y
    c2w[:3, 2] = z
    c2w[:3, 3] = eye
    return c2w


def _w2c(c2w: np.ndarray) -> np.ndarray:
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def intrinsics_matrix(width: int, height: int, fov_deg: float) -> np.ndarray:
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _ramp(frames: int, ease: bool) -> np.ndarray:
    if frames <= 1:
        return np.zeros(max(frames, 0))
    t = np.linspace(0.0, 1.0, frames)
    return t * t * (3.0 - 2.0 * t) if ease else t  # smoothstep


def _orbit(frames, radius, degrees, elevation_deg, ease):
    t = _ramp(frames, ease)
    phi = math.radians(degrees) * t
    el = math.radians(elevation_deg) * t
    eyes = np.zeros((frames, 3))
    eyes[:, 0] = radius * np.sin(phi) * np.cos(el)
    eyes[:, 1] = -radius * np.sin(el)
    eyes[:, 2] = radius - radius * np.cos(phi) * np.cos(el)
    targets = np.tile([0.0, 0.0, radius], (frames, 1))
    return eyes, targets


def _translate(frames, axis, distance, look_dist, ease):
    t = _ramp(frames, ease)
    eyes = np.zeros((frames, 3))
    eyes[:, axis] = distance * t
    targets = eyes.copy()
    targets[:, 2] += look_dist
    return eyes, targets


def _rotate_in_place(frames, degrees, axis, look_dist, ease):
    t = _ramp(frames, ease)
    ang = math.radians(degrees) * t
    targets = np.zeros((frames, 3))
    targets[:, 2] = np.cos(ang) * look_dist
    targets[:, axis] = np.sin(ang) * look_dist
    eyes = np.zeros((frames, 3))
    return eyes, targets


def _keyframes(frames, keys, ease):
    if len(keys) < 2:
        raise ValueError("keyframes mode needs at least 2 keys")
    eyes_k = np.array([k["eye"] for k in keys], dtype=np.float64)
    tgts_k = np.array([k["target"] for k in keys], dtype=np.float64)
    t = _ramp(frames, ease)
    seg = t * (len(keys) - 1)
    i0 = np.clip(np.floor(seg).astype(int), 0, len(keys) - 2)
    frac = (seg - i0)[:, None]
    eyes = eyes_k[i0] + (eyes_k[i0 + 1] - eyes_k[i0]) * frac
    targets = tgts_k[i0] + (tgts_k[i0 + 1] - tgts_k[i0]) * frac
    return eyes, targets


def build_trajectory(
    motion: str,
    frames: int,
    *,
    width: int = 1280,
    height: int = 720,
    fov_deg: float = 76.0,
    ease: bool = True,
    radius: float = 3.0,
    degrees: float = 30.0,
    elevation_deg: float = 0.0,
    distance: float = 2.0,
    look_dist: float = DEFAULT_LOOK_DIST,
    keys: list | None = None,
) -> dict:
    if frames < 1:
        raise ValueError("frames must be >= 1")
    if motion == "orbit":
        eyes, targets = _orbit(frames, radius, degrees, elevation_deg, ease)
    elif motion == "dolly":
        eyes, targets = _translate(frames, 2, distance, look_dist, ease)
    elif motion == "truck":
        eyes, targets = _translate(frames, 0, distance, look_dist, ease)
    elif motion == "pedestal":  # +distance moves up (world-up is -y)
        eyes, targets = _translate(frames, 1, -distance, look_dist, ease)
    elif motion == "pan":
        eyes, targets = _rotate_in_place(frames, degrees, 0, look_dist, ease)
    elif motion == "tilt":  # +degrees tilts up
        eyes, targets = _rotate_in_place(frames, -degrees, 1, look_dist, ease)
    elif motion == "keyframes":
        eyes, targets = _keyframes(frames, keys or [], ease)
    else:
        raise ValueError(f"unknown motion {motion!r}; choose from {MOTIONS}")

    w2c = np.stack([_w2c(look_at(eyes[i], targets[i])) for i in range(frames)]).astype(np.float32)
    K = intrinsics_matrix(width, height, fov_deg)
    intrinsics = np.repeat(K[None], frames, axis=0).astype(np.float32)
    return {
        "w2c": w2c,
        "intrinsics": intrinsics,
        "image_height": np.int64(height),
        "image_width": np.int64(width),
    }


def save_npz(path: str | Path, traj: dict) -> None:
    np.savez(str(path), **traj)


def build_and_save(motion: str, out_path: str | Path, frames: int, **kwargs) -> Path:
    traj = build_trajectory(motion, frames, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_npz(out_path, traj)
    return out_path
