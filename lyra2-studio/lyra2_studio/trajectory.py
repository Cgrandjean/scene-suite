"""Build a ``trajectory.npz`` camera path for Lyra 2.0 custom-trajectory inference.

Lyra 2.0's ``lyra2_custom_traj_inference`` expects an ``.npz`` with:

    w2c          (N, 4, 4) float32  world-to-camera matrices
    intrinsics   (N, 3, 3) float32  pinhole intrinsics (pixels)
    image_height ()        int64    resolution the intrinsics refer to
    image_width  ()        int64

Convention (verified against the official example trajectories): OpenCV camera
frame -- x right, y down, z forward -- with frame 0 at the world origin looking
down +z. The input image is anchored to frame 0, so every preset below starts
from identity and *moves away* from it.

CLI examples:
    python -m lyra2_studio.trajectory orbit  --frames 161 --degrees 30 --radius 3 --out traj.npz
    python -m lyra2_studio.trajectory dolly  --frames 161 --distance 2          --out traj.npz
    python -m lyra2_studio.trajectory keyframes --frames 241 --keys path/to/keys.json --out traj.npz
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

# +y is "down" in this convention, so world-up is -y.
WORLD_DOWN = np.array([0.0, 1.0, 0.0])
# Distance to the look-at target for pure-translation / rotation presets.
DEFAULT_LOOK_DIST = 4.0
MOTIONS = ("orbit", "dolly", "truck", "pedestal", "pan", "tilt", "keyframes")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v if n < 1e-8 else v / n


def look_at(eye, target, world_down=WORLD_DOWN) -> np.ndarray:
    """Camera-to-world matrix (OpenCV convention) looking from ``eye`` at ``target``."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    z = target - eye
    z = np.array([0.0, 0.0, 1.0]) if np.linalg.norm(z) < 1e-8 else _normalize(z)
    down = world_down
    if abs(float(np.dot(down, z))) > 0.999:  # avoid gimbal lock looking along the down axis
        down = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = _normalize(np.cross(down, z))  # right
    y = np.cross(z, x)                 # down
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
    """Pinhole intrinsics from a horizontal field of view (square pixels)."""
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [[fx, 0.0, width / 2.0],
         [0.0, fx, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _ramp(frames: int, ease: bool) -> np.ndarray:
    if frames <= 1:
        return np.zeros(max(frames, 0))
    t = np.linspace(0.0, 1.0, frames)
    return t * t * (3.0 - 2.0 * t) if ease else t  # smoothstep


# --- motion presets: each returns (eyes (N,3), targets (N,3)) ----------------

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
    """Return a dict with keys ``w2c``, ``intrinsics``, ``image_height``, ``image_width``."""
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


def camera_centers(w2c: np.ndarray) -> np.ndarray:
    """World-space camera positions, for a quick sanity print."""
    centers = np.empty((w2c.shape[0], 3), dtype=np.float64)
    for i in range(w2c.shape[0]):
        R = w2c[i, :3, :3]
        t = w2c[i, :3, 3]
        centers[i] = -R.T @ t
    return centers


# --- CLI ---------------------------------------------------------------------

def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--frames", type=int, default=161, help="number of camera poses")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", type=float, default=76.0, help="horizontal field of view (deg)")
    p.add_argument("--linear", action="store_true", help="constant speed (default eases in/out)")
    p.add_argument("--look-dist", type=float, default=DEFAULT_LOOK_DIST)
    p.add_argument("--out", required=True, help="output .npz path")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build a trajectory.npz for Lyra 2.0")
    sub = parser.add_subparsers(dest="motion", required=True)

    p = sub.add_parser("orbit", help="arc around a point in front of the camera")
    _add_common(p)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--degrees", type=float, default=30.0)
    p.add_argument("--elevation-deg", type=float, default=0.0)

    for name, helptext in (("dolly", "move forward (+) / back (-)"),
                           ("truck", "slide right (+) / left (-)"),
                           ("pedestal", "move up (+) / down (-)")):
        p = sub.add_parser(name, help=helptext)
        _add_common(p)
        p.add_argument("--distance", type=float, default=2.0)

    for name, helptext in (("pan", "rotate yaw in place"),
                           ("tilt", "rotate pitch in place (+ up)")):
        p = sub.add_parser(name, help=helptext)
        _add_common(p)
        p.add_argument("--degrees", type=float, default=30.0)

    p = sub.add_parser("keyframes", help="interpolate eye/target keyframes from JSON")
    _add_common(p)
    p.add_argument("--keys", required=True,
                   help='JSON file: [{"eye":[x,y,z],"target":[x,y,z]}, ...]')

    args = parser.parse_args(argv)

    kwargs = dict(
        frames=args.frames, width=args.width, height=args.height,
        fov_deg=args.fov, ease=not args.linear, look_dist=args.look_dist,
    )
    if args.motion == "orbit":
        kwargs.update(radius=args.radius, degrees=args.degrees, elevation_deg=args.elevation_deg)
    elif args.motion in ("dolly", "truck", "pedestal"):
        kwargs.update(distance=args.distance)
    elif args.motion in ("pan", "tilt"):
        kwargs.update(degrees=args.degrees)
    elif args.motion == "keyframes":
        kwargs.update(keys=json.loads(Path(args.keys).read_text()))

    traj = build_trajectory(args.motion, **kwargs)
    save_npz(args.out, traj)

    centers = camera_centers(traj["w2c"])
    frame0_identity = bool(np.allclose(traj["w2c"][0], np.eye(4), atol=1e-4))
    print(f"wrote {args.out}")
    print(f"  motion={args.motion} frames={traj['w2c'].shape[0]} "
          f"resolution={int(traj['image_width'])}x{int(traj['image_height'])}")
    print(f"  frame0 == identity: {frame0_identity}")
    print(f"  camera start={np.round(centers[0], 3).tolist()} "
          f"end={np.round(centers[-1], 3).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
