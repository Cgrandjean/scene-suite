"""Unified settings: scene-studio drives a Lyra 2.0 checkout and a Wan 2.2 checkout,
each in its own conda env, from one place.

Per backend you set three env vars (prefix LYRA2_ or WAN22_):
  <PREFIX>_HOME        the cloned repo
  <PREFIX>_PYTHON      that backend's conda-env python (default: current interpreter)
  <PREFIX>_CKPT_DIR    checkpoint dir (has a sensible default)

A backend is simply "absent" if its _HOME is unset, so you can install only the
backends you need (e.g. just Wan for `animate`).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Backend:
    name: str
    home: Path
    python_bin: str
    ckpt_dir: str


def _backend(prefix: str, marker: str, default_ckpt: str) -> Backend | None:
    home = os.environ.get(f"{prefix}_HOME")
    if not home:
        return None
    home_path = Path(home).expanduser().resolve()
    target = home_path / marker
    if not target.exists():
        raise SystemExit(
            f"{prefix}_HOME={home_path} does not look like the expected checkout "
            f"(missing `{marker}`)."
        )
    return Backend(
        name=prefix,
        home=home_path,
        python_bin=os.environ.get(f"{prefix}_PYTHON", sys.executable),
        ckpt_dir=os.environ.get(f"{prefix}_CKPT_DIR", default_ckpt),
    )


@dataclass
class Settings:
    lyra: Backend | None
    wan: Backend | None
    output_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            lyra=_backend("LYRA2", marker="lyra_2", default_ckpt="checkpoints/model"),
            wan=_backend("WAN22", marker="generate.py", default_ckpt="Wan2.2-I2V-A14B"),
            output_dir=Path(os.environ.get("SCENE_OUTPUT_DIR", "outputs")).expanduser(),
        )

    def require(self, which: str) -> Backend:
        backend = getattr(self, which)
        if backend is None:
            prefix = {"lyra": "LYRA2", "wan": "WAN22"}[which]
            raise SystemExit(
                f"This mode needs the {which} backend, but {prefix}_HOME is not set.\n"
                f"  export {prefix}_HOME=/path/to/checkout   (and {prefix}_PYTHON for its conda env)"
            )
        return backend


def run_in(
    backend: Backend,
    args: list[str],
    *,
    extra_env: dict | None = None,
    label: str = "",
    dry_run: bool = False,
) -> int:
    """Run `<python> <args...>` from the backend's repo dir, or print it (dry_run)."""
    cmd = [backend.python_bin, *args]
    if dry_run:
        envstr = "".join(f"{k}={shlex.quote(v)} " for k, v in (extra_env or {}).items())
        printable = " ".join(shlex.quote(c) for c in cmd)
        print(f"cd {shlex.quote(str(backend.home))} && {envstr}\\\n  {printable}")
        return 0
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[scene-studio] {label or backend.name} in {backend.home}", file=sys.stderr)
    return subprocess.run(cmd, cwd=str(backend.home), env=env).returncode
