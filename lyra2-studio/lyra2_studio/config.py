"""Locations and environment needed to drive a local Lyra 2.0 checkout."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    lyra2_home: Path      # the cloned `Lyra-2` directory (contains the `lyra_2` package)
    checkpoint_dir: str   # checkpoint dir, relative to lyra2_home (e.g. "checkpoints/model")
    output_dir: Path      # where generated videos are written
    python_bin: str       # python interpreter that has the `lyra2` env installed

    @classmethod
    def from_env(cls) -> "Settings":
        home = os.environ.get("LYRA2_HOME")
        if not home:
            raise SystemExit(
                "LYRA2_HOME is not set.\n"
                "Point it at the `Lyra-2` directory you cloned, e.g.:\n"
                "  export LYRA2_HOME=$HOME/lyra/Lyra-2"
            )
        home_path = Path(home).expanduser().resolve()
        if not (home_path / "lyra_2").is_dir():
            raise SystemExit(
                f"LYRA2_HOME={home_path} does not look like a Lyra-2 checkout "
                "(no `lyra_2/` package found inside)."
            )
        # Prefer the lyra2 env's interpreter for the inference subprocess. Launching the
        # wrapper from `base`/system python is an easy trap (e.g. `conda activate lyra2`
        # silently fails when conda isn't on PATH), yielding a cryptic "No module named
        # 'cv2'". Falling back to the env's python makes it work regardless.
        python_bin = os.environ.get("LYRA2_PYTHON")
        if not python_bin:
            env_name = os.environ.get("LYRA2_ENV", "lyra2")
            cand = Path.home() / "miniconda3" / "envs" / env_name / "bin" / "python"
            python_bin = str(cand) if cand.is_file() else sys.executable
        return cls(
            lyra2_home=home_path,
            checkpoint_dir=os.environ.get("LYRA2_CHECKPOINT_DIR", "checkpoints/model"),
            output_dir=Path(os.environ.get("LYRA2_OUTPUT_DIR", "outputs")),
            python_bin=python_bin,
        )

    def subprocess_env(self) -> dict:
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        # Lyra's modules are imported relative to its own root.
        env["PYTHONPATH"] = "." if not existing else "." + os.pathsep + existing
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        # Quieten log spam: libtorch C++ warnings (the per-conv "Could not initialize
        # NNPACK" flood on server CPUs -- harmless, NNPACK just isn't used) and the
        # Hugging Face tokenizers fork warning. setdefault so the user can override.
        env.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        return env

    def run_module(self, module: str, args: list[str], *, dry_run: bool = False) -> int:
        """Run a Lyra 2.0 inference module from $LYRA2_HOME, or print it with dry_run."""
        cmd = [self.python_bin, "-m", module, *args]
        if dry_run:
            printable = " ".join(shlex.quote(c) for c in cmd)
            print(
                f"cd {shlex.quote(str(self.lyra2_home))} && "
                f"PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n  {printable}"
            )
            return 0
        print(f"[lyra2-studio] running {module} in {self.lyra2_home}", file=sys.stderr)
        return subprocess.run(cmd, cwd=str(self.lyra2_home), env=self.subprocess_env()).returncode
