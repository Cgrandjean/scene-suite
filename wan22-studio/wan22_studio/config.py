"""Locations and environment needed to drive a local Wan 2.2 checkout."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Settings:
    wan_home: Path      # the cloned `Wan2.2` directory (contains generate.py)
    ckpt_dir: str       # checkpoint dir, relative to wan_home or absolute (e.g. "Wan2.2-I2V-A14B")
    output_dir: Path    # where generated videos are written
    python_bin: str     # python interpreter that has the Wan 2.2 deps installed

    @classmethod
    def from_env(cls) -> "Settings":
        home = os.environ.get("WAN22_HOME")
        if not home:
            raise SystemExit(
                "WAN22_HOME is not set.\n"
                "Point it at the Wan2.2 repo you cloned, e.g.:\n"
                "  export WAN22_HOME=$HOME/wan-src/Wan2.2"
            )
        home_path = Path(home).expanduser().resolve()
        if not (home_path / "generate.py").is_file():
            raise SystemExit(
                f"WAN22_HOME={home_path} does not look like a Wan2.2 checkout "
                "(no generate.py found inside)."
            )
        return cls(
            wan_home=home_path,
            ckpt_dir=os.environ.get("WAN22_CKPT_DIR", "Wan2.2-I2V-A14B"),
            output_dir=Path(os.environ.get("WAN22_OUTPUT_DIR", "outputs")),
            python_bin=os.environ.get("WAN22_PYTHON", sys.executable),
        )

    def subprocess_env(self) -> dict:
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        return env

    def run_generate(self, args: list[str], *, dry_run: bool = False) -> int:
        """Run Wan 2.2's generate.py from $WAN22_HOME, or print it with dry_run."""
        cmd = [self.python_bin, "generate.py", *args]
        if dry_run:
            printable = " ".join(shlex.quote(c) for c in cmd)
            print(
                f"cd {shlex.quote(str(self.wan_home))} && "
                f"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\\n  {printable}"
            )
            return 0
        print(f"[wan22-studio] running Wan 2.2 generate.py in {self.wan_home}", file=sys.stderr)
        return subprocess.run(cmd, cwd=str(self.wan_home), env=self.subprocess_env()).returncode
