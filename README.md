# scene-suite

Turn **one image into video, three ways**, on a single GPU. An orchestrator plus two
model backends — clone it on a fresh GPU pod and run one command.

| Folder | What it is |
|--------|------------|
| **[scene-studio/](scene-studio/)** | the orchestrator + CLI you use (`animate` / `travel` / `chain`). **Start here.** |
| **[wan22-studio/](wan22-studio/)** | Wan 2.2 image-to-video backend → mode `animate` (scene comes alive). Apache-2.0. |
| **[lyra2-studio/](lyra2-studio/)** | Lyra 2.0 backend → mode `travel` (camera moves through a frozen scene). Weights non-commercial. |

## Quickstart on a fresh GPU pod (A100 80GB)

```bash
git clone https://github.com/Cgrandjean/scene-suite.git && cd scene-suite
bash scene-studio/scripts/deploy.sh          # Wan only  -> `animate`  (fast,  ~70 GB)
# bash scene-studio/scripts/deploy.sh --all   # + Lyra     -> `travel`/`chain` (~+97 GB)
```

`deploy.sh` builds the backend env(s), downloads the weights (in parallel), installs
the orchestrator, wires `~/scene_env.sh`, and runs a smoke test. Then:

```bash
source ~/scene_env.sh
scene-studio animate --image my.jpg --prompt "the scene comes to life, cinematic" --save-file out.mp4
```

Full usage: **[scene-studio/README.md](scene-studio/README.md)**.

## Modes

- **`animate`** — the scene comes alive, camera ~fixed (Wan 2.2 i2v).
- **`travel`** — the camera moves through a frozen 3D world (Lyra 2.0).
- **`chain`** — sequence segments (e.g. travel → animate) and stitch with ffmpeg.
- **`move-alive`** *(planned)* — camera move *and* living scene in one shot (Wan-Fun-Camera).
