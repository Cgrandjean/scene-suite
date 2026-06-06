# scene-studio

Turn **one image into video, three ways**, from a single CLI — on a single GPU.

`scene-studio` is a thin **orchestrator**. It doesn't reimplement any model; it
drives two installed backends (and soon a third), each in its own conda env:

| Mode | What you get | Backend | Status |
|------|--------------|---------|:------:|
| `animate` | the **scene comes alive**, camera ~fixed | [Wan 2.2 i2v](https://github.com/Wan-Video/Wan2.2) | ✅ |
| `travel` | the **camera moves** through a frozen 3D world | [Lyra 2.0](https://github.com/nv-tlabs/lyra) | ✅ |
| `chain` | **sequence** segments (e.g. travel → animate) + stitch | both + ffmpeg | ✅ |
| `move-alive` | camera move **and** living scene, one shot | Wan-Fun-Camera | ⏳ phase 2 |

> **The mental model:** `animate` = press play on a photo. `travel` = walk around
> inside a frozen photo. `chain` = glue shots together. `move-alive` (coming) = both
> at once.

---

## Why two environments

Lyra and Wan need **incompatible dependency stacks** (different torch / flash-attn),
so they can't share one conda env. scene-studio keeps them separate and dispatches
each mode to the right one. You point it at each backend with env vars:

```
WAN22_HOME, WAN22_PYTHON, WAN22_CKPT_DIR       # for `animate`
LYRA2_HOME, LYRA2_PYTHON, LYRA2_CKPT_DIR       # for `travel`
SCENE_OUTPUT_DIR
```

A backend whose `_HOME` is unset is simply unavailable — so you can install only
Wan (for `animate`) and add Lyra later.

---

## Deploy on a fresh GPU pod (turnkey)

Copy the three repo folders onto the pod **as siblings** (one rsync), then run **one
command**:

```bash
# from your laptop -- copy all repos to the pod (no git needed)
rsync -avz --exclude '.git' --exclude '__pycache__' --exclude outputs \
  -e "ssh -p <PORT> -i ~/.ssh/lium" \
  ~/Documents/Code/scene-studio ~/Documents/Code/wan22-studio ~/Documents/Code/lyra2-studio \
  root@<HOST>:~/

# on the pod -- one command does everything
bash ~/scene-studio/scripts/deploy.sh          # Wan only -> `animate`  (fast,  ~70 GB)
bash ~/scene-studio/scripts/deploy.sh --all    # + Lyra    -> `travel`/`chain` (~+97 GB, slower)
```

`deploy.sh` installs Miniconda if missing, builds the backend env(s), downloads the
weights (in parallel with the build), installs the orchestrator, wires the env vars
into `~/scene_env.sh`, and verifies the wiring. The Wan smoke test leaves a real test
clip at `$WAN22_HOME/outputs/smoke.mp4`.

**On each new login:** `source ~/scene_env.sh` — then the `scene-studio` command and
`python -m scene_studio` both work from anywhere.

<details><summary>Manual / per-backend install</summary>

Install each backend via its own repo (`wan22-studio`, `lyra2-studio`) →
`scripts/bootstrap.sh`, then wire scene-studio to them:

```bash
pip install -e .                         # orchestrator (numpy; needs ffmpeg on PATH)
eval "$(bash scripts/print_env.sh)"      # exports WAN22_* / LYRA2_* from your conda envs
```
</details>

---

## Use it

### animate — the scene comes alive (Wan)

```bash
python -m scene_studio animate \
  --image manama_street.jpg \
  --prompt "gentle traffic and pedestrians move, a flag sways, warm cinematic light" \
  --size "1280*720" --save-file outputs/manama_alive.mp4
```

### travel — move through a frozen scene (Lyra)

```bash
python -m scene_studio travel \
  --image manama_street.jpg --motion dolly --distance 2 --frames 161 \
  --prompt "a quiet street, every detail still" --use-dmd
```

Motions: `orbit dolly truck pedestal pan tilt keyframes` (same knobs as before:
`--degrees --radius --elevation-deg --distance --look-dist`, `--keys file.json`).

### chain — travel then animate, stitched into one clip

```bash
python -m scene_studio chain \
  --image manama_street.jpg \
  --spec examples/chain_travel_then_animate.json \
  --out outputs/manama_chain.mp4 --size "1280*720" --fps 16
```

The chain runs each segment, feeds the **last frame** of one as the start image of
the next, normalises everything to one size/fps and concatenates. Edit the JSON to
design your own sequence (see `examples/chain_travel_then_animate.json`).

> Every mode takes `--dry-run` to print the exact backend command(s) without running.

---

## Honest notes

- **`chain` seams**: Lyra and Wan have different looks; the cut between a `travel`
  and an `animate` segment can be visible. A color-grade pass helps. And it's
  *move-then-alive*, never both at once — that's what `move-alive` (phase 2) is for.
- **Quality ∝ source image**: a sharp, clean frame beats a low-res Maps/Street-View
  grab by a mile.
- **License**: `animate` (Wan) is Apache-2.0, commercial OK. `travel` (Lyra weights)
  is **non-commercial / research only**. See [LICENSE](LICENSE).
- **VRAM**: one mode runs at a time → a single A100 80GB is enough.

## Roadmap

- **Phase 2 — `move-alive`**: a third backend (`Wan2.2-Fun-Camera-Control` via
  [VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)) for camera motion *and* a
  living scene in a single shot — the true "move through a living world".
