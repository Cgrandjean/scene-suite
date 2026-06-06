# lyra2-studio

Run **NVIDIA Lyra 2.0** on a single GPU and use it to generate camera-controlled
videos from one image — through a small command-line tool.

This repo is a thin **orchestration layer** around the official inference code
([nv-tlabs/lyra](https://github.com/nv-tlabs/lyra)). It does not reimplement the
model; it automates installing it, downloading weights, building camera
trajectories, and running generation.

---

## Read this first

**What Lyra 2.0 actually does.** It is *not* a text-to-video model. It takes **one
input image**, builds a persistent, explorable 3D world from it, and renders a
**video walkthrough along a camera path** (and optionally reconstructs 3D Gaussian
Splats). So "a video of your choice" means **your image + your camera move + your
caption** — not an arbitrary scene conjured from text.

**Hardware.** Lyra 2.0 is a 14B model (built on WAN-2.1) and needs an **NVIDIA GPU
on Linux** — it does **not** run on a Mac or on CPU. It loads in bf16 and wants
**~43 GB of VRAM**, so the practical target is a single **80 GB card: an A100 80GB
is plenty** (also fine on H100/H200; an A100 40GB is below the floor).

**Timing.** NVIDIA's published figure on one H100 80GB is ~9 min per 80 frames at
full quality, or **~35 s with the fast 4-step `--use-dmd` mode** (~15× faster). An
A100 80GB runs the same model unchanged, just slower per step — perfectly fine when
you're not racing a clock. Reach for `--use-dmd` when you want quick iterations.

**Disk.** The `nvidia/Lyra-2.0` weights are **~97 GB** (the 14B diffusion model is
~68 GB sharded; plus T5 text encoder ~11 GB, recon model ~13 GB, encoders/VAE/LoRA).
Runtime also pulls the DA3 depth model (~6.8 GB) and MoGe (~1 GB). **Provision
≥ 200 GB of disk** (weights ~105 GB + conda env ~15-20 GB + generated videos +
headroom). Note this is *disk*, not VRAM — loaded, the model needs far less.

**License (important).** The Lyra 2.0 *code* is Apache-2.0, but the *weights* are
under NVIDIA's **Internal Scientific Research and Development Model License —
non-commercial, R&D use only**. Do not deploy this as a production or commercial
service or use it to generate works for sale. See [LICENSE](LICENSE). The wrapper
code in this repo is MIT.

---

## What's in here

```
lyra2_studio/        Python package (the wrapper)
  config.py            paths/env (LYRA2_HOME, checkpoints, interpreter)
  trajectory.py        build trajectory.npz camera paths (orbit/dolly/pan/keyframes/…)
  generate.py          CLI: `preset` and `custom` generation
  reconstruct.py       CLI: video → 3D Gaussian Splats
scripts/
  setup_host.sh        install Lyra 2.0 on the GPU host (mirrors official INSTALL.md)
  download_weights.sh  fetch nvidia/Lyra-2.0 checkpoints from Hugging Face
  smoke_test.sh        sanity-check the install on a bundled sample
examples/keyframes.json  sample camera keyframes
```

---

## Quickstart (on the A100 host)

First get this repo onto the box. If you don't push it to a git remote, just copy
the folder up over SSH (works with the SSH access cloud GPU rentals give you):

```bash
# from your laptop — HOST:PORT come from the rental dashboard
rsync -avz -e "ssh -p <PORT>" ./lyra2-studio/ <user>@<HOST>:~/lyra2-studio/
```

**Fastest path — one command (unattended).** `bootstrap.sh` installs Miniconda if
missing, downloads the ~97 GB of weights *in parallel* with the env build, uses all
CPU cores for the compiles (RAM-capped), skips the recon-only `vipe` build, then
smoke-tests:

```bash
# on the box
export HF_TOKEN=hf_xxx          # accept the license at huggingface.co/nvidia/Lyra-2.0 first
cd ~/lyra2-studio
bash scripts/bootstrap.sh       # ~35–50 min, dominated by the flash-attn + TE compiles
```

Most of that time is two unavoidable from-source compiles (flash-attn, transformer-engine
— no prebuilt wheels exist for Lyra's pinned torch 2.7 / CUDA 12.8). It's a **one-time**
cost: actually generating a video afterwards takes seconds-to-minutes with `--use-dmd`.
This fast path leaves `reconstruct.py` (3D splats) unavailable — set `LYRA2_SKIP_RECON=0`
before running if you want it.

Or the same thing step by step, if you prefer to watch each stage:

```bash
pip install -r requirements.txt

# 1. Install Lyra 2.0 (clones nv-tlabs/lyra, builds the conda env). ~30–60 min.
bash scripts/setup_host.sh ~/lyra-src

# 2. Point the tools at the checkout and grab the weights (research license)
conda activate lyra2
export LYRA2_HOME=~/lyra-src/lyra/Lyra-2
huggingface-cli login        # accept the model license on the HF model page first
bash scripts/download_weights.sh

# 3. Confirm it works (fast DMD mode)
bash scripts/smoke_test.sh
```

`LYRA2_HOME` is the only required setting. Optional overrides:
`LYRA2_CHECKPOINT_DIR` (default `checkpoints/model`), `LYRA2_OUTPUT_DIR`,
`LYRA2_PYTHON`.

You drive everything from a shell on the A100 box (SSH in). Every CLI accepts
`--dry-run` to print the exact underlying command without running it — handy to
inspect a command from your laptop before launching it on the GPU.

---

## Generate a video of your choice

### A) Preset — easiest (image + caption → zoom in/out walkthrough)

```bash
python -m lyra2_studio.generate --use-dmd preset \
  --image my_photo.jpg \
  --prompt "A quiet cobblestone street at golden hour, every detail frozen and still." \
  --output-path outputs/street
# → $LYRA2_HOME/outputs/street/videos/my_photo.mp4
```

> Tip: describe the scene as a *single still photo*. Lyra holds the scene static
> and moves the camera; prompts that imply motion (people walking, water flowing)
> work against it.

### B) Custom — you choose the camera move

Build a trajectory, then run `custom`:

```bash
# 30° orbit over 161 frames
python -m lyra2_studio.trajectory orbit --frames 161 --degrees 30 --radius 3 --out orbit.npz

python -m lyra2_studio.generate --use-dmd custom \
  --image my_photo.jpg \
  --trajectory orbit.npz \
  --prompt "A quiet cobblestone street at golden hour." \
  --num-frames 161 \
  --output-path outputs/street_orbit
# → $LYRA2_HOME/outputs/street_orbit/my_photo.mp4
```

Available motions (all start anchored on your input image and move away from it):

| motion     | knobs                          | effect                          |
|------------|--------------------------------|---------------------------------|
| `orbit`    | `--degrees --radius --elevation-deg` | arc around a point in front |
| `dolly`    | `--distance`                   | push in (+) / pull back (−)     |
| `truck`    | `--distance`                   | slide right (+) / left (−)      |
| `pedestal` | `--distance`                   | rise (+) / descend (−)          |
| `pan`      | `--degrees`                    | rotate yaw in place             |
| `tilt`     | `--degrees`                    | rotate pitch in place (+ up)    |
| `keyframes`| `--keys file.json`             | interpolate eye/target keyframes|

Keyframe path example:

```bash
python -m lyra2_studio.trajectory keyframes --frames 241 \
  --keys examples/keyframes.json --out path.npz
```

**Per-shot prompts** that change as the camera travels (instead of one `--prompt`):
hand-write a small `captions.json` mapping a *frame index* to the caption that
takes effect from that frame, then pass it with `--captions`:

```json
{ "0": "a wide establishing shot of the hall", "160": "closer, emphasising the doorway" }
```

```bash
python -m lyra2_studio.generate custom --image p.jpg --trajectory orbit.npz \
  --captions captions.json --num-frames 241 --output-path outputs/p
```

Each 80-frame autoregressive chunk uses the caption whose key is the largest one
≤ the chunk's start frame.

### C) Reconstruct an explorable 3D scene (optional)

```bash
python -m lyra2_studio.reconstruct --video outputs/street/videos/my_photo.mp4
# → <video>_gs_ours/reconstructed_scene.ply  +  gs_trajectory.mp4
```

---

## Troubleshooting

- **`LYRA2_HOME ... does not look like a Lyra-2 checkout`** — point it at the
  `Lyra-2` subdirectory (the one containing `lyra_2/`), not the repo root.
- **CUDA OOM** — use `--use-dmd`, lower `--num-frames`, and keep one job at a time.
- **Build failures in `setup_host.sh`** — the dependency stack is sensitive to
  driver/CUDA versions; see the notes in `Lyra-2/INSTALL.md` upstream.
- **Custom video looks wrong / drifts** — reduce motion magnitude (smaller
  `--degrees`/`--distance`) or `--pose-scale`; very large camera moves push the
  model past what it can hallucinate consistently.

## Credits

Lyra 2.0 by NVIDIA's Spatial Intelligence Lab — code
[nv-tlabs/lyra](https://github.com/nv-tlabs/lyra), weights
[nvidia/Lyra-2.0](https://huggingface.co/nvidia/Lyra-2.0),
[project page](https://research.nvidia.com/labs/sil/projects/lyra2/).
