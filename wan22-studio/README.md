# wan22-studio

Run **Wan 2.2** on a single GPU to do **image-to-video**: start from **one image**
(for example a Google Maps / Street View frame) and let the **scene continue** as a
short video — via a small command-line tool.

This repo is a thin **orchestration layer** around the official inference code
([Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2)). It does not reimplement
the model; it automates installing it, downloading weights, and running generation.

---

## Read this first

**What this does.** Wan 2.2 image-to-video takes a **single still image** as the
first frame and generates a video that *continues* from it — the scene comes to
life (light, motion, atmosphere) and, with a prompt, evolves how you describe. This
is **not** a camera flythrough of a frozen 3D world (that's a different kind of
model); the camera roughly holds while the **scene plays forward in time**.

> It is a **generative** model: anything not in your starting image is *invented*,
> plausibly. It does not reconstruct the real place from multiple photos.

**Model.** We use **`Wan2.2-I2V-A14B`** — a Mixture-of-Experts model (~14B active /
27B total, a high-noise + a low-noise expert). It generates at **480P or 720P**,
~5 s clips.

**Hardware.** A single **NVIDIA A100 80GB** is the validated target (also H100/H200).
The official single-GPU recipe needs **≥ 80 GB VRAM** with model offload on — which
is exactly what this wrapper enables by default (`--offload_model True` +
`--convert_model_dtype`). Linux only; no Mac/CPU. *Tighter on VRAM? Add `--t5-cpu`,
or switch to the lighter `Wan2.2-TI2V-5B` (8–12 GB, runs on consumer cards).*

**Disk.** The I2V-A14B checkpoint is large (~70 GB). Provision **≥ 150 GB** of disk
(weights + conda env + outputs).

**License.** Wan 2.2 code and weights are **Apache-2.0** — open, **commercial use
permitted**, no license to accept and no gated download. The wrapper code here is MIT.

---

## What's in here

```
wan22_studio/
  config.py            paths/env (WAN22_HOME, checkpoint dir, interpreter)
  generate.py          CLI: image-to-video generation
scripts/
  bootstrap.sh         one-shot deploy on a fresh A100 (install + weights + smoke test)
  setup_host.sh        install Wan 2.2 on a GPU host (mirrors the official README)
  download_weights.sh  fetch Wan2.2-I2V-A14B from Hugging Face
  smoke_test.sh        sanity-check the install on Wan's bundled example image
```

---

## Quickstart (on the A100 host)

**Fastest path — one command (unattended).** `bootstrap.sh` installs Miniconda if
missing, downloads the weights *in parallel* with the env build, then smoke-tests:

```bash
# get this repo onto the box (rsync/scp or git clone), then:
cd ~/wan22-studio
bash scripts/bootstrap.sh
```

Setup is light (no exotic from-source CUDA stack): roughly **10–20 min**, plus the
weight download (overlapped). A HF token is optional (`export HF_TOKEN=hf_...`, only
helps rate limits) — no license acceptance needed.

**Or step by step:**

```bash
bash scripts/setup_host.sh ~/wan-src
conda activate wan22
export WAN22_HOME=~/wan-src/Wan2.2
bash scripts/download_weights.sh
bash scripts/smoke_test.sh
```

`WAN22_HOME` is the only required setting. Optional: `WAN22_CKPT_DIR`
(default `Wan2.2-I2V-A14B`), `WAN22_OUTPUT_DIR`, `WAN22_PYTHON`.

---

## Continue a scene from your image

```bash
python -m wan22_studio.generate \
  --image manama_street.jpg \
  --prompt "Late afternoon light over the street; gentle traffic and pedestrians move, a flag sways, cinematic, photorealistic." \
  --size "1280*720" \
  --save-file outputs/manama.mp4
```

Useful flags:

| flag | effect |
|------|--------|
| `--prompt` | how the scene continues (motion, mood, time of day). Optional but improves results. |
| `--size "1280*720"` | 720P; use `"832*480"` for faster 480P. **Quote it** (the `*`). |
| `--frames N` | clip length (Wan default ~81 ≈ 5 s @ 16 fps) |
| `--steps N` | sampling steps — more = better/slower |
| `--seed N` | reproducible (`-1` = random) |
| `--prompt-extend` | auto-expand the prompt with a local Qwen model (better quality, extra download) |
| `--t5-cpu` | keep the text encoder on CPU to save VRAM |
| `--dry-run` | print the exact Wan command without running it |

### Prompt tips for a Maps frame

- Describe the scene as **continuing**: what *moves* (cars, people, water, clouds,
  light shifting), the mood, the time of day. Wan animates what you name.
- Keep it grounded in what's visible — the model extends your frame, it doesn't
  invent a new location.
- **Image quality matters**: Google Maps / Street View grabs are often low-res or
  watermarked. A clean, sharp frame gives far better results.

---

## Troubleshooting

- **`WAN22_HOME ... does not look like a Wan2.2 checkout`** — point it at the cloned
  `Wan2.2` directory (the one containing `generate.py`).
- **CUDA OOM on 80 GB** — keep the defaults (offload on), add `--t5-cpu`, lower
  `--size` to `"832*480"` or `--frames`.
- **`flash_attn` build fails during setup** — `setup_host.sh` already retries it last;
  if it still fails, install the other deps first, then `flash_attn` on its own.
- **The `*` in `--size` does nothing / globs** — quote it: `--size "1280*720"`.

## Credits

Wan 2.2 by the Wan team (Alibaba) — code & weights
[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2),
[Wan-AI on Hugging Face](https://huggingface.co/Wan-AI). Apache-2.0.
