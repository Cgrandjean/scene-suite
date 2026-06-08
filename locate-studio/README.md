# locate-studio — object detection on images & videos (NVIDIA LocateAnything-3B)

Open-vocabulary object **detection** and **grounding** with
[`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) — a 3B
vision-language model. Describe what you want in plain words; get bounding boxes back.

- **Detection**: give a list of categories → boxes for every instance of each.
- **Grounding**: give a phrase ("the man in the red shirt") → boxes for what it refers to.
- **Pointing**: get a point instead of a box.
- Works on a single **image** or, frame-by-frame, on a **video**.

Light: **~6–8 GB VRAM** at bf16, **no from-source CUDA compile** (unlike Lyra/Wan).
Tested arch: Ampere (A100) / Hopper (H100) / Lovelace (RTX 4090) / Blackwell.

> ⚠️ The model is **gated** and **non-commercial** (NVIDIA license). Accept it on the
> model page and `export HF_TOKEN=hf_...` before downloading.

## Deploy (fresh GPU box)
```bash
export HF_TOKEN=hf_xxxx
bash scripts/bootstrap.sh        # miniconda + env + gated download + load check
```
Or step by step: `setup_host.sh` → `download_weights.sh` → `smoke_test.sh`.

## Use
```bash
conda activate locate

# image: detect categories -> annotated PNG + JSON
python -m locate_studio.detect --image street.jpg --classes "person,car,traffic light"

# video: detect every 5th frame -> annotated MP4 + per-frame JSON
python -m locate_studio.detect --video clip.mp4 --classes "person,dog" --stride 5 --max-frames 300

# referring grounding (one phrase instead of a class list)
python -m locate_studio.detect --image room.jpg --query "the lamp on the left"
```
Outputs land in `--out` (default `outputs/locate/`): `<name>_annotated.{png,mp4}` + `<name>.json`.

## Key options
| Flag | Meaning |
|---|---|
| `--image` / `--video` | input (one or the other) |
| `--classes "a,b,c"` | open-vocab categories to detect (queried per-class for reliable labels) |
| `--query "..."` | a single referring phrase (grounding); add `--point` to get a point |
| `--stride N` | video: run detection every Nth frame (default 5) |
| `--max-frames M` | video: stop after M processed frames (0 = whole clip) |
| `--out DIR` | output directory |
| `--model` | HF id or local path (default `nvidia/LocateAnything-3B`, or `$LOCATE_MODEL`) |
| `--temperature` / `--max-new-tokens` / `--generation-mode` | decoding controls |

## How it works
`LocateAnythingWorker` (in `locate_studio/worker.py`) loads the model once via
`transformers` (`trust_remote_code=True`) and prompts it with the model's templates;
the model answers with `<box><x1><y1><x2><y2></box>` tokens normalised to `[0,1000]`,
which `parse_boxes` rescales to pixels. `detect.py` draws the boxes and writes JSON.
Video is processed per frame (LocateAnything has no native video mode).
