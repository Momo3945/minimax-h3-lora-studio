# MiniMax-H3 LoRA Studio

Two Google Colab notebooks for **MiniMax-H3** image-to-video generation and **FL2VA LoRA training**, built on [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) and the low-memory **MiniMax-H3-NF4** checkpoint.

Author: **Muhammad Hoosen**

---

## What this is

MiniMax-H3 is a video diffusion model that supports **FL2VA** — first-frame / last-frame guided generation, turning a still image (plus a text prompt) into a short video with synchronized audio. This repo has two notebooks that cover the full workflow:

| Notebook | What it does | |
|---|---|---|
| `notebooks/H3_LoRA_Training.ipynb` | Trains a LoRA on top of MiniMax-H3-NF4 using your own short video clips | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Momo3945/minimax-h3-lora-studio/blob/main/notebooks/H3_LoRA_Training.ipynb) |
| `notebooks/H3_Inference.ipynb` | Generates videos from a first-frame image, with an optional trained LoRA, through a Gradio UI | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Momo3945/minimax-h3-lora-studio/blob/main/notebooks/H3_Inference.ipynb) |

`H3_Inference.ipynb` is verified against DiffSynth-Studio commit [`6343ded`](https://github.com/modelscope/DiffSynth-Studio/commit/6343deda06b3e09efc9b1ce23c135c35a341d143) (pinned explicitly, not a blind `git pull`); `H3_LoRA_Training.ipynb` against [`b8e3811`](https://github.com/modelscope/DiffSynth-Studio/commit/b8e3811e5c4abd83a44b99b2bcff7ccabdb26a71) — not written from documentation alone, but built and run end-to-end against the actual current upstream training/inference API.

---

## How it works

```text
Colab (GPU runtime)
    ↓
Google Drive (persistent storage: dataset, model cache, checkpoints)
    ↓
DiffSynth-Studio (training + inference engine)
    ↓
MiniMax-H3-NF4 (quantized base model) + your LoRA
```

### `H3_LoRA_Training.ipynb`

1. **Runtime check** — confirms GPU, VRAM, host RAM, disk.
2. **Google Drive** — mounts Drive and sets up persistent folders for datasets, model cache, checkpoints, and experiment logs.
3. **DiffSynth-Studio setup** — clones and installs the framework, records the exact commit.
4. **Dataset validation** — checks your `metadata.csv` + video clips against MiniMax-H3's actual requirements (frame count, captions, audio) before anything expensive runs.
5. **Training config** — editable cell for resolution, frame count, LoRA rank, learning rate, epochs, save interval, and memory-saving options.
6. **Train** — launches LoRA training via `accelerate launch`, with logs and failure handling (it stops and reports rather than silently continuing through an error).
7. **Validation** — generates a video with each saved checkpoint's LoRA, using a fixed prompt/seed so checkpoints are comparable.
8. **Experiment summary** — writes out a JSON record of the run (settings, timing, VRAM, results).

### `H3_Inference.ipynb`

A Gradio app for everyday generation, now **A100 80GB-aware** with four
live-benchmarked model profiles instead of a single low-VRAM path:

| Profile | What it is | Speed |
|---|---|---|
| **Fast NF4** | Pruned NF4 DiT, fully GPU-resident | Fastest — 50 steps in ~68s |
| **Hybrid** *(experimental)* | Pruned BF16 DiT + NF4 encoder/VAEs, GPU-resident | ~218s — near-max quality, 5.6x faster than Max Quality |
| **Max Quality** | Pruned BF16 DiT + full BF16 encoder, CPU-staged | ~20 min — final renders only |
| **Compatibility** | Same as Fast NF4 but CPU-staged | OOM fallback, any GPU |

Measured live on an actual A100 80GB runtime — see the notebook's final
section and `docs/diffsynth_h3_api_notes.md` for the full numbers.

- Download ready-to-use community LoRAs — [`Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime`](https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime) (a hand-drawn 2D anime look) and **both** verified checkpoints of [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) (distilled LoRAs for fast previews — 4 or 8 steps instead of H3's normal 50) — no training required; cached to Drive so each only downloads once
- Two independent LoRA slots — **Style** and **Speed** — a precise strength field on Speed (not a coarse slider, since Turbo checkpoints need an exact value — see below) and a slider on Style, so a style LoRA (anime) and a speed LoRA (Turbo) can be stacked and used together, or either used alone; also usable for LoRAs you trained yourself. Selecting a recognized Turbo checkpoint auto-applies its verified steps/shift/strength recipe.
- **Turbo Speed LoRAs are blocked outright on the Max Quality and Hybrid profiles** — confirmed live to produce corrupted output (see "Notes from real runs" below) — use Fast NF4 or Compatibility for Turbo generations.
- Resolution presets (preview/final, landscape/portrait) plus a quality preset that sets profile + resolution together; custom values still available under "Advanced."
- Upload a first-frame image, write a prompt (dialogue can be written directly into the prompt — there's no separate "sound prompt" field in the current model)
- Pick frame count (up to 345 frames / ~14.4s at 24fps), seed, inference steps
- A post-generation status card reports memory mode, VRAM usage, effective settings, active LoRAs, and generation time
- Generate a previewable MP4, saved to the Colab session's local disk (`/content/generations`) — **not** Drive. Only the model cache and any LoRA checkpoints you train live on Drive; generated videos are treated as disposable per-session scratch output. A final "Browse generated videos" cell lists and inline-previews everything you've generated so far, so you can review results without leaving the notebook — just download anything you want to keep before the runtime disconnects.

---

## Requirements

- A **GPU Colab runtime**. `H3_Inference.ipynb`'s A100 80GB profiles were built and measured on an actual `NVIDIA A100-SXM4-80GB` (Colab Pro+) — the Max Quality profile alone needs ~105GB of BF16 weights on disk. `H3_LoRA_Training.ipynb` targets the lower-memory NF4 path and was tested on more modest hardware; see its own runtime-check cell.
- A **Google Drive** with enough free space for the model cache — budget ~150GB+ if you want all four inference profiles cached (NF4 ~33GB, pruned BF16 DiT ~40GB, full BF16 text encoder+VAEs ~70GB), or just ~33GB if you only use the Fast NF4 / Compatibility profiles.
- Your own short video clips + captions if you're training a LoRA (a few seconds each, any resolution — the notebook resizes/crops to your target).

---

## Notes from real runs

A few things that came up building and running this, worth knowing before you start:

- **A Turbo/distilled LoRA's correct `strength` depends on how its official reference script calls `--lora-alpha`, not on the pipeline's `flow_shift`.** DiffSynth-Studio's `load_lora(..., alpha=X)` uses `alpha` as a raw multiplier, while PEFT-style reference scripts use `alpha/rank`. Both `lightx2v/Minimax-h3-Turbo` checkpoints are rank 128: the 4-step 768p one's reference script overrides `lora-alpha=128` (→ `strength=1.0`), the 8-step one doesn't (default `lora-alpha=8` → `strength=0.0625`). Using `strength=1.0` on the 8-step checkpoint is a 16x overscale that produces static-noise output easily mistaken for a broken checkpoint or wrong sampling schedule — we initially misdiagnosed it as the latter before finding the actual math.
- **Turbo LoRAs don't transfer across quantization variants of the "same" base model, even with matching tensor shapes.** Applying a Turbo checkpoint's exact verified recipe to a full-precision BF16 DiT (instead of the NF4-quantized DiT it was distilled against) structurally patches every target tensor with zero load-time error, but produces a corrupted watermark/glyph artifact — not a subtle quality loss. Now blocked outright in the Gradio UI rather than left to silently corrupt output.
- **A third community LoRA key-naming format (kohya/sd-scripts) went completely unrecognized and silently no-op'd.** DiffSynth-Studio's LoRA loader only auto-converts its own native format and lightx2v/Diffusers-style separate-QKV naming. A LoRA using kohya's flat underscore-joined keys (`lora_unet_blocks_0_attn_out_proj...`) reports `"0 tensors are patched"` with zero visual effect — no error. The bundled anime style LoRA uses exactly this format and was never actually applying in this notebook until a small key-converter was added. If a downloaded community LoRA seems to have no effect, check the notebook output for a "0 tensors are patched" line before assuming the strength is just too low.
- **`peft` needs `torchao>=0.16.0`** if `torchao` is importable at all, or LoRA training throws an `ImportError`. The training notebook installs the right version automatically.
- **Never leave a dataset's `input_audio` column blank.** DiffSynth's loader reads a blank cell as `NaN` and crashes before training starts, even though it already has a graceful silent-audio fallback for a video file with no audio track. Point `input_audio` at the video's own filename instead, even for silent clips.
- **Model downloads are cached to an absolute Drive path** (`DIFFSYNTH_MODEL_BASE_PATH`), not a directory-relative one — so the ~32GB download only ever has to happen once, and survives Colab runtime restarts.

---

## License

No license file yet — treat as "all rights reserved" until one is added.
