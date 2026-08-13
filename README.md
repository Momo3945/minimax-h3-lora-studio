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

Both notebooks are verified against DiffSynth-Studio commit [`b8e3811`](https://github.com/modelscope/DiffSynth-Studio/commit/b8e3811e5c4abd83a44b99b2bcff7ccabdb26a71) — not written from documentation alone, but built and run end-to-end against the actual current upstream training/inference API.

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

A Gradio app for everyday generation — the notebook you'd use once you have (or don't need) a trained LoRA:

- Load MiniMax-H3-NF4 (or BF16)
- Download ready-to-use community LoRAs — [`Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime`](https://huggingface.co/Inner-Reflections/MiniMax-H3-Looping-Sketch-Anime) (a hand-drawn 2D anime look) and [`lightx2v/Minimax-h3-Turbo`](https://huggingface.co/lightx2v/Minimax-h3-Turbo) (an 8-step distilled LoRA for fast previews, vs. H3's normal 50 steps) — no training required; cached to Drive so each only downloads once
- Two independent LoRA slots — **Style** and **Speed** — each with its own strength slider, so a style LoRA (anime) and a speed LoRA (Turbo) can be stacked and used together, or either used alone; also usable for LoRAs you trained yourself
- Upload a first-frame image, write a prompt (dialogue can be written directly into the prompt — there's no separate "sound prompt" field in the current model)
- Pick resolution, frame count, seed, inference steps
- Generate a previewable MP4, saved to the Colab session's local disk (`/content/generations`) — **not** Drive. Only the ~32GB model cache and any LoRA checkpoints you train live on Drive; generated videos are treated as disposable per-session scratch output. A final "Browse generated videos" cell lists and inline-previews everything you've generated so far, so you can review results without leaving the notebook — just download anything you want to keep before the runtime disconnects.

---

## Requirements

- A **GPU Colab runtime**. MiniMax-H3-NF4's FL2VA weights total **~32GB** — despite the "NF4" (4-bit quantized) name, that's larger than a free-tier T4's VRAM *and* host RAM. A high-RAM / A100 runtime (Colab Pro or Pro+) is what this was actually tested on.
- A **Google Drive** with enough free space for the model cache (~32GB) plus whatever checkpoints/videos you generate.
- Your own short video clips + captions if you're training a LoRA (a few seconds each, any resolution — the notebook resizes/crops to your target).

---

## Notes from real runs

A few things that came up building and running this, worth knowing before you start:

- **`peft` needs `torchao>=0.16.0`** if `torchao` is importable at all, or LoRA training throws an `ImportError`. The training notebook installs the right version automatically.
- **Never leave a dataset's `input_audio` column blank.** DiffSynth's loader reads a blank cell as `NaN` and crashes before training starts, even though it already has a graceful silent-audio fallback for a video file with no audio track. Point `input_audio` at the video's own filename instead, even for silent clips.
- **Model downloads are cached to an absolute Drive path** (`DIFFSYNTH_MODEL_BASE_PATH`), not a directory-relative one — so the ~32GB download only ever has to happen once, and survives Colab runtime restarts.

---

## License

No license file yet — treat as "all rights reserved" until one is added.
