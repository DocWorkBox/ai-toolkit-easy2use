# chengyu Model Sources

Shared model inventory checked on 2026-11-10 through the platform's file manager
and read-only terminal. This records directory and filename matches, not a full
weight-integrity check or a GPU training test.

## Existing Deployment

The supplied old options contain 33 entries. The current plugin has matches for
32 entries; their sources, including existing repo defaults, are preserved.
The 23 different local defaults were migrated to `/chenyudata/Other/`, except
LTX-2.3, which uses the supplied checkpoint below. Flex.2 is not registered by the
current plugin and was not restored from the old options.

| Model | Local Source |
| --- | --- |
| LTX-2.3 | `/chenyudata/ComfyUI/models/checkpoints/ltx/ltx-2.3-22b-dev.safetensors` |
| Anima | `/chenyudata/Other/Anima-Base-v1.0-Diffusers` |
| Qwen-Image-2.1 / Turbo weights | `/chenyudata/Other/Comfy-Org/Qwen-Image-2.1` |
| Qwen-Image-2.1 configuration | `/chenyudata/Other/Qwen-Image-2.1` |
| MiniMax-H3 / Ref2VA weights | `/chenyudata/Other/Comfy-Org/MiniMax-H3` |
| MiniMax-H3 original configuration | `/chenyudata/Other/MiniMax/MiniMax-H3` |
| LTX-2.5 | `/chenyudata/Other/LTX-2.5` |
| FLUX.3 Image Klein Base | `/chenyudata/Other/flux-3-image-klein-base` |
| ERNIE-Image | `/chenyudata/Other/ERNIE-Image` |
| Ideogram4 | `/chenyudata/Other/ideogram-4-fp8` |
| Krea 2 Raw / Edit | `/chenyudata/Other/Krea-2-Raw` |
| Krea 2 Turbo / Edit Turbo | `/chenyudata/Other/Krea-2-Turbo` |
| Boogu Image | `/chenyudata/Other/Boogu-Image-0.1-Base` |
| Boogu Image Edit | `/chenyudata/Other/Boogu-Image-0.1-Edit` |
| Qwen3-VL-4B configuration and HF weights | `/chenyudata/Other/Qwen3-VL-4B-Instruct` |
| Qwen3-VL-8B configuration and captioning | `/chenyudata/Other/Qwen3-VL-8B-Instruct` |
| Krea Qwen-Image VAE source | `/chenyudata/Other/Qwen-Image-2512` |
| FLUX.2 Mistral text encoder | `/chenyudata/Other/Mistral-Small-3.1-24B-Instruct-2503` |
| LTX-2 Gemma text encoder | `/chenyudata/Other/gemma-3-12b-it-qat-q4_0-unquantized` |

FastH3 8-Step V2 and Omni keep official repo defaults for their missing DiTs,
but reuse the confirmed local MiniMax-H3 text encoder and both VAEs.

MOSS and Fibo keep official repo identifiers in the UI to retain variant and
precision selection. Existing weights are resolved under
`/chenyudata/ComfyUI/models`, including these observed subdirectories:

- `diffusion_models/MOSS-Image/`: Base and SFT, bf16 and int8 ConvRot.
- `diffusion_models/Fibo/`: Fibo 2.0 dev and edit turbo, bf16 and int8 ConvRot.
- `text_encoders/qwen3.5/`: Qwen3.5-9B bf16 and int8 ConvRot.
- `text_encoders/qwen3vl/`: Qwen3-VL-4B/8B precision variants.
- `vae/`: shared FLUX.2 and Qwen VAE files.

FLUX.2 Klein can use `text_encoder/` inside its existing local 4B/9B model folders.
Its fallback repos remain `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`.

## Not Matched Locally

These versions were not found in the inspected shared model directories or
targeted checkpoint search. They retain official repo defaults:

- P-H3-Image; FastH3 8-Step V2; FastH3 Omni.
- Ming-Image; Nucleus-Image; HiDream-O1; Z-Image L2P; PRX Pixel; IRIS-3B;
  Mage-Flow Base and Edit Base.
- Qwen3.8-27B W4A8; Qwen3.6-27B and its Huihui variant; Huihui Qwen3-VL-8B.
- Qwen2.5-Omni-7B and H3 Prompt Rewriter; Qwen3-Omni variants.
- YuE2; MERT-v2-FullSong; SheetSage2; YuE2 audio tokenizer; MOSS-Music-8B.
- ACE-Step captioner/transcriber and toolkit-specific ACE-Step AIO checkpoints.
- Toolkit-specific MelBandRoformer and training adapters.

Qwen3.6-35B-A3B and several GGUF models are present but are not substitutes for
the requested parameter sizes and formats. Shared MelBandRoformer fp16/fp32
files differ from `melbandroformer_vocals_kj.safetensors`; they were not assumed
to contain the toolkit's required embedded configuration.

## Creator Downloads

The requested missing models download to `/poddata/ai-toolkit-models/` using
`scripts/chengyu_models.json`. Model repo defaults remain valid download sources;
their on-disk destinations are selected by the storage layer.

- `diffusion_models/`: P-H3, FastH3 V2/Omni DiTs and Z-Image L2P.
- `text_encoders/`: the P-H3 encoder, Qwen3.8 W4A8 and the default Huihui
  Qwen3-Omni Thinking ConvRot checkpoint. Other Omni variants are not prefetched.
- `vae/`: the image-only P-H3 VAE; shared H3 VAEs are not duplicated.
- `checkpoints/`: YuE2, both toolkit ACE-Step 1.5 AIOs and MelBandRoformer.
- `audio_encoders/`: the Comfy-format SheetSage2 weights.
- `ai_toolkit/`: YuE2 semantic-tokenizer head and NAR LoRA.
- `huggingface/<owner>/<repo>/`: MERT, MOSS-Music, ACE-Step captioner/transcriber,
  and SheetSage2 configuration/code, preserving each HF directory layout.
- `loras/training_adapters/`: the current default training adapters.
- `configs/`: small P-H3, Qwen3.8 and Omni assets staged for the project cache.

The downloader pins revisions, skips complete files, reuses matching persistent
models through links, resumes incomplete Hub transfers and refuses to overwrite
an existing destination with a different size. It writes `download.log` when
launched with output redirection, and always records `download-state.json`.

From a chengyu checkout, start/resume or inspect the download with:

```bash
python scripts/download_chengyu_models.py --root /poddata/ai-toolkit-models --workers 3
python scripts/download_chengyu_models.py --root /poddata/ai-toolkit-models --status
```

Models outside the requested set, such as Ming, Nucleus, HiDream-O1, PRX, IRIS,
Mage-Flow, Qwen3.6 and the Qwen2.5-Omni H3 rewriter, are not prefetched.

## Cache and Selection Rules

The shared Comfy directory is an additional read-only search root. Explicit
component overrides and project caches keep their previous precedence.
Precision ranking and variants are unchanged. Missing files download into the
configured writable `MODELS_PATH`, whose chengyu default is now
`/poddata/ai-toolkit-models`, not the shared mount. Explicit environment/custom
settings still override the default. Saved settings equal to the old project
default are recognized without a database migration.

H3-family training adapters retain repo references and use
`<MODELS_PATH>/loras/training_adapters/`. Existing Qwen3.8, P-H3, FLUX.3,
MOSS, Omni and LTX tokenizer/config caches remain enabled. API captioning,
localization, custom optimizers and staged sampling are unchanged.

Small downloaded P-H3, Qwen3.8 and Omni configurations seed the existing
`<TOOLKIT_ROOT>/models/` caches without copying weights or overwriting valid
cached files. Completed full HF snapshots are read offline from the creator
volume; missing/partial snapshots are completed in the same location.
