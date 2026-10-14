# Staged Sampling Audit (2026-10-14)

## Scope and contract

`model.sample_staged_offloading` is opt-in and defaults to `false`. The UI label
is `采样分阶段装卸（实验）`. It may be enabled together with toolkit layer
offloading, including partially offloaded components and nested training adapters.
Model compilation remains mutually exclusive.

Only sampling residency changes. The controller restores component placement,
module training flags, parameter gradient flags, LoRA activation/multipliers,
random states, and temporary pipeline hooks. It does not change the optimizer,
training settings, model paths, database, or installed dependencies. It does not
offload optimizer states and cannot solve a training working set that is itself
too large. Layer-offloader CUDA scratch buffers can remain allocated.

The layer manager's pinned CPU weights, layer partition and forward/backward
hooks are retained. Only its resident parameters/buffers and explicit bare
resident tensors move between phases. Ordinary trainable parameter identity and
gradients are preserved. Trainable quantized tensor subclasses are rejected
because moving them may replace Parameter objects held by the optimizer.

## Default ComfyUI-directory architectures

These are the four families explicitly using `configured_models` in
`portable_models.json`, covering seven architecture choices in the UI. Each now
has the experimental option, with its own sampling integration.

| Architecture | Implementation | Important differences |
| --- | --- | --- |
| Qwen Image 2.1 / Turbo | Prompt encoder, optional reference VAE encoding, denoiser, VAE decoding | The live Turbo training adapter and its nested layer managers are included. |
| Ming-Image 0.1 Design | Prompt encoder, optional reference VAE encoding, denoiser, VAE decoding | Both query/direct embeddings remain available; RGBA decoding is unchanged. |
| MiniMax-H3 / Ref2V / FastH3 | Prompt encoder, reference encoding, joint video/audio denoiser, VAE bundle | Reference videos can be encoded before entering the sampler. Both audio and video VAEs move together, not one at a time. |
| LTX-2.5 | Prompt encoder, connectors/denoiser, video/audio VAEs and vocoder | Dedicated integration intercepts VAE encode/decode and component forward boundaries. I2V switches back to denoising after first-frame encoding. The sampling pipeline has an explicit execution device so parked CPU modules cannot cause CPU latent allocation. |

Sources:

- `toolkit/staged_sampling.py`
- `toolkit/models/base_model.py` (`generate_images`, `sample_stage`)
- `extensions_built_in/diffusion_models/qwen_image_2/src/pipeline.py`
- `extensions_built_in/diffusion_models/ming_image/{ming_image.py,src/pipeline.py}`
- `extensions_built_in/diffusion_models/minimax_h3/{minimax_h3.py,src/pipeline.py}`
- `extensions_built_in/diffusion_models/ltx2/ltx2.py` (`LTX25Model`, staged pipeline classes)

Sharing a ComfyUI directory is NOT the compatibility criterion. In particular,
LTX requires connectors, audio components and execution-device handling which
the original Qwen-only implementation did not cover.

## Sharded Checkpoints and Other Architectures

Checkpoint sharding is a storage/loading concern, not a sampling schedule.
After loading ordinary modules, the number of checkpoint files does not prevent
staging. The runtime component graph and device-management hooks determine
whether it is safe. This agrees with the upstream descriptions of
[sharded checkpoints and offloading](https://huggingface.co/docs/diffusers/optimization/memory)
and [Accelerate device dispatch](https://huggingface.co/docs/accelerate/main/concept_guides/big_model_inference).

The following are code-review judgments, NOT newly enabled support or full-model
performance claims:

| Model family | Judgment | Required work before exposing the option |
| --- | --- | --- |
| Krea2 Raw/Turbo and Edit | Suitable in principle; custom sampler is relatively direct | Stage reference-image VAE encoding, denoising and final decode; remove the early whole-denoiser move in `krea2.py:generate_single_image`; test edit references and cached conditioning. |
| Flux2 / Klein | Suitable in principle, including directory/sharded checkpoints | `flux2_model.py:get_generation_pipeline` currently moves the entire pipeline to GPU. Its custom pipeline also encodes reference images. Integrate those boundaries and test both distilled and non-distilled conditioning. |
| Original Qwen Image / Edit / Edit Plus | Suitable in principle | Their pipelines eagerly move to GPU, and edit pipelines have image-conditioning paths separate from Qwen 2.1. They do not inherit the Qwen 2.1 integration. |
| Z-Image | Suitable in principle | Replace the eager pipeline/model GPU placement and cover the sampler's reference/decode paths, including variants. |
| Wan 2.2 5B | Suitable in principle, with a dedicated video sampler integration | Cover image-conditioning and VAE phases; do not assume it shares the 14B schedule. |
| Wan 2.2 14B / I2V | Not compatible with the generic three-group implementation as-is | Uses two denoisers and an existing `aggressive_offload` high/low-noise handoff. Needs per-denoiser stages and coordination with existing offloading; moving both denoisers together would defeat memory savings. |
| LTX-2 / 2.3 | Likely reusable after validation, but not enabled here | They share the parent sampler but have different checkpoint/component conversion and text encoders. This change enables only audited LTX-2.5. |
| Legacy SD / SDXL / Flux1 and other unlisted architectures | Not covered by this implementation | Some use the legacy `StableDiffusion` sampling class rather than `BaseModel`; control adapters, multiple text encoders, refiner paths and full-finetuning need separate coverage. |

Some original Qwen, Z-Image and Wan v2 loaders also support optional ComfyUI-file
fallbacks (`toolkit/models/v2/diffusion_models/`). That fallback does not turn them
into one of the four default ComfyUI families above, nor make their sampling
paths compatible automatically. Their toggles remain hidden rather than doing
nothing or silently overriding an incompatible offloader.

Runtime combinations intentionally rejected:

- Accelerate CPU/disk offload hooks.
- Split `hf_device_map` placements, disk dispatch, or unmanaged mixed-device roots.
- Whole-model compilation.
- Nested staged-sampling sessions.

These cases need dedicated coordination, not unconditional `.to("cpu")` calls.
The default-off path keeps its original behavior.

## Validation Boundaries

- All 22 staged-sampling and pipeline tests passed. They exercise phase residency, exception/cancellation cleanup, RNG and
  flag restoration, train/sample/train optimizer references, regular and nested
  layer managers, bare resident tensors, and preview hooks.
- Ming and H3 tests execute production sampler control flow with tiny components
  and compare staged/non-staged results. H3 includes image/video/audio/reference
  paths. These are not full-checkpoint quality or performance tests.
- LTX tests use the production adapter and actual T2V/I2V constructors, exercise
  connector/VAE/denoiser/vocoder calls, and verify cleanup. Full LTX checkpoint
  generation is not claimed.
- UI verification checks all seven choices, default-off state, simultaneous
  switches, generated YAML, and narrow-screen label wrapping. No task was saved.
- Real Qwen combined-offloading verification uses an independent four-step
  configuration, independent dataset caches/output, 50% layer offloading for
  transformer and text encoder, and cached text embeddings. The original job is
  not edited. Run `staged-sampling-combined-20261014-2` completed with exit code 0
  in 142.44 seconds, including loading, caching, four training steps and saving.
  Baseline / middle / final samples took 13.56 / 13.03 / 13.25 seconds at
  1024x1024, CFG 1, 8 steps. The middle sample was followed by steps 3 and 4;
  the checkpoint and optimizer state were saved. The generated image was
  visually inspected. This is one configuration, not an all-model guarantee.
- A first combined run stopped before sampling because the training adapter's
  nested layer managers were not detected. The implementation now detects them,
  and a dedicated regression test covers their pinned CPU weight preservation.
- Audio/video VAE placement is captured separately because only one VAE may
  have been loaded onto GPU during dataset caching.
- Frontend production build and 58 existing portable/manager regression tests
  passed. No dependency installation, user task edits, or Git push was performed.

## Phase-major batches and generation page

The opt-in path now queues the whole sampling round in host memory: encode all
prompts serially, prepare all references serially, denoise each request serially,
then decode/save each result serially. It does not increase the GPU batch size.
Each request retains its own RNG snapshot, dimensions, conditioning and LoRA
multiplier. Pending embeddings/reference latents/output latents are kept on CPU;
system RAM usage grows with the number and size of samples, especially videos.
Completed images appear during the final decoding phase, not during denoising.

Reference encoding can require a separate VAE phase before denoising. Cached
prompts can skip text-encoder residency entirely. Audio/video components remain
model-specific, including H3 condition noise order and LTX normalized latents,
decode noise, audio VAE and vocoder. Production pipeline comparisons cover those
paths with tiny components; they are not full-checkpoint performance claims.

Each phase exposes count/total progress; denoising also exposes step progress.
Per-step VAE previews are suspended so they cannot force a VAE transfer every
step. The inference engine's latent-only preview/progress hook stays active,
including cancellation. The generation page exposes the same seven supported
architecture choices as training, defaults to off and permits layer offloading
at the same time. Unsupported architectures do not display or submit this flag.

Additional verification:

- All 29 current staged-sampling, batch, engine and pipeline tests passed.
- The 10-request batch test checks phase-major ordering and bit-exact serial
  RNG equivalence across preparation, denoising and decoding.
- Engine tests execute the production request runner with a tiny fake holder,
  checking model flag parsing, phase/step events, success and cancellation at
  phase/step boundaries, and hook cleanup. Batch cancellation clears queued data.
- Real Qwen 2.1 Turbo run `staged-batch10-20261014-1` produced 30 full-size images
  across baseline/middle/final sampling rounds (10 each), with separate thumbnails.
  Each round logged a single denoiser phase then a single VAE decode phase.
  The three denoise phases took 97.12 / 99.85 / 102.55 seconds; decoding took
  9.48 / 8.91 / 9.09 seconds. These are stage timings, not total startup time.
  Training resumed after the middle round, reached 4/4, and saved the checkpoint
  and optimizer. One generated image was visually inspected. Settings were
  1024x1024, CFG 1, 8 steps, cached text embeddings and 50% layer offloading.
  GPU telemetry was recorded every second (10,925 MiB maximum total GPU usage).
- Follow-up `staged-batch2-uncached-20261014-1` used the final deployed code with
  text-embedding caching disabled and two samples per round. It completed with
  exit code 0 in 224.45 seconds, produced six images, resumed training to 4/4
  after the middle sample and saved model/optimizer states. Every round logged
  exactly one text-encoder, one denoising and one VAE decoding phase. Denoising
  took 19.06 / 18.77 / 18.68 seconds for two images per round. Startup, model
  loading, training, device transfers and saving are included only in total time.
- Generation UI checks used the built production frontend plus an isolated
  simulated engine stream: all seven switches, unsupported-model hiding,
  default-off, simultaneous layer offloading, request payload, phase and step
  bars were checked at narrow and desktop widths. This is UI verification,
  not a full-model HTTP generation benchmark; the real model run was separate.
- The portable production build succeeded. Standalone TypeScript checking still
  reports 78 diagnostics in other files; none target the changed generation,
  model-options or documentation files. Full-project typecheck is not claimed.

## Launcher distribution

The portable launcher updates from `DocWorkBox/ai-toolkit-easy2use`, branch
`protable`, through `manager.portable_update`. The updater copies the new Python
module, model pipelines, inference engine and both UI configuration pages; it
protects datasets, models, output, runtime, database and generated UI folders.
The UI source fingerprint changes, so the next service start rebuilds Next.js
and the worker once before starting. Later unchanged starts reuse that build.
These sampling features do not introduce new package dependencies or require a
new launcher executable. Stop the service before updating, then restart the
service/inference engine and refresh the browser to use the new code.

All 26 launcher-update, build-cache and portable-layout tests passed, including
an archive test using the real staged-sampling source files. That test checks
backend/UI delivery, old-build invalidation and exactly one requested rebuild.
