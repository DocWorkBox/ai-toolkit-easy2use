# AI Toolkit – Easy to Use (English README)

This repository is an easy-to-use variant of the AI Toolkit (ai-toolkit-easy2use). It restores the UI back to English while preserving all adaptive UI improvements. The goal is simpler installation, smoother onboarding, and practical day‑to‑day use.

> Original project by Ostris; this repository maintained by DocWorkBox.

---

## Overview

- Unified toolkit for training and inference of diffusion models.
- Supports common image/video model families and consumer GPUs.
- Provides both CLI and a Next.js‑based Web UI.

## Requirements

- Python ≥ 3.10
- Git
- NVIDIA GPU (VRAM suitable for your tasks)
- Node.js ≥ 18 (for Web UI)

## Installation (Linux / Windows)

```bash
git clone https://github.com/DocWorkBox/ai-toolkit-easy2use.git
cd ai-toolkit-easy2use
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows (PowerShell)
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Install PyTorch according to your CUDA/driver environment (example for CUDA 12.6):

```bash
pip install --no-cache-dir \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

## Run the UI

The Web UI is built on Next.js. It does not need to run continuously to execute jobs; use it to start/stop/monitor.

Development mode (http://localhost:3000):

```bash
cd ui
npm install
npm run dev
```

Production mode (port 8675):

```bash
cd ui
npm run build_and_start
```

Access:

- `http://localhost:8675` (local)
- `http://<your-ip>:8675` (remote)

## GPU via Docker Compose

If you prefer containerized deployment with GPU, declare device reservations in `docker-compose.yml`:

```yaml
services:
  ai-toolkit:
    image: coco1006/ai-toolkit-easy2use:0.7.2
    ports:
      - "8675:8675"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
```

Launch:

```bash
docker compose up -d
```

Verify GPU inside the container:

```bash
nvidia-smi
```

## UI Adaptive Improvements (kept)

- Sidebar: mobile collapsed by default; fixed responsive widths to avoid layout jitter.
- Sidebar Toggle Button: fixed at mid‑left, works on mobile/desktop via global events.
- Top Bar: truncates long titles on mobile, prevents horizontal scroll and overflow.
- Sample Images Grid: responsive columns (2 on phones, up to 8 max), stable Tailwind classes.
- Select Menus: rendered via Portal to avoid clipping by parent `overflow`.
- GPU/CPU widgets: refined labels and consistent English copy.

These improvements are preserved while reverting all visible UI copy to English.

## FAQ

- Out of VRAM?
  - Enable conservative options (e.g., `low_vram`) or offload/quantize parts to reduce memory.
- Windows setup issues?
  - Ensure Python/CUDA/driver versions match; WSL is also a viable choice.
- UI/API not responding?
  - Check Node.js version (≥18), run `npm install`, and ensure the dev server is up (`npm run dev`).

## Repository Layout

- `config/` – Configuration examples and templates.
- `ui/` – Next.js UI source (English, with adaptive improvements).
- `requirements.txt` – Python dependencies.
- Training scripts, adapters, and model utilities are organized by module.

## License

Follows the original project’s license. Ensure model/data licenses are respected for distribution or commercial use.