# AI Video Highlights

Turn long videos into concise, meaningful highlight clips — entirely on your machine.

The app analyzes a lecture, interview, podcast, tutorial, gameplay recording, meeting, or product demo, finds the strongest moments, drops repetition and low-value padding, and renders a short highlight that still makes sense.

Example:

```text
Input:  5 minute video
Target: 20–30 seconds
Output: a coherent 20–30 second highlight of the strongest moments
```

Everything runs locally. Your videos are not uploaded to third-party AI services. There is no telemetry and no cloud processing.

---

## Features

- Local upload of MP4, MOV, MKV, AVI, WebM, and M4V
- Background jobs (the HTTP request never blocks on transcoding)
- Scene detection, transcription, silence analysis, and keyframe sampling
- Optional local vision + language understanding through Ollama (`qwen3-vl:8b-instruct` recommended)
- Light and dark theme (header switch, preference saved locally)
- Semantic repetition removal (paraphrases, not just identical frames)
- Diversity-aware clip selection against a duration range
- Review timeline: keep, remove, trim, extend, regenerate
- Must-include / must-exclude ranges
- High-quality FFmpeg render (CRF, optional hardware encoders, no watermark)
- Optional subtitles (SRT or burned-in) and social 9:16 output
- Analysis cache so changing 20s → 30s does not re-transcribe
- Cancel, retry from the last valid stage, job history, storage cleanup
- Deterministic fallback when Ollama is offline

---

## Requirements

| Component | Notes |
|-----------|--------|
| Python | 3.11+ (Docker image uses 3.11) |
| Node.js | 20+ only if you build the UI outside Docker |
| FFmpeg / FFprobe | Required for probe, cuts, concat, silence detect |
| Docker / Compose | Recommended deployment |
| Ollama | Optional but recommended for meaning-aware highlights |
| RAM | 8 GB minimum; 16–32 GB comfortable with Whisper `small` + `qwen3-vl:8b-instruct` |
| GPU | Optional. Helps Whisper, Ollama, and NVENC/QSV/AMF/VideoToolbox |
| Disk | Several GB for models, plus space for source videos and cache |

CPU-only operation is supported. It will be slower, especially transcription and visual analysis.

---

## Install Ollama

Ollama is **not** bundled in the Flask image. Install it on the host (recommended for GPU users) or run the optional Compose profile.

### Windows

1. Download the installer from [https://ollama.com/download](https://ollama.com/download)
2. Run it. The service listens on port `11434`
3. Confirm: `ollama --version`

### macOS

```bash
brew install ollama
ollama serve
```

Or install the macOS app from the Ollama website.

### Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
systemctl enable --now ollama
```

---

## Recommended models

This app needs two Ollama models, plus Whisper for speech:

1. **Vision** — score sampled scene frames (what is happening, intro vs demo, on-screen text, action).
2. **Text** — score transcript segments (importance, hooks, filler, repetition).
3. **Whisper** — transcribe speech with word timestamps (`faster-whisper`).

On startup the app reads CPU, RAM, and GPU memory and, in **Automatic** mode, selects a pair that fits. A large amount of system RAM does not select an 8B or 30B model. Those stay available as manual choices. Settings shows the reason, and **Test local AI** checks that Ollama is actually using the GPU.

Ollama remains the runtime for vision and text (CPU, NVIDIA CUDA, AMD ROCm, Vulkan, and Apple Metal). vLLM is a poor fit for a single desktop, especially a CPU-only one. Whisper stays on faster-whisper: CUDA when an NVIDIA card has memory left, and CPU `int8` otherwise.

| Machine | Vision | Text | Whisper |
|---------|--------|------|---------|
| CPU, under 16 GB RAM | `qwen3-vl:2b-instruct` | `qwen3:4b` | `tiny` or `base` |
| CPU, 16 GB RAM or more | `qwen3-vl:4b-instruct` | `qwen3:4b` | `base` |
| GPU under 6 GB free | `qwen3-vl:2b-instruct` or `:4b-instruct` | `qwen3:4b` | `base` on CPU |
| GPU about 6–10 GB | `qwen3-vl:4b-instruct` | `qwen3:4b` | `small` (CUDA only if about 2 GB stays free) |
| GPU about 10–16 GB | `qwen3-vl:8b-instruct` | `qwen3:8b` | `small`, one model loaded at a time |
| GPU 16 GB and above | `qwen3-vl:8b-instruct` | `qwen3:8b` | `large-v3-turbo` on CUDA |
| GPU 24 GB and above, with room free | `qwen3-vl:30b-a3b-instruct` | `qwen3:8b` | `large-v3-turbo` (`large-v3` stays optional) |

`distil-large-v3` is faster still, but it transcribes **English only**. Choose it in Settings only for English videos.

**Use the `-instruct` vision tags.** The bare `qwen3-vl:2b`, `:4b`, `:8b`, `:30b`, and `:latest` tags are the *thinking* versions. They ignore the request to skip reasoning, spend the whole reply budget thinking, and return no frame scores, so every scene falls back to motion scoring after a long wait. Settings warns when the selected vision model is a thinking model. The instruct tags are the same size.

```bash
# CPU or 6–8 GB VRAM
ollama pull qwen3-vl:4b-instruct
ollama pull qwen3:4b

# About 10 GB VRAM or more
ollama pull qwen3-vl:8b-instruct
ollama pull qwen3:8b
```

You can install these from Settings. The app does not download the 30B models unless you choose them.

### Why Qwen3-VL is the vision family

| Criterion | Winner |
|-----------|--------|
| Best Ollama-native vision family for video (2026) | Qwen3-VL |
| Official Ollama `qwen3-vl:latest` | **8B** (~6.1 GB, 256K context, image + text) |
| Lectures, slides, product UI, OCR | 8B over 4B |
| Structured JSON scoring + language quality | 8B + `qwen3:8b` |
| Speed on 6–8 GB VRAM | `qwen3-vl:4b-instruct` |

Qwen3-VL is currently the strongest open vision-language line you can run in Ollama for this workload. It has native long-video / multi-image reasoning, spatial understanding, and OCR across many languages — which matches highlight extraction (keyframes + “is this a demo / slide / speaker / intro?”).

The official Ollama library tags `qwen3-vl:latest` to **`qwen3-vl:8b`**, the thinking build; this app uses the same-size `qwen3-vl:8b-instruct`. Independent local video-frame tests found `qwen3-vl:4b` can match 8B on simple frame tagging at about 2× the speed and half the VRAM (~3.3 GB). The **8B** still wins for this product because tutorials, presentations, and product demos often depend on reading on-screen text; 8B is clearly stronger on document/OCR-style tasks (for example DocVQA-class gaps versus 4B). Pair it with **`qwen3:8b`** for transcript scoring — do not send every frame to a giant model.

Gemma 3 is a weaker fit here: combining visual frames with speech context (exactly what this pipeline does) has been shown to *hurt* Gemma 3 while *helping* Qwen3-VL. Llama 3.2 Vision is behind Qwen3-VL on current local VLM comparisons. Newer Qwen 3.6 vision builds are not a reliable Ollama one-liner yet.

Requires **Ollama 0.12.7+** for Qwen3-VL. Settings shows a warning, and analysis falls back to motion scoring, when Ollama is older or the selected vision model cannot read images.

### Size guide

| Model | Disk (approx.) | When to use |
|-------|----------------|-------------|
| `qwen3-vl:2b-instruct` | ~1.9 GB | CPU / very low VRAM only |
| `qwen3-vl:4b-instruct` | ~3.3 GB | Fast path; 6–8 GB VRAM; many scenes |
| **`qwen3-vl:8b-instruct`** | **~6.1 GB** | **Best default when the GPU has about 8 GB free** |
| `qwen3-vl:30b-a3b-instruct` / `:32b-instruct` | ~20–21 GB | Higher quality if you have the VRAM |
| `qwen3-vl:235b` | ~143 GB | Do not pull unless you know the hardware |

Text model: `qwen3:8b` when the vision model is 8B and the GPU can hold it; `qwen3:4b` otherwise. `qwen3:30b` is an advanced option on a 20 GB GPU, and it should not stay loaded next to a 30B vision model.

Settings installs a model only when you ask, shows download progress, and refuses a download that will not fit on disk.

---

## Verify Ollama

```bash
curl http://localhost:11434/api/tags
```

You should see JSON with a `models` list. If this fails, the app still runs in degraded mode (scenes, transcript heuristics, audio, repetition) and the UI will show that visual AI is unavailable.

---

## Environment configuration

Copy `.env.example` to `.env` and adjust. None of these are mandatory if you are happy with the defaults.

```env
# Bind address for the Flask + Socket.IO server
APP_HOST=0.0.0.0
APP_PORT=5000
FLASK_ENV=production

# Ollama API root. Leave it unset: Docker uses http://host.docker.internal:11434
# and a native install uses http://127.0.0.1:11434.
# Set it for Ollama on another LAN box, or for the Compose "ollama" profile:
# OLLAMA_BASE_URL=http://192.168.1.50:11434
OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct
OLLAMA_TEXT_MODEL=qwen3:8b
OLLAMA_TIMEOUT=900
OLLAMA_MAX_RETRIES=2

DATA_DIR=/app/data
TEMP_DIR=/app/data/tmp
MAX_UPLOAD_SIZE_GB=10

TARGET_MIN_DURATION=20
TARGET_MAX_DURATION=30

ENABLE_WHISPER=true
WHISPER_MODEL=base
ENABLE_SCENE_DETECTION=true
ENABLE_VISUAL_ANALYSIS=true
ENABLE_REPETITION_ANALYSIS=true

OUTPUT_FORMAT=mp4
OUTPUT_VIDEO_CODEC=auto
OUTPUT_CRF=18

KEEP_CACHE=true
CACHE_RETENTION_DAYS=7
DEBUG_ANALYSIS=false
```

`DEBUG_ANALYSIS=true` keeps sampled frames and raw scoring files for tuning. Leave it false in normal use so temporary frames are deleted after analysis.

---

## Docker installation

```bash
cp .env.example .env
docker compose build
docker compose up -d
```

### Access

Open [http://localhost:18527](http://localhost:18527) if you use the default `APP_PUBLISH_PORT` from `.env.example`, or [http://localhost:5000](http://localhost:5000) if you publish container port 5000 directly.

### Optional Ollama container

Default recommended mode: **host Ollama + Dockerized Flask app** (easier GPU use for models).

To run Ollama in Compose as well:

```bash
docker compose --profile ollama up -d
```

Then set `OLLAMA_BASE_URL=http://ollama:11434` in `.env` so the app container talks to the Ollama service. Models persist in the `ollama-models` volume and are not redownloaded on every restart.

---

## Ollama + Docker networking

The Flask container must reach Ollama. Compose includes:

```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

| Host OS | What this does |
|---------|----------------|
| Windows / macOS Docker Desktop | `host.docker.internal` already points at the host. The extra host is harmless. |
| Linux | `host-gateway` maps `host.docker.internal` to the host, so `http://host.docker.internal:11434` reaches host Ollama. |

If `OLLAMA_BASE_URL` still points at `127.0.0.1` inside the container (an older `.env`), the app also tries `host.docker.internal` and the Compose `ollama` service before giving up.

If Ollama listens only on localhost, that is usually enough. If the container still cannot connect:

1. Confirm `curl http://localhost:11434/api/tags` on the host
2. On Linux, allow the Docker bridge to reach port 11434 (firewall / `OLLAMA_HOST=0.0.0.0`)
3. Or point `OLLAMA_BASE_URL` at another machine on your LAN, for example `http://192.168.1.50:11434`

Never commit videos to a public registry. The default Compose volume keeps job files on the local Docker disk.

---

## GPU

### Ollama on the host

Install NVIDIA/AMD/Apple GPU drivers as you normally would for Ollama. The Flask container does not need the GPU for inference if Ollama runs on the host.

### NVIDIA encoding inside the app container

Hardware encoding (NVENC) needs the NVIDIA Container Toolkit and an image with FFmpeg linked to NVENC. The default CPU image still encodes with `libx264`. To pass a GPU into Compose, uncomment the `deploy.resources.reservations.devices` block on the service that should see the GPU (usually `ollama`, sometimes `app`).

### CPU-only mode

Leave Ollama unset or offline and the app still analyzes video with scene, audio, and transcript heuristics. Automatic mode selects `qwen3-vl:4b-instruct` or `qwen3-vl:2b-instruct` instead of a large model. Set `ENABLE_VISUAL_ANALYSIS=false` to skip vision calls entirely. Transcription uses `faster-whisper` on CPU (`int8`). Rendering uses libx264.

If a GPU is listed in Settings but **Test local AI** reports CPU, Ollama did not place the model in VRAM. Check the driver, run Ollama on the machine with the GPU, or pick a smaller model. On AMD, restart Ollama with `OLLAMA_VULKAN=1` when ROCm cannot load Qwen3-VL.

On a CPU-only machine, model replies cost far more time than reading prompts, so these Ollama server settings help without lowering quality: `OLLAMA_FLASH_ATTENTION=1` and `OLLAMA_KV_CACHE_TYPE=q8_0` (less memory for the same context). Avoid `q4_0` for the KV cache; it lowers accuracy. Each analysis log ends with per-stage timings and the token counts of every model stage, so you can see where the time goes.

### Optional GPU on another machine

Vision and transcript scoring can run on Ollama on another computer with a GPU: a desktop on your network, or a rented cloud GPU. Whisper, FFmpeg, and rendering stay on this machine, so CPU-only operation keeps working without it.

```env
OLLAMA_BASE_URL=http://192.168.1.50:11434       # the GPU machine
OLLAMA_FALLBACK_URL=http://127.0.0.1:11434      # optional: use this computer's Ollama while that one is offline
# OLLAMA_API_KEY=...                             # optional: sent as a Bearer token, for a proxy that checks one
```

- On the GPU machine, start Ollama with `OLLAMA_HOST=0.0.0.0` and pull the models there.
- The app sends the CPU thread setting only to an Ollama on this computer. A remote server uses its own default.
- When the remote server stops answering and `OLLAMA_FALLBACK_URL` is set, the job continues on the fallback. The app returns to the remote server once it answers again (checked at most once a minute). Without a fallback, scoring falls back to heuristics as before.
- Automatic setup picks models for **this** computer's hardware. To use bigger models on the remote GPU, choose them in Settings (custom mode).
- Ollama has no login of its own. Do not expose port 11434 to the internet. Reach the server over a VPN (Tailscale, WireGuard), an SSH tunnel (`ssh -L 11434:localhost:11434 gpu-box`), or an HTTPS reverse proxy that checks `OLLAMA_API_KEY`.
- Sampled frames and transcript text are sent to that server. Use only machines you trust.

---

## Native (non-Docker) install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
copy .env.example .env   # Windows
python run.py
```

Frontend dev mode (UI on 5173, API proxied to 5000):

```bash
python run.py
cd frontend && npm install && npm run dev
```

System check:

```bash
python scripts/check_system.py
```

---

## How analysis works

The pipeline is hierarchical. It does **not** send every frame to the LLM.

1. Probe metadata (FFprobe)
2. Detect scenes (PySceneDetect)
3. Extract audio (FFmpeg)
4. Transcribe (faster-whisper, local)
5. Detect silence / dead air
6. Sample a few representative frames per scene
7. Optional Ollama vision on those frames only
8. Score transcript segments (Ollama text model, with heuristics as fallback)
9. Generate many candidate windows and snap to sentence / pause / scene boundaries
10. Remove semantically repeated ideas
11. Diversity-aware selection into the target duration range
12. Show a clip plan for review
13. Render with FFmpeg (AI never emits shell commands)

Quality is preferred over padding the duration with weak clips. If the source is already shorter than the target, it is not artificially expanded.

---

## Tests

```bash
pip install pytest pydantic pydantic-settings flask flask-cors flask-socketio sqlalchemy numpy scikit-learn requests python-dotenv
python -m pytest tests -q
```

Unit tests cover timestamps, scoring, repetition, duration optimization, FFmpeg argv construction, JSON repair, probe/silence parsing, and API validation. Integration tests that need a real GPU or Whisper weights are skipped unless those tools are present.

---

## Troubleshooting

| Problem | What to check |
|---------|----------------|
| Ollama unavailable | `curl http://localhost:11434/api/tags`; URL in Settings; Docker extra_hosts |
| Model missing | UI shows “not installed” + Install model, or `ollama pull qwen3-vl:8b-instruct` |
| FFmpeg missing | Install FFmpeg so `ffmpeg` and `ffprobe` are on `PATH` |
| GPU unavailable | App continues on CPU; encoding falls back to libx264 |
| Permission denied | Data directory must be writable (`DATA_DIR`) |
| Out of disk space | Settings → Storage; clear tmp/cache; retention days |
| Unsupported codec | Re-wrap/transcode the source locally; FFmpeg must decode it |
| Very slow inference | Settings → Use recommended setup, or pick `qwen3-vl:4b-instruct` / Whisper `base` or `tiny` |
| GPU listed, test says CPU | Driver, Ollama on the GPU machine, a smaller model, or `OLLAMA_VULKAN=1` on AMD |
| Container cannot reach host Ollama | Linux `host-gateway`, `OLLAMA_HOST=0.0.0.0`, or a LAN URL |
| Vision takes long and every scene says motion fallback | The vision model is a thinking tag (`qwen3-vl:4b`); install and choose `qwen3-vl:4b-instruct` |
| Job failed mid-vision | Retry continues from cache; one failed scene uses fallback scoring |
| Analysis ran while Ollama or Whisper was down | Start them and run the analysis again: scenes and transcript segments that used fallback scores are re-scored, and a failed transcription is retried |

---

## Privacy

- No cloud AI APIs (no OpenAI, no Gemini, no SaaS video APIs)
- The only network calls are to the Ollama URLs you configure (typically localhost or LAN). A remote Ollama receives sampled frames and transcript text
- Uploads are stored under `data/jobs/<id>/` as files, not in SQLite blobs

---

## License

MIT — see `LICENSE`.
