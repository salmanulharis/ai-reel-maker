# Performance on CPU-only machines and external GPU options

Investigation notes, 2026-10-14. Measured on the development PC: Intel 12th-gen, 6 cores / 12 threads, 15.8 GB RAM, no NVIDIA GPU, Ollama 0.40.3 on the same machine.

Prices, free allowances and limits of external services change often. Check each provider's pricing page before relying on the figures below.

## 1. Where the time goes on CPU (measured)

| Step | Cost | Main driver |
|---|---|---|
| Vision (`qwen3-vl:2b-instruct`) | About 30 s per scene | Each frame costs about 1,000 prompt tokens whatever its resolution (Ollama upscales Qwen3-VL images), plus about 80 reply tokens |
| Transcript scoring (`qwen3:4b`) | About 100 s per 12 segments | About 95% is reply generation: about 45 tokens per segment (the model pretty-prints JSON) at about 5 tokens/s |
| Whisper, scene detection, frame extraction | Minutes for long videos | Decoding the video and transcription |

On CPU, reply tokens dominate model time. Reading the prompt is comparatively cheap, and a repeated prompt prefix is cached by Ollama.

## 2. CPU options

### Already done, no quality loss

- **Instruct vision tags.** The bare `qwen3-vl:2b/4b/8b/30b/latest` tags are thinking models. They ignore `think: false`, spend the whole reply budget on hidden reasoning and return empty content, so every scene fell back to motion scoring. On a 2-minute video that wasted 12.6 minutes. The catalog, defaults and README now use the `-instruct` tags, and Settings warns about thinking models.
- **Batches fit the context window.** Transcript batches are sized by an estimated token budget (`plan_text_batches`) and vision batches by about 1,100 tokens per image (`fitting_vision_batch`). Previously, the default "4 images at a 4,096 context" and "2 images at 2,048" overflowed.
- **Reply budget per transcript segment raised from 40 to 48 tokens.** Measured usage was 42–46, so full batches were being cut off and generated again as halves.
- **Frames are extracted with 2–4 FFmpeg calls at once.**
- **Instrumentation.** Each job logs per-stage timings and token usage, and warns when a call fills the context window. `scripts/pipeline_benchmark.py` runs the pipeline with timings.

### Still available

| Option | Expected gain | Quality risk |
|---|---|---|
| Turn off "Understand what is on screen" for speech-heavy videos (user choice in Settings) | Removes about 30 s per scene | Loses visual judgement; fine for talks and lectures |
| Run vision only on scenes that could be selected: score transcript and motion first, always include scenes without speech, keep a generous top-K | 50–70% of vision time on long videos | Low to medium; check recall against current results |
| Skip AI scoring for near-duplicate frames (perceptual hash, strict threshold) | 0–50%, depends on content (slides, screen recordings) | Low |
| Ollama server settings `OLLAMA_FLASH_ATTENTION=1`, `OLLAMA_KV_CACHE_TYPE=q8_0` | Small on CPU; frees memory | Negligible (avoid `q4_0`) |
| Whisper `beam_size=1` (currently the default 5) | About 1.5–2× faster transcription | Slightly higher word error rate; measure first |
| Shorter reply keys with a 0–10 scale | About 20% on transcript scoring | **Rejected:** slightly worse agreement with a `qwen3:14b` reference (top-8 overlap 5/8 vs 6/8) |
| Smaller or more heavily quantized models (Q3 or lower), Whisper `tiny`, frames under about 448 px | Faster | **Avoid:** clear loss of accuracy |

Running more requests at once (`OLLAMA_NUM_PARALLEL`), or overlapping Whisper with vision, gives nothing on a CPU-only machine: both compete for the same cores.

## 3. External GPU and cloud API options

| Option | Speed | Quality | Free amount | Setup | Main limits |
|---|---|---|---|---|---|
| **Lightning AI** Studio | GPU, roughly 10–20× faster on vision | 8B models | About 15 credits/month, quoted as about 80 hours on interruptible (spot) GPUs. Not confirmed; GPU may need phone verification or switching the machine type | **No code:** SSH tunnel plus `.env` | Free account showed CPU only during testing |
| **Modal** | GPU | 8B models | About $30/month in credits, roughly 50 h on a T4, renewed monthly | Small: deploy script plus custom auth headers in the app | Cold start after idle |
| **Ollama Cloud** | Fast | Large models; vision on the free plan unclear | Small monthly allowance, 1 concurrent request | Two `.env` lines | Frames and transcript go to Ollama's servers (stated: not retained, not used for training); compatibility with the app's health checks untested |
| **Kaggle** notebooks | GPU | 8B models | About 30 GPU hours/week | Public tunnel (ngrok or Cloudflare) | Terms limit use to personal, non-commercial; sessions end after 9–12 h; tunnel has no auth |
| **Gemini API** (free tier) | Fast | High | Daily caps disputed across sources (about 20–50 up to 1,500 requests/day) | New code (OpenAI-compatible or Gemini client) | Free-tier data may be used to improve Google's products; image input on free models unclear |
| **Groq** (free tier) | Very fast | Medium; few vision models | About 1,000 requests/day (secondary sources) | New code (OpenAI-compatible) | Limited vision support |
| **OpenRouter** free models | Varies | Varies | 50 requests/day (1,000 after buying $10 of credits) | New code (OpenAI-compatible) | Free model IDs change often; Qwen VL is paid |
| Google Colab | — | — | — | — | **Not suitable:** its FAQ bans remote proxies and web services unrelated to interactive use, on all runtimes |

**Request budget:** one 1-hour video needs roughly 150–300 vision calls plus about 20 transcript calls. Free API caps cover about one long video a day at best. Sending frames to a hosted API also goes against the README's "videos stay local" promise.

### How a remote GPU works with this app

- The video file never leaves the PC. FFmpeg, scene detection, Whisper and rendering stay local.
- Only one 512 px JPEG per scene (about 30–50 KB) and the transcript text are sent. A 5 GB, 1-hour video with about 300 scenes sends about 10–15 MB in total.
- Network latency is negligible next to model time (seconds per call).
- Ollama has no authentication. Reach a remote Ollama over a VPN or SSH tunnel, or an HTTPS proxy that checks a token. Never expose port 11434 publicly.

## 4. Recommendation

1. **Default: local CPU.** Keep the fixes above. Turn off visual analysis for talking-head videos. Next CPU improvement: vision only on scenes that could be selected (Section 2).
2. **Best free GPU:**
   - **Lightning AI**, if the free credits unlock a GPU. Works today with no code changes.
   - **Modal** otherwise. The most reliable free option, within its terms, private to the account. Needs small app changes.
3. **Hosted AI APIs** (Gemini, Groq, OpenRouter): only as an optional, clearly labelled backend. The free tiers are too small and conflict with privacy. Low priority.

## 5. Integration approach (local CPU stays the fallback)

**Already in the app:**
- `OLLAMA_BASE_URL` can point at a remote Ollama.
- `OLLAMA_FALLBACK_URL` names a second Ollama (usually this PC) used while the remote one is unreachable. The client returns to the configured URL once it answers again (checked at most once a minute).
- `OLLAMA_API_KEY` is sent as a Bearer token and never returned by the settings API.
- The CPU thread setting is sent only to an Ollama on this machine.
- If no Ollama is reachable, scoring falls back to heuristics, so the job still finishes.

**Proposed, in order:**

1. **Custom auth headers** (`OLLAMA_HEADERS`) for Modal's private endpoints and other authenticating proxies.
2. **Show the active backend** (remote, fallback, heuristics) in the Settings status.
3. **When the GPU is remote:** run Whisper on the local CPU while vision runs on the GPU, and send 2–4 batches at once (needs `OLLAMA_NUM_PARALLEL` on the server).
4. **Optional:**
   - a separate backend per role (for example vision remote, transcript local);
   - model recommendations based on the remote GPU instead of the local hardware.
5. **Optional, later:** an OpenAI-compatible provider using the same `chat()` contract and JSON schema, behind an explicit "send frames to a cloud service" opt-in in Settings, with the same fallback chain.

### Setting up a remote GPU (Lightning example)

On the GPU machine:

```bash
curl -fsSL https://ollama.com/install.sh | sh
export OLLAMA_MODELS=/teamspace/studios/this_studio/ollama-models   # Studio's persistent folder (confirm path)
nohup ollama serve > ~/ollama.log 2>&1 &
ollama pull qwen3-vl:8b-instruct
ollama pull qwen3:8b
```

On this PC, keep an SSH tunnel open (port 11435 locally, because the local Ollama uses 11434):

```powershell
ssh -N -L 11435:localhost:11434 s_xxxxx@ssh.lightning.ai
curl.exe http://127.0.0.1:11435/api/tags
```

In `.env`:

```env
OLLAMA_BASE_URL=http://127.0.0.1:11435
OLLAMA_FALLBACK_URL=http://127.0.0.1:11434
```

Then in Settings, choose **Choose models myself**, select `qwen3-vl:8b-instruct` and `qwen3:8b`, and run **Test AI**.

Because the tunnel makes the GPU look like `127.0.0.1`, the app treats it as local:
- It sends the local thread count (minor impact with full GPU offload).
- Its free-disk check before a model install looks at the local disk, not the Studio's.

Stop the Studio when done to avoid charges.

## Sources

**Lightning AI**
- [Lightning AI: SSH access](https://lightning.ai/docs/platform/build/ai-studio/ssh-access)
- [Lightning AI: Connect local IDE](https://lightning.ai/docs/overview/ai-studio/connect-local-ide)
- [Lightning AI: On-start actions](https://lightning.ai/docs/platform/build/ai-studio/on-start-actions)
- [Lightning AI free tier (usagepricing.com)](https://usagepricing.com/blueprint/lightning-ai)
- [Free cloud GPUs in 2026 (gpuperhour.com)](https://gpuperhour.com/blog/free-cloud-gpus-and-credits)

**Modal**
- [Modal free tier (aicreditmart.com)](https://aicreditmart.com/ai-credits-providers/modal-free-tier-how-to-get-30-month-in-compute-credits-2026/)
- [Modal pricing explained (Beam)](https://www.beam.cloud/blog/modal-pricing-explained)

**Kaggle**
- [Kaggle: efficient GPU usage](https://www.kaggle.com/docs/efficient-gpu-usage)
- [Kaggle free tier 2026 (agentdeals.dev)](https://agentdeals.dev/vendor/kaggle)

**Ollama Cloud**
- [Ollama pricing](https://ollama.com/pricing)
- [Ollama API introduction](https://docs.ollama.com/api/introduction)
- [Ollama privacy policy](https://registry.ollama.ai/privacy)
- [Ollama Cloud free vs Pro limits (DevToolHub)](https://devtoolhub.com/ollama-cloud-free-vs-pro-limits-pricing-2026/)

**Gemini API**
- [Gemini API pricing 2026 (agentdeals.dev)](https://agentdeals.dev/gemini-api-pricing-2026)
- [Gemini API free tier limits (tinkerllm.com)](https://tinkerllm.com/blog/gemini-api-free-tier-limits-rate-quotas/)
- [Gemini API free plan (CostBench)](https://www.costbench.com/software/llm-api-providers/google-gemini-api/free-plan/)

**Groq**
- [Groq pricing 2026 (eesel.ai)](https://eesel.ai/blog/groq-pricing)
- [Groq vs Together AI (kunalganglani.com)](https://www.kunalganglani.com/blog/groq-vs-together-ai-inference.md)

**OpenRouter**
- [OpenRouter free tier (benchlm.ai)](https://benchlm.ai/md/free-tier/openrouter.md)
- [OpenRouter free credits limit (aireiter.com)](https://aireiter.com/blog/openrouter-free-credits-limit)
- [OpenRouter: Qwen VL Max](https://openrouter.ai/qwen/qwen-vl-max)

**Google Colab**
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Decrypt: Colab restricts remote UIs (2023)](https://decrypt.co/197428?p=197428)
