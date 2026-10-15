/**
 * In-app help. Each article is structured data rendered by pages/Docs.tsx, so adding or editing
 * help needs no extra library and stays in one place.
 *
 * Keep it accurate: describe only what the app does, use the exact labels shown in the UI, and
 * update the matching article when a setting, model, or fallback changes. Measured numbers come
 * from a 6-core / 12-thread processor with 16 GB of memory and no graphics card.
 *
 * Inline text supports **bold** and `code`.
 */

export type DocBlock =
  | { kind: "p"; text: string }
  | { kind: "h"; text: string }
  | { kind: "steps"; items: string[] }
  | { kind: "list"; items: string[] }
  | { kind: "tip" | "note" | "warning"; text: string }
  | { kind: "code"; text: string; label?: string }
  | { kind: "table"; head: string[]; rows: string[][] }
  | { kind: "settings"; section: string; label: string }
  | { kind: "see"; article: string; label: string };

export interface DocArticle {
  title: string;
  summary: string;
  /** Extra words people may search for that do not appear in the text. */
  keywords?: string;
  blocks: DocBlock[];
}

export const DOC_ARTICLES = {
  "how-it-works": {
    title: "How a highlight is made",
    summary: "The steps the app runs, and which ones use AI.",
    keywords: "pipeline steps stages overview",
    blocks: [
      { kind: "p", text: "Everything runs on your computer (or on a computer you choose). Your videos are never uploaded to an online AI service." },
      {
        kind: "steps",
        items: [
          "**Reading the video** and **finding scene changes** (FFmpeg and PySceneDetect, no AI).",
          "**Extracting the sound** and **writing down what is said** (Whisper, a speech-to-text model). Skipped when speech is turned off.",
          "**Finding pauses** and **capturing pictures from each scene**, then measuring picture quality (sharp, dark, blurry, black) and loudness.",
          "**Fast picture matching** (optional, SigLIP) ranks scenes in a fraction of a second each.",
          "**Understanding what is said**: an AI model scores each sentence for importance.",
          "**Understanding what is on screen**: an AI video model looks at the most promising scenes.",
          "**Choosing the clips**: repeats are removed and the best moments are fitted to your length.",
          "You review the clips, then the app **creates your video** with FFmpeg.",
        ],
      },
      { kind: "p", text: "If an AI step is unavailable, the app does not stop: it falls back to simpler scoring based on motion, sound, picture quality, and word counts, and says so in the detailed log." },
      { kind: "see", article: "where-time-goes", label: "See which steps take the longest" },
    ],
  },

  requirements: {
    title: "What you need",
    summary: "Software, memory, and disk space for local AI.",
    keywords: "install prerequisites ram disk python ffmpeg ollama",
    blocks: [
      {
        kind: "table",
        head: ["Item", "Needed for", "Notes"],
        rows: [
          ["FFmpeg", "Everything", "Must be installed and on PATH."],
          ["Ollama 0.12.7 or newer", "Understanding what is said and on screen", "Optional, but highlights are much better with it."],
          ["Memory (RAM)", "AI models", "8 GB works with small models; 16 GB is comfortable."],
          ["Disk space", "Models and videos", "Models take 2-10 GB; keep about twice your video's size free."],
          ["Graphics card", "Speed", "Optional. Everything works on the processor, more slowly."],
        ],
      },
      { kind: "note", text: "Python packages are installed with `pip install -r requirements.txt`. Fast picture matching uses onnxruntime and tokenizers, which come with that install." },
      { kind: "see", article: "setup-ollama", label: "Set up the AI models" },
    ],
  },

  "cpu-speed": {
    title: "Fastest settings without a graphics card",
    summary: "A checklist to make analysis quicker on a processor-only computer.",
    keywords: "slow speed faster cpu performance optimize quick",
    blocks: [
      { kind: "p", text: "Work through this list from the top. The first items save the most time and keep quality where it matters." },
      {
        kind: "steps",
        items: [
          "Turn on **Automatically optimize for my system**. It picks models, threads, and memory sizes that fit your computer and explains each choice.",
          "Keep **Which scenes the video model looks at** on **Only scenes that could make the highlight**. The video model is the slowest step (about 30 seconds per scene on a typical processor), so looking only at promising scenes saves the most time.",
          "Download **Fast picture matching**. It ranks scenes in about a third of a second each, so the shortlist picks better scenes for the video model.",
          "For talks, interviews, and lectures, turn off **Understand what is on screen**. Speech is still analyzed; pictures are judged by quick checks. Analysis often drops from minutes to seconds.",
          "For music, sports, gaming, and travel videos, tick **Visuals and sound only** on the Create page. No transcription is done, which saves a lot of time.",
          "Keep **Frame rate** on **Same as the original** and leave **Smooth motion** off. Smooth motion can make creating the video several times slower.",
          "Close other heavy programs while a highlight is being made. AI models need free memory and processor time.",
        ],
      },
      { kind: "warning", text: "Smaller AI models are faster but understand less. Automatic optimization only chooses a smaller model when your memory requires it, and tells you when it does." },
      { kind: "settings", section: "optimize", label: "Open Automatic optimization" },
      { kind: "see", article: "where-time-goes", label: "Find out what is slow on your computer" },
    ],
  },

  "where-time-goes": {
    title: "Find what is slowing you down",
    summary: "Read the step times, the detailed log, and the AI test.",
    keywords: "bottleneck timing log tokens per second measure",
    blocks: [
      { kind: "p", text: "While a highlight is being made, the progress page shows the time for every step on the right, and the total time so far at the top. Look for the step with the largest time." },
      {
        kind: "table",
        head: ["Step", "Typical time without a graphics card", "What helps"],
        rows: [
          ["Understanding what is on screen", "About 30 s per scene", "Shortlist, Fast picture matching, or turning it off"],
          ["Understanding what is said", "About 5-10 words per second of reply", "A smaller speech model, or Visuals and sound only"],
          ["Writing down what is said", "About 1/7 of the video's length with the base model", "A smaller Whisper model, or turning speech off"],
          ["Fast picture matching", "About 0.3 s per scene", "Already fast"],
          ["Creating your video", "Depends on length and quality", "Same frame rate as the original, Smooth motion off"],
        ],
      },
      { kind: "h", text: "The detailed log" },
      { kind: "p", text: "Open **Detailed log** on the progress page. Useful lines:" },
      {
        kind: "list",
        items: [
          "`Visual AI: 8 of 24 scenes could make the highlight` shows the shortlist at work.",
          "`Visual AI: 8 calls, ... reply tokens in 64.0s (9.8 tok/s)` shows how fast the video model answers. Under about 5 tok/s, close other programs or use a smaller model.",
          "`filled the model's context window` means a request ran out of working memory; see Threads and memory.",
          "`Stage timings: ...` lists every step's time at the end.",
        ],
      },
      { kind: "h", text: "Test AI" },
      { kind: "p", text: "**Settings → AI models → Test AI** runs each model once and shows its speed and whether it runs on the graphics card (GPU) or the processor (CPU)." },
      { kind: "settings", section: "ai", label: "Open AI models" },
    ],
  },

  "choose-models": {
    title: "Choose models for your computer",
    summary: "Which AI models fit your memory and graphics card.",
    keywords: "qwen3 qwen3-vl 2b 4b 8b 30b model size ram vram",
    blocks: [
      { kind: "p", text: "With automatic optimization on, the app chooses for you using these rules. To choose yourself, click **Use this model** on a model card in AI models (this turns automatic optimization off)." },
      {
        kind: "table",
        head: ["Your computer", "Video model", "Speech model", "Speech to text"],
        rows: [
          ["No graphics card, under 16 GB memory", "qwen3-vl:2b-instruct", "qwen3:4b-instruct-2507-q4_K_M", "tiny or base"],
          ["No graphics card, 16 GB or more", "qwen3-vl:4b-instruct", "qwen3:4b-instruct-2507-q4_K_M", "base"],
          ["Graphics card with under 6 GB free", "qwen3-vl:2b-instruct or :4b-instruct", "qwen3:4b-instruct-2507-q4_K_M", "base on the processor"],
          ["Graphics card with about 6-10 GB", "qwen3-vl:4b-instruct", "qwen3:4b-instruct-2507-q4_K_M", "small"],
          ["Graphics card with 10 GB or more", "qwen3-vl:8b-instruct", "qwen3:8b", "small or large-v3-turbo"],
          ["Graphics card with 24 GB or more", "qwen3-vl:30b-a3b-instruct", "qwen3:8b", "large-v3-turbo"],
        ],
      },
      { kind: "warning", text: "Always use the **-instruct** video models. The plain `qwen3-vl:2b`, `:4b`, `:8b` and `:latest` models are \"thinking\" versions: they spend their whole answer thinking and return no scores, so every scene falls back to quick scoring after a long wait. Settings warns you if one is selected." },
      { kind: "tip", text: "Lots of memory but no graphics card does not make the 8B models a good idea: they fit, but run about twice as slowly as the 4B ones." },
      { kind: "settings", section: "ai", label: "Open AI models" },
    ],
  },

  "threads-memory": {
    title: "Threads and memory",
    summary: "What the Speed and memory settings do, and when to change them.",
    keywords: "cpu threads context num_ctx batch memory ram out of memory",
    blocks: [
      { kind: "p", text: "These are set for you by automatic optimization. Change them only if you see a problem; changing one turns automatic optimization off." },
      {
        kind: "table",
        head: ["Setting", "What it does", "When to change"],
        rows: [
          ["Processor threads for AI", "How many processor threads the AI may use", "Default is your core count minus one (at most 8), so your computer stays responsive. Use all cores only if nothing else runs."],
          ["Scenes per AI request", "Scenes sent to the video model at once", "Lower it if the log says the context window filled up. The app also lowers it automatically when it would not fit."],
          ["Video model memory (tokens)", "Working memory for the video model", "Each picture needs about 1,100 tokens. 2048 fits 1 scene, 4096 fits 2. Higher uses more RAM."],
          ["Speech model memory (tokens)", "Working memory for the speech model", "Raise it (for example to 4096) if the log says speech requests ran out of room and you have 16 GB or more."],
        ],
      },
      { kind: "note", text: "Making pictures smaller does not speed up the video model: it enlarges every picture to the same size before reading it." },
      { kind: "settings", section: "performance", label: "Open Speed and memory" },
    ],
  },

  "ollama-tuning": {
    title: "Ollama settings for extra speed",
    summary: "Two Ollama options that save memory without lowering quality.",
    keywords: "flash attention kv cache q8_0 environment variable setx",
    blocks: [
      { kind: "p", text: "These are settings of the Ollama app itself, not of this app. They save memory (and on some computers time) without a noticeable quality loss." },
      { kind: "code", label: "Windows (PowerShell), run once", text: "setx OLLAMA_FLASH_ATTENTION 1\nsetx OLLAMA_KV_CACHE_TYPE q8_0" },
      { kind: "code", label: "macOS / Linux", text: "export OLLAMA_FLASH_ATTENTION=1\nexport OLLAMA_KV_CACHE_TYPE=q8_0" },
      { kind: "steps", items: ["Run the commands above.", "Quit Ollama from the system tray (or menu bar) and start it again.", "Run **Test AI** to confirm everything still works."] },
      { kind: "warning", text: "Do not use `q4_0` for the cache: it saves more memory but lowers accuracy." },
    ],
  },

  "setup-ollama": {
    title: "Set up the AI models (Ollama)",
    summary: "Install Ollama and the video and speech understanding models.",
    keywords: "install ollama pull models download setup",
    blocks: [
      {
        kind: "steps",
        items: [
          "Download Ollama from ollama.com/download and install it. On Windows it starts automatically and listens on port 11434.",
          "Open **Settings → Automatic optimization** and turn on **Automatically optimize for my system** (or pick models yourself with **Use this model** in AI models).",
          "Open **Settings → AI models** and click **Install recommended models**. Progress is shown on each model card.",
          "Click **Test AI**. Each model should show a speed and GPU or CPU.",
        ],
      },
      { kind: "p", text: "You can also install models from a terminal:" },
      { kind: "code", label: "Example for a computer with 16 GB and no graphics card", text: "ollama pull qwen3-vl:4b-instruct\nollama pull qwen3:4b-instruct-2507-q4_K_M" },
      { kind: "tip", text: "Expected result: the header shows **AI ready**. If it says **AI offline**, Ollama is not running." },
      { kind: "settings", section: "ai", label: "Open AI models" },
    ],
  },

  "setup-whisper": {
    title: "Speech to text (Whisper)",
    summary: "Choose and download a speech-to-text model, or turn speech off.",
    keywords: "whisper transcription subtitles tiny base small medium large turbo speech off",
    blocks: [
      { kind: "p", text: "Whisper writes down what is said. It runs inside the app (no Ollama needed) and downloads its model the first time it is used, or when you click **Download** in Settings." },
      {
        kind: "table",
        head: ["Model", "Download", "Good for"],
        rows: [
          ["tiny", "80 MB", "Very slow computers; more mistakes"],
          ["base", "150 MB", "The usual choice without a graphics card"],
          ["small", "500 MB", "Better accuracy; still fine on a processor"],
          ["large-v3-turbo", "1.6 GB", "NVIDIA graphics cards"],
        ],
      },
      { kind: "h", text: "Turning speech off" },
      { kind: "list", items: ["For every highlight: **Settings → Speech to text → Use speech**.", "For one highlight: tick **Visuals and sound only** on the Create page. Subtitles are turned off with it, because they come from the transcript."] },
      { kind: "note", text: "Turning speech back on reuses earlier transcripts, so nothing is transcribed twice." },
      { kind: "settings", section: "speech", label: "Open Speech to text" },
    ],
  },

  "setup-image-matching": {
    title: "Fast picture matching (SigLIP)",
    summary: "A small picture model that ranks scenes quickly.",
    keywords: "siglip onnx image embedding clip picture ranking lookalike duplicate",
    blocks: [
      { kind: "p", text: "Fast picture matching compares each scene's picture with short descriptions such as \"an exciting action moment\" or \"a black screen\". It takes about 0.3 seconds per scene on a processor, about 90 times faster than the video model." },
      { kind: "h", text: "What it does" },
      { kind: "list", items: ["Ranks scenes, so the video model looks at the most promising ones.", "Spots scenes that look like an earlier one and skips them.", "Matches your focus text (Custom style) to pictures.", "Improves the quick scores of scenes the video model did not look at."] },
      { kind: "warning", text: "It cannot judge what happens in a scene, so it never replaces the video model's scores. On its own it can rank an empty room as interesting." },
      { kind: "h", text: "Set it up" },
      {
        kind: "steps",
        items: [
          "Open **Settings → AI models → Fast picture matching**.",
          "Click **Download** (about 780 MB, once). The tag changes to **Downloaded**.",
          "Keep the switch on. The next highlight's log shows a line like `Image matching (CPU): 24 scenes in 7.5s`.",
        ],
      },
      { kind: "note", text: "It runs on the processor. With an NVIDIA (CUDA) or Windows DirectML build of onnxruntime installed, it uses the graphics card instead, and falls back to the processor if the card fails." },
      { kind: "settings", section: "ai", label: "Open AI models" },
    ],
  },

  "setup-embeddings": {
    title: "Meaning matching (optional)",
    summary: "Finds repeated ideas said in different words, and matches your focus to speech.",
    keywords: "nomic-embed-text embeddings repetition paraphrase focus",
    blocks: [
      { kind: "p", text: "A small Ollama model compares the meaning of sentences. It helps remove repeated ideas worded differently and raises moments whose words match your focus. It is fast on any computer." },
      { kind: "steps", items: ["Open **Settings → AI models → Meaning matching** and click **Install**, or run the command below.", "Nothing else to set: the next highlight uses it automatically."] },
      { kind: "code", text: "ollama pull nomic-embed-text" },
      { kind: "note", text: "Without it, the app compares words instead. Everything still works." },
    ],
  },

  "leave-out": {
    title: "Leaving things out",
    summary: "Keep sponsor talk, title cards, dark shots, intros, and your own words out of the highlight.",
    keywords: "exclude avoid never remove skip unwanted what to leave out sponsor swearing filler screens",
    blocks: [
      { kind: "p", text: "Under **What to leave out**, pick the kinds of moments you do not want. You can also type your own words, separated by commas." },
      {
        kind: "table",
        head: ["Choice", "How it is found"],
        rows: [
          ["Filler & greetings, Sponsor & ad talk, Swearing", "From what is said (needs speech)."],
          ["Long talking heads", "Continuous speech while little happens in the picture; picture matching confirms a close-up face."],
          ["Dark or blurry shots, Silence & dead air", "From picture quality and quiet stretches. No AI needed."],
          ["Text, titles & slides, Screens & phones, Empty scenes", "Picture matching, and the video model when it is installed."],
          ["Intros & outros", "The first and last seconds; full strength with a greeting, title card, or sign-off."],
          ["Your own words", "Exact words in what is said, similar meaning with meaning matching, and the pictures with picture matching."],
        ],
      },
      { kind: "p", text: "**Avoid when possible** ranks matching moments much lower, so they are used only when nothing else fits. **Always remove** removes clear matches; the highlight can then come out shorter. If every moment matches, they are ranked lower instead, so you still get a highlight." },
      { kind: "note", text: "Exact time ranges to skip are under **More options → Never include**. An **Always include** range is always kept, even when it matches something to leave out." },
      { kind: "see", article: "setup-image-matching", label: "Picture matching" },
    ],
  },

  "hardware-detection": {
    title: "How your hardware is detected",
    summary: "What the app checks, and how to confirm the graphics card is used.",
    keywords: "gpu detection nvidia-smi rocm vram cpu cores check",
    blocks: [
      { kind: "p", text: "When the app starts, it reads your processor, memory, free disk space, and graphics card. NVIDIA cards are read with `nvidia-smi`, AMD cards with `rocm-smi`, Apple Silicon from the system, and other cards by name only." },
      { kind: "list", items: ["See the result in **Settings → This computer**. Click **Check computer again** after changing hardware or drivers.", "The badge at the top of **AI models** shows where models actually run: **GPU-accelerated**, **Partial GPU** (part of the model did not fit), or **CPU-only**.", "**Test AI** measures this for real by loading each model."] },
      { kind: "note", text: "If a graphics card's memory cannot be read (common with Intel and some AMD cards on Windows), the app assumes the processor and suggests running Test AI." },
      { kind: "settings", section: "system", label: "Open This computer" },
    ],
  },

  "gpu-setup": {
    title: "Use a graphics card (GPU)",
    summary: "What uses the graphics card, and what each brand needs.",
    keywords: "nvidia amd intel apple cuda rocm vulkan directml metal gpu acceleration",
    blocks: [
      {
        kind: "table",
        head: ["Part", "Uses a graphics card when", "Otherwise"],
        rows: [
          ["Video and speech understanding (Ollama)", "Automatic with NVIDIA, AMD, and Apple Silicon", "Processor"],
          ["Speech to text (Whisper)", "\"Speech to text runs on\" is set to NVIDIA graphics card (CUDA) and CUDA 12 with cuDNN is installed", "Processor; a CUDA failure falls back to the processor automatically"],
          ["Fast picture matching", "onnxruntime-gpu (NVIDIA) or onnxruntime-directml (Windows, any brand) is installed", "Processor"],
          ["Creating your video", "Not used: highlights are made at the highest quality, which uses the processor encoder", "Processor (libx264)"],
        ],
      },
      { kind: "h", text: "By brand" },
      { kind: "list", items: ["**NVIDIA:** install the latest driver. Ollama uses the card automatically.", "**AMD:** if Test AI still says CPU, start Ollama with `OLLAMA_VULKAN=1`.", "**Apple Silicon:** Ollama uses the GPU automatically; memory is shared with the system.", "**Intel:** run **Test AI** to see whether Ollama uses the card on your computer; picture matching can use DirectML."] },
      { kind: "warning", text: "`onnxruntime-gpu` and `onnxruntime-directml` replace the normal `onnxruntime` package (which speech to text also uses). Uninstall `onnxruntime` first, install one of them, and run a highlight to check speech to text still works." },
      { kind: "see", article: "fallbacks", label: "What happens when the graphics card fails" },
    ],
  },

  fallbacks: {
    title: "What happens when something fails",
    summary: "How the app keeps working when a model or the graphics card is unavailable.",
    keywords: "fallback error degraded offline cpu fallback",
    blocks: [
      {
        kind: "table",
        head: ["When this fails", "The app does this"],
        rows: [
          ["Ollama is not running", "Scores speech by simple rules and pictures by motion, sound, and quality"],
          ["The video model fails on a scene", "Uses quick scoring for that scene; trying again later redoes only those scenes"],
          ["Whisper on the graphics card", "Transcribes on the processor instead"],
          ["Fast picture matching on the graphics card", "Runs on the processor instead"],
          ["Picture matching or meaning matching is not installed", "Ranks by speech, motion, and sound; compares words instead of meaning"],
          ["A GPU on another computer goes offline", "Uses this computer's Ollama if a fallback address is set"],
        ],
      },
      { kind: "note", text: "The detailed log on the progress page always says when a fallback was used." },
    ],
  },

  "remote-gpu": {
    title: "Use a graphics card on another computer",
    summary: "Run the AI models on a stronger computer or a cloud GPU.",
    keywords: "remote gpu cloud lightning ssh tunnel ollama base url fallback api key",
    blocks: [
      { kind: "p", text: "Video and speech understanding can run on Ollama on another computer. Your video stays here: only one small picture per scene and the transcript text are sent." },
      { kind: "steps", items: ["On the other computer, install Ollama and pull the models (for example the 8B ones).", "Connect to it safely: through a VPN such as Tailscale, or an SSH tunnel (example below).", "In this app's `.env` file, set the address and a fallback, then restart the app.", "In Settings, choose the larger models yourself (automatic optimization picks for this computer) and run **Test AI**."] },
      { kind: "code", label: "SSH tunnel (keep this window open)", text: "ssh -N -L 11435:localhost:11434 user@gpu-computer" },
      { kind: "code", label: ".env", text: "OLLAMA_BASE_URL=http://127.0.0.1:11435\nOLLAMA_FALLBACK_URL=http://127.0.0.1:11434" },
      { kind: "warning", text: "Ollama has no password. Never open port 11434 to the internet. If you use a proxy that checks a token, put it in `OLLAMA_API_KEY`." },
      { kind: "note", text: "Speech to text, fast picture matching, and creating the video still run on this computer." },
    ],
  },

  "trouble-ai": {
    title: "AI offline or a model is missing",
    summary: "Fix \"AI offline\", missing models, and the thinking-model warning.",
    keywords: "ai offline not reachable model not found not installed thinking warning connection refused",
    blocks: [
      {
        kind: "table",
        head: ["You see", "Do this"],
        rows: [
          ["AI offline", "Start Ollama. Check **Settings → AI connection → Ollama address** is `http://127.0.0.1:11434` for Ollama on this computer."],
          ["AI partly ready / Not installed", "Click **Install** on the model card, or **Install recommended models**."],
          ["\"... is a thinking model\"", "Install and choose the matching `-instruct` model."],
          ["\"needs a newer Ollama\"", "Update Ollama from ollama.com/download and restart it."],
          ["Settings changes do not apply", "Values saved in Settings take priority over the `.env` file. Change them in Settings."],
        ],
      },
    ],
  },

  "trouble-memory": {
    title: "Out of memory or very slow",
    summary: "Fix memory errors, freezes, and slow analysis.",
    keywords: "out of memory ram vram freeze slow hang insufficient memory cuda out of memory",
    blocks: [
      {
        kind: "table",
        head: ["Problem", "Fix"],
        rows: [
          ["\"does not fit in memory\" or the computer freezes", "Choose the next smaller model, close other programs, or turn on automatic optimization."],
          ["Partial GPU in AI models", "Part of the model did not fit in graphics memory. Choose a smaller video model for full speed."],
          ["\"filled the model's context window\" in the log", "Lower Scenes per AI request, or raise the model memory if you have 16 GB or more."],
          ["Analysis is very slow", "Follow Fastest settings without a graphics card, and check which step is slow."],
          ["Creating the video is slow", "Turn off Smooth motion and keep the original frame rate."],
        ],
      },
      { kind: "see", article: "cpu-speed", label: "Fastest settings without a graphics card" },
    ],
  },

  "trouble-setup": {
    title: "Setup errors",
    summary: "Fix FFmpeg, Whisper, and picture-matching setup problems.",
    keywords: "ffmpeg missing whisper faster-whisper metadata_errors onnxruntime tokenizers error install",
    blocks: [
      {
        kind: "table",
        head: ["Error", "Fix"],
        rows: [
          ["FFmpeg missing", "Install FFmpeg, make sure `ffmpeg` and `ffprobe` run in a terminal, then restart the app."],
          ["\"Whisper is not installed\"", "Run `pip install -r requirements.txt` in the app's environment."],
          ["\"unexpected keyword argument 'metadata_errors'\"", "Your PyAV is too new. Run `pip install -r requirements.txt` again; it installs a compatible version."],
          ["Fast picture matching: \"Needs onnxruntime and tokenizers\"", "Run `pip install -r requirements.txt`."],
          ["Picture matching download fails", "Check your internet connection and free disk space (about 1 GB), then click Download again."],
          ["\"GPU detected, running on CPU\"", "Update the graphics driver; on AMD start Ollama with `OLLAMA_VULKAN=1`; or choose a smaller model."],
        ],
      },
    ],
  },
} satisfies Record<string, DocArticle>;

export type DocArticleId = keyof typeof DOC_ARTICLES;

export const DOC_CATEGORIES: Array<{ title: string; articles: DocArticleId[] }> = [
  { title: "Getting started", articles: ["how-it-works", "requirements", "setup-ollama", "leave-out"] },
  { title: "Speed without a graphics card", articles: ["cpu-speed", "where-time-goes", "choose-models", "threads-memory", "ollama-tuning"] },
  { title: "Lightweight AI models", articles: ["setup-whisper", "setup-image-matching", "setup-embeddings"] },
  { title: "Graphics cards", articles: ["hardware-detection", "gpu-setup", "fallbacks", "remote-gpu"] },
  { title: "Troubleshooting", articles: ["trouble-ai", "trouble-memory", "trouble-setup"] },
];

export const FIRST_ARTICLE: DocArticleId = "how-it-works";

/** Plain text of an article, for search. */
export function articleText(article: DocArticle): string {
  const parts: string[] = [article.title, article.summary, article.keywords || ""];
  for (const block of article.blocks) {
    if ("text" in block) parts.push(block.text);
    if ("items" in block) parts.push(...block.items);
    if (block.kind === "table") parts.push(...block.head, ...block.rows.flat());
    if (block.kind === "settings" || block.kind === "see") parts.push(block.label);
  }
  return parts.join(" ").replace(/[*`]/g, "").toLowerCase();
}

/** Articles matching every word of the query, best title matches first. */
export function searchDocs(query: string): DocArticleId[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  const ids = Object.keys(DOC_ARTICLES) as DocArticleId[];
  return ids
    .filter((id) => {
      const text = articleText(DOC_ARTICLES[id]);
      return words.every((word) => text.includes(word));
    })
    .sort((a, b) => {
      const inTitle = (id: DocArticleId) => words.filter((w) => DOC_ARTICLES[id].title.toLowerCase().includes(w)).length;
      return inTitle(b) - inTitle(a);
    });
}
