## AI Reel Maker for Windows

Turn long videos into short highlights and narrated reels, on your own computer.

### What's new in 1.1.0

- **Better stories from "Describe it"**: the AI editor first plans a story (hook, context, key moments, ending) and then picks moments for each part. Nothing is added after the ending, and clips that nearly touch become one.
- **Cleaner cuts**: shortened clips start and end at pauses, sentence ends or scene changes, never mid-word.
- **Cut to music**: add your own song on the Create page; every cut lands on a beat, with quicker cuts where the music is energetic.
- **Processing details**: for every video, see where the time went, what slowed it down and what would help, the AI models' work, warnings and the full log.
- **Faster analysis**: large and AV1/HEVC videos are read from a small copy, the video model looks at fewer scenes, and videos are encoded faster.
- **Languages**: choose the spoken language on the Create page; a garbled transcript is detected and no longer steers the choice of moments.
- **Optional**: let Claude choose the moments (Settings), and have the AI editor check its cut once more.
- Lengths like "the original 17-minute video" are no longer read as the wanted length, "as short as possible" works, and captions are only added when you ask for them.

### Install

1. Download **AIReelMaker-Setup-{version}.exe** below and run it. No administrator rights are needed.
2. Windows SmartScreen may say "Windows protected your PC", because the installer is not code-signed. Choose **More info → Run anyway**.
3. Start **AI Reel Maker** from the Start menu. It opens in its own window, and closing the window stops it.

Needs Windows 10 or 11 (64-bit). Python, FFmpeg 9.0 and Node 24 are included.

### Install separately

- **[Ollama](https://ollama.com)** for AI analysis. Without it, the app still works using speech, motion and sound.
- **AI models**: download them from Settings (speech, picture matching, sound matching, narration voice).
- **Narrated videos**: press **Settings → Narrated videos → Prepare video tools** once (about 200 MB from npm).
- **Claude-written briefs** (optional): the [Claude Code CLI](https://claude.com/claude-code), signed in with your Claude subscription.

### Where things are

- Program: `%LOCALAPPDATA%\Programs\AI Reel Maker`
- Your projects, videos and settings: `%LOCALAPPDATA%\AI Reel Maker`. They're kept when you update; the uninstaller asks before deleting them.
- Log: `%LOCALAPPDATA%\AI Reel Maker\logs\app.log`

To update, run a newer Setup.exe over the installed one.

Third-party licences, including FFmpeg (GPL v3) and Remotion's licence terms, are in `THIRD-PARTY-NOTICES.txt` in the program folder.
