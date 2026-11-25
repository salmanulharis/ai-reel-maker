## AI Reel Maker for Windows

Turn long videos into short highlights and narrated reels, on your own computer.

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
