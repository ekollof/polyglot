# Polyglot

Real-time multilingual transcription and translation TUI.

Listens to any audio source (applications, monitors, hardware inputs), transcribes speech with [Whisper](https://github.com/openai/whisper), detects the spoken language automatically, and translates to your chosen target language — all live in a terminal UI.

```
┌────────────────────────┬────────────────────────┐
│  Source (transcribed)  │  Translation           │
│                        │                        │
├────────────────────────┴────────────────────────┤
│  Audio: [Select]  Target: [Select]  Model: [..]  Translate: [..] │
│  Status: ● turbo ready                          │
└─────────────────────────────────────────────────┘
```

## Features

- **Auto language detection** — no need to tell it what language is being spoken; switches mid-stream
- **Real-time partials** — dim italic in-progress text appears while speech is ongoing
- **Per-utterance translation** — each committed utterance is translated independently in a thread pool (no serial queue stalls)
- **Multiple translation engines** — Google, MyMemory, LibreTranslate, Ollama (local LLM), Argos (fully offline)
- **Robust language ID** — detection runs on a rolling 5–30 s buffer of confirmed speech rather than raw 3 s VAD snapshots
- **Script-mismatch suppression** — Latin-script partials are suppressed when the active language is Arabic/Persian-script and vice versa
- **Speaker-change detection** — visual separator inserted when encoder embeddings differ between utterances
- **Grouped audio source picker** — Applications / Monitors / Hardware Inputs

## Requirements

- Python ≥ 3.14
- [uv](https://github.com/astral-sh/uv) (package/venv manager)
- PyTorch (CPU or ROCm/CUDA) — installed system-wide via your package manager
- PulseAudio or PipeWire-Pulse
- `ffmpeg` (used internally by Whisper)
- Internet access for Google/MyMemory translation (Argos/Ollama work offline)

### GPU note

Tested on AMD ROCm (~16 GB VRAM) with the `turbo` Whisper model. The `medium` model is very slow on ROCm; `turbo` or `small` are recommended.

## Installation

```bash
git clone git@github.com:ekollof/polyglot.git
cd polyglot
uv sync
```

`torch`, `numpy`, `scipy` and other system packages are excluded from the uv environment (see `pyproject.toml`) and must be installed via your distro's package manager (e.g. `python-pytorch-rocm` on Arch Linux).

## Running

```bash
uv run python app.py
# or with debug logging:
uv run python app.py 2>>/tmp/polyglot.log
```

## Key bindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Clear both panes |
| `l` | Focus audio source selector |
| `t` | Focus target language selector |
| `m` | Focus Whisper model selector |
| `e` | Focus translation engine selector |

## Translation engines

| Engine | Requires | Notes |
|--------|----------|-------|
| Google | Internet | Fast, high quality |
| MyMemory | Internet | Free tier, rate-limited |
| LibreTranslate | Self-hosted or Internet | Set URL in code |
| Ollama | Local Ollama server | Needs a model pulled, e.g. `ollama pull mistral` |
| Argos | None (offline) | Downloads language packages on first use |

## Architecture

```
AudioCapture (sounddevice)
    │ float32 mono 16kHz chunks
    ▼
Transcriber
  ├── VAD thread (Silero-VAD)
  │     Detects speech/silence boundaries; posts work items:
  │     ("detect", audio) | ("partial", audio, lang) | ("commit", audio, lang)
  │
  ├── Inference thread (Whisper)
  │     Processes queue items; manages rolling speech buffer for language ID;
  │     fires on_partial / on_result / on_non_speech callbacks
  │
  └── LangID thread (SpeechBrain VoxLingua107)
        Independent audio-domain language classifier; fused with Whisper's
        detect_language for higher confidence

TranslationWorker (ThreadPoolExecutor, 4 workers)
    Translates committed utterances concurrently; delivers results to
    the Textual main thread via call_from_thread

PolyglotApp (Textual TUI)
    Renders source + translation panes; slot-based in-place translation
    updates so late-arriving results always land in the correct row
```

## Language detection design

Detection runs on a **rolling speech buffer** of 5–30 seconds of confirmed-speech audio (only utterances that passed all filters). This makes detection far more stable than per-utterance 3-second snapshots:

- Whisper's `detect_language` and SpeechBrain's classifier are fused (weighted average when they agree, discounted winner when they disagree)
- The buffer is cleared on long silence so a new speaker starts with a clean slate
- A Lingua text-based cross-check corrects the language tag when Whisper transcribes the right script but labels it wrong

## File overview

| File | Purpose |
|------|---------|
| `app.py` | Textual TUI, translation worker, UI callbacks |
| `transcriber.py` | VAD, Whisper inference, language detection |
| `translator.py` | Multi-engine translator (Google/MyMemory/LibreTranslate/Ollama/Argos) |
| `audio.py` | PulseAudio device enumeration, sounddevice capture |
| `suppress_alsa.py` | Suppress ALSA error spam on Linux |
