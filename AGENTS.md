# AGENTS.md — Instructions for AI coding agents

This file describes the project, conventions, and working rules for AI agents (OpenCode, Copilot, etc.) working on this codebase.

## Project

**Polyglot** is a real-time multilingual transcription and translation TUI written in Python.

- Audio captured via sounddevice/PulseAudio
- Speech detected by Silero-VAD
- Transcription by openai-whisper (GPU, ROCm AMD)
- Language detection: Whisper `detect_language` + SpeechBrain VoxLingua107, fused
- Text language cross-check: Lingua
- Translation: deep-translator (Google/MyMemory), LibreTranslate, Ollama, argostranslate
- UI: Textual TUI

## Environment

- OS: Linux (Arch), AMD ROCm GPU (~16 GB VRAM)
- Python: 3.14, managed with `uv`
- Venv: `.venv/` — always use `.venv/bin/python`
- Run: `cd /home/andrath/Code/polyglot && .venv/bin/python app.py 2>>/tmp/polyglot.log`
- Syntax check after every change: `.venv/bin/python -m py_compile <file>`

## Key files

| File | Role |
|------|------|
| `app.py` | Textual TUI, `TranslationWorker`, all UI callbacks |
| `transcriber.py` | VAD thread, Inference thread, LangID thread, `Transcriber` class |
| `translator.py` | `Translator` class — multi-engine, thread-safe |
| `audio.py` | PulseAudio enumeration, `AudioCapture` |
| `suppress_alsa.py` | ALSA noise suppression, tqdm pre-warm |

## Architecture rules

### Threading model
- **VAD thread** (`_run`): reads audio, runs Silero-VAD, posts to `_infer_queue`. Never touches Whisper.
- **Inference thread** (`_inference_loop`): sole consumer of `_infer_queue`. Only thread that calls Whisper. Owns `_detected_lang`, `_speech_buffer`, etc. — no locks needed.
- **LangID thread**: runs SpeechBrain independently; writes `_sb_lang`/`_sb_confidence` (GIL-atomic).
- **TranslationWorker**: `ThreadPoolExecutor` (4 workers) — translations run concurrently.
- **Textual main thread**: UI only. Cross-thread calls use `call_from_thread`.

### No locking
Do not introduce `threading.Lock`, semaphores, or any explicit synchronisation. The design is intentionally lockless:
- `_infer_queue` is the only cross-thread channel between VAD and Inference
- Simple attribute reads/writes between threads rely on CPython's GIL atomicity
- `_speech_buffer` and all `_detected_*` fields are inference-thread-only

### Language detection
- `_detected_lang` is owned exclusively by the Inference thread
- It is updated **only** by `_infer_detect()`, never by `_infer_commit()`
- `_infer_detect()` uses `_buffer_audio_for_detect()` which returns the rolling speech buffer (≥5 s of confirmed speech) instead of raw VAD snapshots
- After each successful commit (`any_emitted=True`), audio is pushed to `_speech_buffer` via `_push_speech_buffer()`
- The buffer is cleared on `reset_lang` (long silence)

### Translation pane
- Each committed utterance gets a pre-allocated slot in `_trans_slots` (row_id → line index)
- `_update_translation_row` does a keyed slot lookup — order of arrival doesn't matter
- Translations arriving before their slot is registered are stashed in `_pending_trans`
- **Never** submit partial translations to `TranslationWorker` (causes slot/strip races)

## Conventions

- Always read `/tmp/polyglot.log` before diagnosing issues
- Syntax-check with `.venv/bin/python -m py_compile` after every edit
- LSP/pyright type annotation warnings are cosmetic — do not fix them unless asked
- Do not pin source language — keep auto-detection
- Partials display as dim italic in the source pane; the translation pane shows `…` until the utterance is committed
- GPU is ROCm (AMD) — avoid CUDA-specific assumptions

## Whisper models (ROCm performance)

| Model | VRAM | Speed |
|-------|------|-------|
| tiny | ~1 GB | ~32× real-time |
| base | ~1 GB | ~16× |
| small | ~2 GB | ~6× |
| turbo | ~6 GB | ~3× (recommended) |
| medium | ~5 GB | ~0.2× (too slow for live use on ROCm) |
| large | ~10 GB | baseline |

## Commit style

Short imperative subject line. Body explains *why*, not *what*. Reference the specific mechanism fixed (e.g. "slot index corruption", "NameError in _infer_commit").

## Do not

- Push to `main` with force
- Add `threading.Lock` or semaphores
- Submit partial translations to `TranslationWorker`
- Update `_detected_lang` from `_infer_commit`
- Pass `initial_prompt` to Whisper commits (causes hallucination cascades on multilingual audio)
- Pin source language (breaks auto-detection)
