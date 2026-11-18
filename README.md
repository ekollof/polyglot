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
- **Sentence-based translation** — unfinished text is joined across rolling audio chunks before translation, with completed sentences translated concurrently
- **Multiple translation engines** — Google, MyMemory, LibreTranslate, Ollama (local LLM), Argos (fully offline)
- **Robust language ID** — detection runs on a rolling 5–30 s buffer of confirmed speech rather than raw 3 s VAD snapshots
- **Script-mismatch suppression** — Latin-script partials are suppressed when the active language is Arabic/Persian-script and vice versa
- **Speaker-change detection** — visual separator inserted when encoder embeddings differ between utterances
- **Grouped audio source picker** — Applications / Monitors / Hardware Inputs

## Requirements

- Python ≥ 3.14
- [uv](https://github.com/astral-sh/uv) (package/venv manager)
- PulseAudio or PipeWire-Pulse
- `ffmpeg` (used internally by Whisper)
- Internet access for Google/MyMemory translation (Argos/Ollama work offline)

### Inference backends

At model load, Polyglot automatically tries these backends in order:

1. Native PyTorch GPUs: CUDA/ROCm, Intel XPU, then Apple MPS. Every visible CUDA/XPU device is considered; a failed device does not prevent trying the next one. Additional accelerators registered with PyTorch (including vendor NPU extensions) are also considered.
2. OpenVINO NPU, GPU, then CPU for the audio encoder, using actual device IDs reported by the runtime.
3. PyTorch CPU for any component without a working accelerated backend.

Native backends must execute an encoder pass, language detection, and cached decoding successfully before selection. CUDA/XPU first try FP16, then FP32; MPS and additional vendor backends use FP32. Sparse alignment metadata and audio preprocessing stay on CPU to avoid unsupported accelerator operations. Detection uses runtime capabilities rather than CPU-vendor or OS guesses. These checks validate compatibility, not every device's performance against every other device.

With OpenVINO, the **encoder and decoder can use different devices**. For example, Intel AI Boost can encode audio on NPU, followed by token decoding on the Intel GPU. GPU decoding is checked against PyTorch's logits, then compared with CPU decoding over three short prefill/cached-step probes. The GPU is selected only if its median probe latency is at least 10% lower. This is a startup estimate under the current system load, not a guarantee of faster transcription for every utterance. Without a compatible/faster GPU decoder, decoding stays on CPU.

The status bar and `/tmp/polyglot.log` show the actual component placement and decoder probe timings. Whisper's language probabilities, sampling loop, segment confidence filters, and speaker embeddings remain available. OpenVINO caches use Whisper's module keys, preserving beam reordering and allowing either component to fall back to PyTorch CPU after a runtime failure. SpeechBrain and Silero VAD continue to use PyTorch; SpeechBrain uses CPU on the new non-CUDA backends.

The first OpenVINO load converts and compiles the supported components; this may take time, especially for larger models. Compiled models are cached in `$XDG_CACHE_HOME/polyglot/openvino` (normally `~/.cache/polyglot/openvino`). Start with `tiny` or `base` on Intel NPU/CPU; the default `turbo` model has a much larger encoder and can be slow to compile or decode.

Installation uses CPU builds of PyTorch and torchaudio by default, including on Intel systems; OpenVINO is installed alongside them on Linux/Windows x86-64. CUDA/ROCm/XPU users must configure matching PyTorch and torchaudio sources for their driver using the [uv PyTorch guide](https://docs.astral.sh/uv/guides/integration/pytorch/), then run `uv sync`. AMD users on a distro that packages ROCm PyTorch can instead reuse those packages, as described under [Using a distro ROCm PyTorch](#using-a-distro-rocm-pytorch-amd). Apple GPU detection uses [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html). Other vendor accelerators need a registered PyTorch extension capable of running Whisper. Runtime detection does not install or replace drivers or Python wheels.

NPU support follows the installed runtime: Intel NPUs use OpenVINO; vendor PyTorch NPUs are attempted and validated if registered. Apple Neural Engine/Core ML and Qualcomm QNN are not separate backends here. The inference selection layer is portable; the application's existing PulseAudio audio integration still requires Linux.

On Arch, Intel NPU support requires `intel-npu-driver` and `intel-npu-compiler`, the `intel_vpu` kernel module, and access to `/dev/accel/accel0` (normally membership in `render`). New group membership takes effect on login. Intel GPU inference requires the appropriate Intel compute runtime. An Intel CPU alone does not provide CUDA or ROCm support.

To check the devices OpenVINO can access as your normal user:

```bash
.venv/bin/python -c 'import openvino as ov; print(ov.Core().available_devices)'
```

## Installation

```bash
git clone git@github.com:ekollof/polyglot.git
cd polyglot
./install.sh
```

`install.sh` detects the hardware and builds `.venv` for the best available backend: ROCm on AMD (reusing the distro's ROCm PyTorch packages), CUDA on NVIDIA, and the CPU/OpenVINO environment otherwise. It finishes by running an FP16 matmul and a Triton kernel on the selected device, so a broken accelerator stack fails at install time instead of silently demoting the app to OpenVINO CPU at runtime. A `polyglot` launcher is written to `~/.local/bin` that runs this checkout's `.venv`.

| Option | Effect |
|--------|--------|
| `--device auto\|rocm\|cuda\|cpu` | Override detection |
| `--check` | Report detection and verify the existing `.venv`, change nothing |
| `--verify-model` | Also load the Whisper model through the app's own backend selection |
| `--install-system` | Install missing system ROCm packages with pacman (Arch) |
| `--model NAME` | Model used by `--verify-model` (default `turbo`) |
| `--bin-dir DIR` | Where to write the `polyglot` launcher (default `~/.local/bin`) |
| `--no-launcher` | Do not write the launcher |

Without the script, `uv sync` builds the CPU/OpenVINO environment:

```bash
uv sync
```

`uv sync` installs the Python dependencies, including Textual, NumPy, SciPy, Whisper, Silero VAD, and OpenVINO, into `.venv`. System Python packages and access to another user's model cache are not required. Install `ffmpeg`, PortAudio, and PulseAudio or PipeWire-Pulse using your distro's package manager.

### Using a distro ROCm PyTorch (AMD)

`./install.sh` selects this automatically on an AMD GPU. It relies on the Arch packages `python-pytorch-rocm`, `python-torchaudio-rocm`, and `python-triton`, which provide a ROCm build that matches the installed driver, and reuses them instead of downloading the CPU wheels:

```bash
uv venv --system-site-packages
uv sync --no-install-package torch --no-install-package torchaudio --no-install-package triton
```

All three exclusions matter: the venv must not shadow the system `torch`, `torchaudio`, or `triton`. The locked `triton` is built for the CPU wheel and fails ROCm kernel compilation with `PassManager::run failed`, which makes Whisper's warmup fail and silently demotes the run to OpenVINO CPU. `uv sync` installs the excluded packages again unless the flags are repeated, and `uv run` syncs before running, so start the app with `uv run --no-sync python app.py` or `.venv/bin/python app.py`. The log then reports `Whisper model loaded — PyTorch ROCm` with FP16 enabled.

## Running

```bash
polyglot
# or with debug logging:
polyglot 2>>/tmp/polyglot.log
```

Without the launcher:

```bash
uv run --no-sync python app.py
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
    Translates assembled sentences concurrently; delivers results to
    the Textual main thread via call_from_thread

PolyglotApp (Textual TUI)
    Buffers unfinished sentences while keeping source previews live;
    renders source + translation panes; slot-based in-place translation
    updates so late-arriving results always land in the correct row
```

## Language detection design

Detection keeps up to 30 seconds of **confirmed speech** and examines a recent 5–12 second window. The newest utterance is preferred; older speech is included only when needed to reach five seconds. Before confirmed speech is available, VAD requests a detection after five seconds of active speech.

- Whisper and SpeechBrain probability distributions are fused. Weak SpeechBrain results do not dilute confident Whisper decisions, and strong disagreement remains uncertain instead of arbitrarily picking a winner.
- SpeechBrain receives timestamped snapshots and complete utterances of at least two seconds. Only results captured within six seconds before the matching audio endpoint, and after the last silence reset, can influence that decision.
- Uncertain language changes require agreement across distinct audio windows; repeatedly decoding the same buffer does not count as new evidence. Strong corroborated changes can be accepted immediately.
- Inference work is coalesced without reordering commits, resets, or stop events. Detection is retained before commits and refreshed after successfully committed speech, so it cannot starve behind partial decoding.
- Long silence clears the buffer and language context even when the audio stream keeps delivering silent chunks.
- Lingua needs enough text, a confident result and a clear margin. It can relabel a transcript only when matching audio evidence supports that language; fluent generated text alone cannot rescue a low-confidence segment.

## Transcription quality

Every utterance keeps Whisper's source-language auto-detection. No source language or previous transcript is supplied as a prompt. Partials wait for two seconds of context and use deterministic greedy decoding. Committed speech starts with the same fast decode; uncertain speech may receive one three-beam retry, retained only if its token-weighted log-probability improves. Confident speech and noise-only output do not trigger this extra pass.

Continuous speech now starts with a four-second commit window. The window adapts between four and eight seconds to amortize processing costs on slower hardware. Expensive previews are skipped when they compete with final transcripts, and quality retries run only when there is no commit backlog and the initial decode leaves sufficient time. Committed audio is kept in order rather than dropped to hide lag.

Rolling cuts prefer a recent pause observed by Silero (at least 120 ms) instead of always cutting at the clock deadline. Audio after the selected cut is retained exactly for the next window. When no suitable pause is available, the pipeline waits up to an additional 600 ms before making a bounded hard cut. Stable partial text alone cannot force an early cut without a supporting acoustic pause.

An inference-only two-entry encoder cache retains Whisper's distinct language-detection and decoding inputs. Speaker detection reuses the current decode's features. Stable language checks can be deferred for up to ten seconds when fresh SpeechBrain evidence and the current transcript agree; conflicting evidence still triggers a check immediately. The log's `Realtime:` entries report processing time, queue delay, end-of-utterance lag, adaptive window and preview scheduling. Capturing the utterance itself adds its window duration to the first word's latency.

Audio gain is capped and limited by the signal's peak instead of clipping amplified speech. Partials and commits share confidence, silence and repetition checks. High silence probability alone does not discard confidently recognized speech; rejected partial segments cannot reappear through a raw-text fallback. Confidence from an old cached language is not attached to a different utterance language.

These safeguards target the observed failure modes, rather than guaranteeing a particular word error rate. Difficult utterances can take longer because of the bounded retry; model size, recording quality, accents and code switching still affect accuracy.

## Translation sentence boundaries

Rolling audio windows are processing boundaries, rather than sentence boundaries. Confirmed text ending in `...` or `…` is held and joined to the next confirmed chunk. Ellipses at the join are removed, while internal punctuation is retained. Abbreviations, decimals, closing quotes, and common non-Latin sentence punctuation are handled before a sentence is submitted for translation.

An unfinished sentence remains visible as a dim italic source preview, with `…` in the translation pane. Live ASR partials are displayed alongside that confirmed tail but never sent to the translator. Sentence-ending punctuation submits the completed sentence immediately. A genuine silence reset, speaker/language change, model/source/target change, or engine change flushes any remaining confirmed text as an utterance; long punctuation-free text is also flushed at a 2,400-character threshold. Clear discards pending text.

Whisper can omit punctuation in Persian and other languages. Actual VAD utterance endings, and clearer pauses of at least 350 ms used for rolling cuts, therefore also release confirmed text after the matching transcript has been processed. Ordinary rolling cuts keep it pending. This provides punctuation-independent spoken boundaries without guessing Persian grammar, forcing a source language, or inventing full stops. An acoustic pause is still a cue, not proof of a grammatical sentence boundary.

This adds a deliberate wait for sentence completion in the translation pane. The source pane stays live, and rolling audio inference does not wait or re-decode whole sentences. Joining text cannot recover a word Whisper has already misrecognized at an audio boundary.

## File overview

| File | Purpose |
|------|---------|
| `app.py` | Textual TUI, translation worker, UI callbacks |
| `transcriber.py` | VAD, Whisper inference, language detection |
| `inference_backend.py` | Accelerator detection, validation, encoder placement, and fallback |
| `openvino_decoder.py` | GPU decoder graphs, Whisper-compatible KV caches, and CPU/GPU probes |
| `speech_quality.py` | Queue coalescing, timestamped language evidence, fusion, and confidence checks |
| `sentence_buffer.py` | Confirmed sentence assembly across audio windows before translation |
| `translator.py` | Multi-engine translator (Google/MyMemory/LibreTranslate/Ollama/Argos) |
| `audio.py` | PulseAudio device enumeration, sounddevice capture |
| `suppress_alsa.py` | Suppress ALSA error spam on Linux |
