"""
app.py — Polyglot: Real-time multilingual transcription and translation TUI.

Layout:
  ┌────────────────────────┬────────────────────────┐
  │  Source (transcribed)  │  Translation           │
  │  [RichLog]             │  [RichLog]             │
  ├────────────────────────┴────────────────────────┤
  │  Audio: [Select]   Target lang: [Select]  [Clr] │
  │  Status: ● …                                    │
  └─────────────────────────────────────────────────┘

Run with:
    uv run python app.py
"""

from __future__ import annotations

# Pre-warm tqdm's multiprocessing lock FIRST, before any fd redirects.
# This triggers the resource_tracker subprocess while all fds are clean.
from suppress_alsa import prewarm_tqdm_lock

prewarm_tqdm_lock()

import asyncio
import concurrent.futures
import os
import logging
import queue
import threading
import time
from typing import Optional

from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    RichLog,
    Select,
    Static,
)
from textual.widgets._select import SelectOverlay  # type: ignore[attr-defined]
from textual.widgets.option_list import Option  # type: ignore[attr-defined]

from audio import AudioCapture, default_device, list_input_devices
from sentence_buffer import SentenceBuffer
from transcriber import Transcriber
from translator import (
    ENGINE_ARGOS,
    ENGINE_GOOGLE,
    ENGINE_LIBRETRANSLATE,
    ENGINE_MYMEMORY,
    ENGINE_NLLB,
    ENGINE_OLLAMA,
    ENGINE_OPENAI,
    Translator,
)

logging.basicConfig(
    filename="/tmp/polyglot.log",
    level=logging.DEBUG,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RTL language detection
# ---------------------------------------------------------------------------

# ISO 639-1 codes for right-to-left scripts.
_RTL_LANGS: frozenset[str] = frozenset(
    {"ar", "fa", "he", "ur", "yi", "dv", "ps", "ug", "ku", "sd"}
)


def _lang_tag(from_lang: str, confidence: Optional[float]) -> str:
    """Format a compact language+confidence tag, e.g. '[fa 94%]' or '[fa ?%]'."""
    if confidence is not None:
        pct = f"{int(round(confidence * 100))}%"
    else:
        pct = "?%"
    return f"[{from_lang} {pct}] "


def _make_lang_tag_line(
    translation: str,
    from_lang: str,
    confidence: Optional[float],
    is_final: bool,
) -> Text:
    """Build a Rich Text line with a dim tag prefix and the translation body.

    The tag is always dim so it doesn't compete visually with the translation.
    The translation body is normal weight for final results, dim italic for
    in-progress partials.
    """
    tag = _lang_tag(from_lang, confidence)
    body_style = "" if is_final else "dim italic"
    line = Text("", justify="left")
    line.append(tag, style="dim")
    line.append(translation, style=body_style)
    return line


def _make_text(text: str, style: str = "", lang: str = "") -> Text:
    """Create a Rich Text object with correct justification for the language.

    RTL languages (Arabic, Persian, Hebrew, etc.) are right-justified so that
    wrapped lines align correctly and the reading direction is preserved.
    """
    justify = "right" if lang in _RTL_LANGS else "left"
    return Text(text, style=style, justify=justify)


# ---------------------------------------------------------------------------
# Language display names (ISO-639-1 → human readable)
# ---------------------------------------------------------------------------

LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "uk": "Ukrainian",
    "el": "Greek",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "ca": "Catalan",
    "la": "Latin",
    "af": "Afrikaans",
    "sq": "Albanian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "gl": "Galician",
    "ka": "Georgian",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "is": "Icelandic",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "mn": "Mongolian",
    "ne": "Nepali",
    "pa": "Punjabi",
    "si": "Sinhala",
    "sl": "Slovenian",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tl": "Filipino",
    "ur": "Urdu",
    "uz": "Uzbek",
}

TARGET_LANGUAGES: list[tuple[str, str]] = sorted(
    LANGUAGE_NAMES.items(), key=lambda x: x[1]
)  # sorted by display name

# Translation engines available in the UI.  Composite values carry the model
# for engines that can host several; selecting a missing model offers to
# download it before switching.
OLLAMA_MODELS = [
    ("aya-expanse:8b", "5.1 GB"),
    ("qwen2.5:7b", "4.7 GB"),
    ("gemma2:9b", "5.4 GB"),
]
# POLYGLOT_OLLAMA_MODEL lets a user keep a different local model; show it too.
_OLLAMA_DEFAULT = os.environ.get("POLYGLOT_OLLAMA_MODEL", OLLAMA_MODELS[0][0])
_OLLAMA_OPTIONS = list(OLLAMA_MODELS)
if _OLLAMA_DEFAULT not in {model for model, _ in OLLAMA_MODELS}:
    _OLLAMA_OPTIONS.insert(0, (_OLLAMA_DEFAULT, ""))
NLLB_MODELS = [
    ("facebook/nllb-200-distilled-600M", "1.2 GB"),
    ("facebook/nllb-200-distilled-1.3B", "2.7 GB"),
]

TRANSLATION_ENGINES: list[tuple[str, str]] = [
    ("Google (internet)", ENGINE_GOOGLE),
    ("MyMemory (internet)", ENGINE_MYMEMORY),
    ("LibreTranslate", ENGINE_LIBRETRANSLATE),
    *[
        (
            f"Ollama · {model}" + (f" ({size})" if size else ""),
            f"{ENGINE_OLLAMA}:{model}",
        )
        for model, size in _OLLAMA_OPTIONS
    ],
    ("OpenAI-compatible (configured)", ENGINE_OPENAI),
    *[
        (f"NLLB-200 · {model.rsplit('/', 1)[-1]} ({size})", f"{ENGINE_NLLB}:{model}")
        for model, size in NLLB_MODELS
    ],
    ("Argos (offline)", ENGINE_ARGOS),
]

DEFAULT_ENGINE_OPTION = f"{ENGINE_OLLAMA}:{_OLLAMA_DEFAULT}"

# Whisper models in order from fastest/smallest to slowest/largest.
# Model      Params   VRAM    Relative speed   Notes
# ---------  -------  ------  ---------------  -----------------------------------
# tiny       ~39 M    ~1 GB   ~32×             English-only variant available.
#                                              Noticeable accuracy loss; good for
#                                              prototyping or very fast hardware.
# base       ~74 M    ~1 GB   ~16×             Better than tiny with still very
#                                              low resource use.
# small      ~244 M   ~2 GB   ~6×              Good balance for single-language
#                                              streams; weaker on code-switching.
# turbo      ~809 M   ~6 GB   ~3× (ROCm)      Fine-tuned large-v3 with attention
#                                              heads pruned for speed. Runs at ~3×
#                                              real-time on ROCm (AMD GPU) with the
#                                              qkv_attention patch applied.
# medium     ~769 M   ~5 GB   ~0.2× (ROCm)   Strong multilingual accuracy but
#                                              catastrophically slow on ROCm (~0.24×
#                                              real-time) — do not use for live work.
# large      ~1.55 B  ~10 GB  1× (baseline)   Highest accuracy, slowest.
#                                              large-v3 under the hood. Use when
#                                              medium still misses words.
#
# "medium" is the recommended default — strong multilingual accuracy and
# compatible with ROCm (AMD GPU) builds.
WHISPER_MODELS: list[tuple[str, str]] = [
    ("tiny", "Tiny   (~39M)"),
    ("base", "Base   (~74M)"),
    ("small", "Small  (~244M)"),
    ("medium", "Medium (~769M)"),
    ("turbo", "Turbo  (~809M)"),
    ("large", "Large  (~1.5B)"),
]

# ---------------------------------------------------------------------------
# Grouped audio device selector
# ---------------------------------------------------------------------------

# Negative sentinel values used as the "value" of group-header pseudo-options.
# They must not collide with any real sounddevice index (always >= 0).
_GRP_APP: int = -1
_GRP_MONITOR: int = -2
_GRP_INPUT: int = -3

_GROUP_ORDER: list[str] = ["app", "monitor", "sound"]
_GROUP_LABELS: dict[str, str] = {
    "app": "── Applications ──",
    "monitor": "── Monitors ──",
    "sound": "── Hardware Inputs ──",
}


class GroupedSelect(Select[int]):
    """A Select[int] subclass that renders sentinel-valued options as disabled
    group-header rows.  All options with a negative value are treated as headers
    (shown dimmed and not selectable); non-negative values behave normally.
    """

    def _setup_options_renderables(self) -> None:  # type: ignore[override]
        """Override to turn negative-value options into disabled headers."""
        from rich.text import Text as RichText

        options: list[Option] = []
        for prompt, value in self._options:  # type: ignore[attr-defined]
            if value == self.NULL:
                options.append(Option(RichText(self.prompt, style="dim")))
            elif isinstance(value, int) and value < 0:
                options.append(Option(prompt, disabled=True))
            else:
                options.append(Option(prompt))

        option_list = self.query_one(SelectOverlay)
        option_list.clear_options()
        option_list.add_options(options)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
Screen {
    background: $surface;
}

#panes {
    height: 1fr;
}

.pane-container {
    width: 1fr;
    border: solid $primary-darken-2;
    padding: 0 1;
}

.pane-title {
    text-align: center;
    background: $primary-darken-2;
    color: $text;
    padding: 0 1;
    height: 1;
}

RichLog {
    height: 1fr;
    scrollbar-size: 1 1;
}

#controls {
    height: auto;
    background: $surface-darken-1;
    border-top: solid $primary-darken-2;
    padding: 0 1;
    align: left middle;
}

#controls-row {
    height: 3;
    align: left middle;
}

.control-label {
    width: auto;
    padding: 0 1 0 0;
    color: $text-muted;
}

#source-select {
    width: 38;
}

#lang-select {
    width: 22;
}

#model-select {
    width: 20;
}

#engine-select {
    width: 20;
}

#clear-btn {
    width: auto;
    margin-left: 1;
}

#status-bar {
    height: 1;
    background: $surface-darken-2;
    padding: 0 1;
    color: $text-muted;
}

#status-bar.status-loading {
    color: $warning;
}

#status-bar.status-listening {
    color: $success;
}

#status-bar.status-error {
    color: $error;
}

ConfirmDownload {
    align: center middle;
}

#confirm-dialog {
    width: 64;
    height: auto;
    border: thick $primary;
    background: $surface;
    padding: 1 2;
}

#confirm-title {
    text-style: bold;
    margin-bottom: 1;
}

#confirm-detail {
    margin-bottom: 1;
}

#confirm-buttons {
    height: auto;
    align: right middle;
}

#confirm-buttons Button {
    margin-left: 1;
}
"""


# ---------------------------------------------------------------------------
# Translation worker
# ---------------------------------------------------------------------------


class TranslationWorker:
    """
    Runs translations concurrently using a thread pool.

    Each submitted job is dispatched immediately to a ThreadPoolExecutor so
    that slow or stalled HTTP requests (Google Translate, etc.) never block
    other pending translations.  Results are delivered to the main thread via
    call_from_thread in whatever order they complete.

    Only finals are ever submitted (partials are not translated).
    """

    # Maximum concurrent translation requests.  4 is enough to absorb burst
    # commits (which can emit 3–5 segments at once) without hammering the
    # translation service.
    _MAX_WORKERS = 4

    def __init__(self, translator: Translator, on_translated):
        self._translator = translator
        self._on_translated = on_translated
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self._MAX_WORKERS,
            thread_name_prefix="TranslationWorker",
        )

    def submit(
        self,
        row_id: int,
        text: str,
        from_lang: str,
        to_lang: str,
        is_final: bool,
        confidence: Optional[float] = None,
    ) -> None:
        """Dispatch a translation job to the thread pool."""
        self._executor.submit(
            self._translate_item,
            (row_id, text, from_lang, to_lang, is_final, confidence),
        )

    def stop(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _translate_item(self, item: tuple) -> None:
        row_id, text, from_lang, to_lang, is_final, confidence = item
        if from_lang == to_lang:
            translated = text
        else:
            try:
                translated = self._translator.translate(
                    text, from_lang=from_lang, to_lang=to_lang
                )
            except Exception as exc:
                logger.error("Translation error: %s", exc)
                translated = text
        self._on_translated(row_id, translated, is_final, from_lang, confidence)


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


class ConfirmDownload(ModalScreen[bool]):
    """Confirm an on-demand model download before it starts."""

    BINDINGS = [Binding("escape", "dismiss_false", "Cancel")]

    def __init__(self, model: str, size: str) -> None:
        super().__init__()
        self._model = model
        self._size_label = size

    def compose(self) -> ComposeResult:
        detail = self._model
        if self._size_label:
            detail += f"  ({self._size_label})"
        with Vertical(id="confirm-dialog"):
            yield Static("Download translation model?", id="confirm-title")
            yield Static(
                f"{detail}\n\nIt will be stored locally and used for translation.",
                id="confirm-detail",
            )
            with Horizontal(id="confirm-buttons"):
                yield Button("Download", variant="primary", id="confirm-yes")
                yield Button("Cancel", id="confirm-no")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "confirm-yes")

    def action_dismiss_false(self) -> None:
        self.dismiss(False)


class PolyglotApp(App):
    """Real-time multilingual transcription and translation."""

    TITLE = "Polyglot"
    SUB_TITLE = "Real-time Transcription & Translation"
    CSS = CSS

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("c", "clear", "Clear"),
        Binding("l", "focus_source", "Audio source"),
        Binding("t", "focus_target", "Target lang"),
        Binding("m", "focus_model", "Model"),
        Binding("e", "focus_engine", "Engine"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._devices = list_input_devices()
        self._audio_queue: queue.Queue = queue.Queue(maxsize=300)
        self._capture: Optional[AudioCapture] = None
        self._transcriber: Optional[Transcriber] = None
        self._translator = Translator(
            engine=ENGINE_OLLAMA,
            on_status=self._on_translator_status,
        )
        self._translation_worker: Optional[TranslationWorker] = None
        self._current_device_index: int = default_device(self._devices)
        self._target_lang: str = "en"
        self._detected_lang: str = "?"
        self._model_name: str = "turbo"
        self._engine: str = ENGINE_OLLAMA
        self._engine_option: str = DEFAULT_ENGINE_OPTION
        # Long-running background work (model loads, downloads, device waits)
        # owns the status bar until it finishes; everything else falls back to
        # the regular listening line.
        self._activity: Optional[str] = None
        self._activity_class = "status-loading"
        self._activity_until = 0.0
        self._model_ready = False

        # Row tracking for in-place partial updates.
        self._next_row_id: int = 0
        self._pending_row_id: Optional[int] = None
        # Strip-start indices are tracked *per pane* because the two RichLogs
        # accumulate lines independently and may have different heights.
        self._source_strip_start: int = 0
        self._trans_strip_start: int = 0
        self._row_count: int = 0

        # Per-utterance translation slots: row_id → trans_log strip_start at
        # the time the "…" placeholder was written.  Translations may arrive
        # after the next utterance has already started (partials for N+1 are
        # shown while N's translation is still in-flight).  Storing the slot
        # per row_id lets us update each translation independently regardless
        # of which utterance is currently showing a partial.
        # Only the *tail* slot can be safely rewritten in-place (all later
        # writes go after it), so we also track the row_id order.
        self._trans_slots: dict[int, int] = {}  # row_id → strip_start
        self._trans_slot_order: list[int] = []  # insertion order of row_ids

        # Final translations that arrived before their slot was registered.
        # Race: translation worker can deliver a result before _finalise_source
        # runs on the main thread (both use call_from_thread from different
        # threads, ordering not guaranteed).  We hold them here and apply
        # immediately when _finalise_source registers the slot.
        self._pending_trans: dict[
            int, tuple
        ] = {}  # row_id → (translation, from_lang, confidence)

        # Last finalised source text, used as context for the next translation
        # so the translator can resolve sentence continuations and pronouns.
        self._sentence_buffer = SentenceBuffer()
        self._last_source_text: str = ""  # retained for potential future use

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Header()

        with Horizontal(id="panes"):
            with Vertical(classes="pane-container"):
                yield Static("Source (original)", classes="pane-title")
                yield RichLog(id="source-log", highlight=False, markup=False, wrap=True)

            with Vertical(classes="pane-container"):
                yield Static("Translation", classes="pane-title")
                yield RichLog(
                    id="translation-log", highlight=False, markup=False, wrap=True
                )

        with Vertical(id="controls"):
            with Horizontal(id="controls-row"):
                yield Label("Audio:", classes="control-label")
                yield GroupedSelect(
                    options=self._device_options(),
                    value=self._current_device_index,
                    id="source-select",
                )
                yield Label("  Target:", classes="control-label")
                yield Select(
                    options=[(name, code) for code, name in TARGET_LANGUAGES],
                    value="en",
                    id="lang-select",
                )
                yield Label("  Model:", classes="control-label")
                yield Select(
                    options=[(label, key) for key, label in WHISPER_MODELS],
                    value="turbo",
                    id="model-select",
                )
                yield Label("  Translate:", classes="control-label")
                yield Select(
                    options=[(label, key) for label, key in TRANSLATION_ENGINES],
                    value=DEFAULT_ENGINE_OPTION,
                    id="engine-select",
                )
                yield Button("Clear", id="clear-btn", variant="default")

        yield Label(
            "● Loading Whisper model…", id="status-bar", classes="status-loading"
        )
        yield Footer()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        self._start_pipeline()
        self._check_default_model()

    @work(exclusive=True)
    async def _check_default_model(self) -> None:
        """Offer to fetch the default translation model on first run."""
        engine, _, model = self._engine_option.partition(":")
        if engine == ENGINE_OLLAMA:
            names = await asyncio.to_thread(self._translator.ollama_models)
            if not isinstance(names, (set, frozenset)):
                if names is None:
                    self._set_status(
                        "Ollama unreachable — using offline fallback", "status-error"
                    )
                return
            installed = model in names or f"{model}:latest" in names
        elif engine == ENGINE_NLLB:
            installed = self._translator.is_model_available(engine, model)
        else:
            return
        if installed:
            return
        if await self._confirm_and_pull(engine, model, self._engine_option):
            self._set_status(f"{model} ready")

    def on_unmount(self) -> None:
        if self._transcriber:
            self._transcriber.stop()
        if self._capture:
            self._capture.stop()
        if self._translation_worker:
            self._translation_worker.stop()

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def _start_pipeline(self) -> None:
        """Start audio capture, transcription, and translation pipeline."""
        # Translation worker — runs translations off the transcriber thread.
        self._translation_worker = TranslationWorker(
            translator=self._translator,
            on_translated=self._on_translation_ready,
        )

        # Audio capture
        current = self._device_entry(self._current_device_index)
        self._capture = AudioCapture(
            device_index=self._current_device_index,
            audio_queue=self._audio_queue,
            device_name=current["name"] if current else "",
            on_status=self._on_capture_status,
        )
        self._capture.start()

        # Transcriber (loads model in background)
        self._transcriber = Transcriber(
            audio_queue=self._audio_queue,
            on_result=self._on_transcription_result,
            on_partial=self._on_partial_result,
            on_speaker_change=self._on_speaker_change,
            on_non_speech=self._on_non_speech,
            model_name=self._model_name,
        )
        self._transcriber.start()
        self._poll_model_ready()

    @work(thread=True)
    def _poll_model_ready(self) -> None:
        """Wait for Whisper model to load, then update status."""
        assert self._transcriber is not None
        self._transcriber._ready.wait()
        self.call_from_thread(self._on_model_ready)

    def _on_model_ready(self) -> None:
        self._model_ready = True
        self._update_status()

    def _restart_transcriber(self) -> None:
        """Stop the current transcriber and start a new one with the selected model."""
        self._flush_sentence_buffer()
        self._model_ready = False
        self._update_status()
        if self._transcriber:
            self._transcriber.stop()
        self._transcriber = Transcriber(
            audio_queue=self._audio_queue,
            on_result=self._on_transcription_result,
            on_partial=self._on_partial_result,
            on_speaker_change=self._on_speaker_change,
            on_non_speech=self._on_non_speech,
            model_name=self._model_name,
        )
        self._transcriber.start()
        self._poll_model_ready()

    # ------------------------------------------------------------------
    # Transcription callbacks (called from transcriber thread)
    # ------------------------------------------------------------------

    def _on_partial_result(
        self, detected_lang: str, partial_text: str, confidence: Optional[float]
    ) -> None:
        """Preview live audio alongside confirmed unfinished sentence text."""
        self.call_from_thread(self._show_live_partial, partial_text, detected_lang)

    def _show_live_partial(self, partial_text: str, detected_lang: str) -> None:
        text, language = self._sentence_buffer.preview(partial_text, detected_lang)
        self._show_partial(self._next_row_id, text, language)

    def _on_transcription_result(
        self, detected_lang: str, source_text: str, confidence: Optional[float]
    ) -> None:
        """Move confirmed chunks to the UI-owned sentence buffer."""
        self.call_from_thread(
            self._accept_transcription, detected_lang, source_text, confidence
        )

    def _accept_transcription(self, detected_lang, source_text, confidence) -> None:
        if not source_text.strip():
            return
        for sentence in self._sentence_buffer.push(
            source_text, detected_lang, confidence
        ):
            self._submit_sentence(sentence)
        if self._sentence_buffer.pending:
            self._show_partial(
                self._next_row_id,
                self._sentence_buffer.pending,
                self._sentence_buffer.language,
            )

    def _submit_sentence(self, sentence) -> None:
        row_id = self._next_row_id
        self._next_row_id += 1
        # Register a keyed slot before any worker can deliver a translation.
        self._finalise_source(row_id, sentence.text, sentence.language)
        self._last_source_text = sentence.text
        assert self._translation_worker is not None
        self._translation_worker.submit(
            row_id=row_id,
            text=sentence.text,
            from_lang=sentence.language,
            to_lang=self._target_lang,
            is_final=True,
            confidence=sentence.confidence,
        )

    def _flush_sentence_buffer(self) -> None:
        ready = self._sentence_buffer.flush()
        for sentence in ready:
            self._submit_sentence(sentence)
        if not ready and self._pending_row_id is not None:
            self._finalise_source(self._next_row_id, "", self._detected_lang)

    # ------------------------------------------------------------------
    # Speaker-change / non-speech callbacks (called from transcriber thread)
    # ------------------------------------------------------------------

    def _on_speaker_change(self) -> None:
        """Called by Transcriber when a speaker change is detected."""
        self.call_from_thread(self._insert_speaker_separator)

    def _on_non_speech(self, reason: str) -> None:
        """Handle rejected audio and ordered acoustic utterance boundaries."""
        logger.debug("Transcription boundary/status: %s", reason)
        # VAD utterance ends are 600 ms phrase pauses, so flushing there
        # submits half sentences to the translator.  sentence_buffer.pending
        # holds unfinished text and joins it with the next commit; only a
        # 3 s silence (or a failure) means the speaker really stopped.
        if reason in ("silence", "inference_error"):
            self.call_from_thread(self._flush_sentence_buffer)

    # ------------------------------------------------------------------
    # Translation callback (called from TranslationWorker thread)
    # ------------------------------------------------------------------

    def _on_translation_ready(
        self,
        row_id: int,
        translation: str,
        is_final: bool,
        from_lang: str,
        confidence: Optional[float],
    ) -> None:
        """Called by TranslationWorker when a translation is ready."""
        self.call_from_thread(
            self._update_translation_row,
            row_id,
            translation,
            is_final,
            from_lang,
            confidence,
        )

    # ------------------------------------------------------------------
    # UI update methods (must run on main thread)
    # ------------------------------------------------------------------

    def _pop_partial(self, log: RichLog, strip_start: int) -> None:
        """Remove all Strip entries from strip_start onwards in log.lines."""
        if strip_start < len(log.lines):
            del log.lines[strip_start:]
            log._line_cache.clear()  # type: ignore[attr-defined]
            log.virtual_size = log.virtual_size._replace(height=len(log.lines))
            log.refresh()

    def _insert_speaker_separator(self) -> None:
        """Insert a dim separator line between speaker turns.

        A thin horizontal rule is added to the source pane.  A blank spacer
        is added to the translation pane so the two logs stay visually aligned.
        Both are treated as finalised rows — they have no translation slot and
        will never be rewritten.
        """
        self._flush_sentence_buffer()
        source_log = self.query_one("#source-log", RichLog)
        trans_log = self.query_one("#translation-log", RichLog)

        # If a dim partial is currently shown, clear it first so the separator
        # appears above the next utterance, not in the middle of the current one.
        if self._pending_row_id is not None:
            self._pop_partial(source_log, self._source_strip_start)
            self._pop_partial(trans_log, self._trans_strip_start)
            self._pending_row_id = None

        separator = Text("─" * 40, style="dim")
        source_log.write(separator)
        trans_log.write(Text("", style="dim"))  # blank spacer keeps panes aligned

    def _show_partial(self, row_id: int, partial_text: str, detected_lang: str) -> None:
        """Write or update the interim (dim) partial transcription line."""
        try:
            source_log = self.query_one("#source-log", RichLog)
            trans_log = self.query_one("#translation-log", RichLog)
        except Exception:
            return  # app is shutting down — widgets already removed

        if self._pending_row_id == row_id:
            # Replace the previous partial strips with updated text.
            self._pop_partial(source_log, self._source_strip_start)
            self._pop_partial(trans_log, self._trans_strip_start)
        else:
            # First partial for this utterance — record where it starts in each pane.
            self._pending_row_id = row_id
            self._source_strip_start = len(source_log.lines)
            self._trans_strip_start = len(trans_log.lines)

        source_log.write(
            _make_text(partial_text, style="dim italic", lang=detected_lang)
        )
        trans_log.write(Text("…", style="dim italic"))

        self._detected_lang = detected_lang
        self._update_status()

    def _finalise_source(
        self, row_id: int, source_text: str, detected_lang: str
    ) -> None:
        """Replace the partial line with the final transcription in normal style."""
        try:
            source_log = self.query_one("#source-log", RichLog)
            trans_log = self.query_one("#translation-log", RichLog)
        except Exception:
            return  # widgets already torn down on quit

        if self._pending_row_id == row_id:
            self._pop_partial(source_log, self._source_strip_start)
            self._pop_partial(trans_log, self._trans_strip_start)
        # else: no dim partial was shown for this row — nothing to pop.

        # Empty source_text means inference failed — partial cleared, nothing to write.
        if not source_text:
            self._pending_row_id = None
            return

        # Write source text, then record where the trans placeholder starts
        # in the trans log (must snapshot *after* any pops, independently per pane).
        source_log.write(
            _make_text(source_text, lang=detected_lang)
        )  # final — normal style

        # Register a translation slot for this row before writing the placeholder
        # so the strip_start points at the "…" line we're about to add.
        slot_start = len(trans_log.lines)
        trans_log.write(Text("…", style="dim"))  # translation placeholder
        self._trans_slots[row_id] = slot_start
        self._trans_slot_order.append(row_id)
        logger.debug(
            "_finalise_source row=%d slot=%d trans_log_len=%d",
            row_id,
            slot_start,
            len(trans_log.lines),
        )
        # _trans_strip_start tracks where the dim partial lives.  After a
        # finalise the "…" is a registered slot, not a dim partial, so clear
        # the pending marker.  _show_partial will set it again when the next
        # interim result arrives.
        self._pending_row_id = None

        # If a translation already arrived before this slot was registered,
        # schedule a deferred flush rather than applying it immediately.
        # Applying it now would re-append the partial placeholder before the
        # remaining _finalise_source calls for sibling segments have run,
        # corrupting all subsequent slot positions.
        if self._pending_trans:
            self.set_timer(0.0, self._flush_pending_trans)

        self._row_count += 1
        self._detected_lang = detected_lang
        self._update_status()

    def _flush_pending_trans(self) -> None:
        """Apply translations that arrived before their slots were registered.

        Called via set_timer(0) so it runs after all _finalise_source calls
        for the current batch of segments have completed and all slots are
        registered at their correct positions.
        """
        for row_id, (translation, from_lang, confidence) in list(
            self._pending_trans.items()
        ):
            if row_id in self._trans_slots:
                self._pending_trans.pop(row_id)
                self._update_translation_row(
                    row_id, translation, True, from_lang, confidence
                )

    def _update_translation_row(
        self,
        row_id: int,
        translation: str,
        is_final: bool,
        from_lang: str = "?",
        confidence: Optional[float] = None,
    ) -> None:
        """Replace the translation placeholder with the actual translation.

        For partial (is_final=False) results: updates the dim partial line in
        the translation pane if the utterance is still in-progress.

        For final results: uses the per-row slot registered in _finalise_source
        so that translations arriving after the next utterance has already
        started are not dropped.
        """
        try:
            trans_log = self.query_one("#translation-log", RichLog)
        except Exception:
            return  # widget not yet mounted or app is shutting down

        # ── Partial translation: update the dim "…" live ────────────────
        if not is_final:
            if self._pending_row_id == row_id:
                # Replace just the dim partial translation line.
                self._pop_partial(trans_log, self._trans_strip_start)
                line = _make_lang_tag_line(
                    translation, from_lang, confidence, is_final=False
                )
                trans_log.write(line)
            # If row is no longer pending, the final translation is already in
            # flight — discard this stale partial result.
            return

        # ── Final translation: slot-based in-place update ────────────────
        slot_start = self._trans_slots.get(row_id)
        logger.debug(
            "_update_translation_row row=%d slot=%s trans_log_len=%d pending_row=%s",
            row_id,
            slot_start,
            len(trans_log.lines),
            self._pending_row_id,
        )
        if slot_start is None:
            # Slot not yet registered — _finalise_source hasn't run yet on the
            # main thread.  Stash and apply when it does.
            if row_id not in self._pending_trans:
                self._pending_trans[row_id] = (translation, from_lang, confidence)
            return

        # Save ALL lines that come after slot_start (subsequent slots'
        # placeholders + any dim partial).  We must restore them so that later
        # translations can still find their correct positions.
        tail_lines: list = list(trans_log.lines[slot_start + 1 :])
        # Remove everything from slot_start onwards (the "…" placeholder + tail).
        self._pop_partial(trans_log, slot_start)

        # Write the translation in place of the placeholder.
        # Build a Rich Text with a dim language+confidence prefix tag followed
        # by the translation text in normal (or dim italic for partials) style.
        line = _make_lang_tag_line(translation, from_lang, confidence, is_final)
        trans_log.write(line)
        # A full sentence may wrap; shift slots by its actual rendered height.
        shift = len(trans_log.lines) - slot_start - 1
        if shift:
            for pending_id, position in list(self._trans_slots.items()):
                if pending_id != row_id and position > slot_start:
                    self._trans_slots[pending_id] = position + shift
            if self._trans_strip_start > slot_start:
                self._trans_strip_start += shift

        # Re-append the tail lines.
        if tail_lines:
            for strip in tail_lines:
                trans_log.lines.append(strip)
            trans_log._line_cache.clear()  # type: ignore[attr-defined]
            trans_log.virtual_size = trans_log.virtual_size._replace(
                height=len(trans_log.lines)
            )
            trans_log.refresh()

        if is_final:
            self._trans_slots.pop(row_id, None)
            try:
                self._trans_slot_order.remove(row_id)
            except ValueError:
                pass
            if self._pending_row_id == row_id:
                self._pending_row_id = None

    # ------------------------------------------------------------------
    # Translator status callback
    # ------------------------------------------------------------------

    def _on_translator_status(self, message: str) -> None:
        # This callback may be invoked from the main thread (e.g. set_engine()
        # called directly from a UI event handler) OR from a background thread
        # (network probe, argostranslate download).  Textual's call_from_thread
        # raises RuntimeError when called from the main thread, so branch on it.
        if threading.current_thread() is threading.main_thread():
            self._apply_translator_status(message)
        else:
            self.call_from_thread(self._apply_translator_status, message)

    def _on_capture_status(self, message: str) -> None:
        """Called from the AudioCapture thread when the device state changes."""
        try:
            self.call_from_thread(self._apply_capture_status, message)
        except Exception:
            pass  # app shutting down

    # ------------------------------------------------------------------
    # Widget event handlers
    # ------------------------------------------------------------------

    async def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "source-select":
            if event.value is Select.BLANK:
                return
            new_idx = int(str(event.value))
            # Ignore clicks on group-header sentinel options
            if new_idx < 0:
                return
            if new_idx != self._current_device_index:
                self._flush_sentence_buffer()
            self._current_device_index = new_idx
            if self._capture:
                entry = self._device_entry(new_idx)
                self._capture.set_device(new_idx, entry["name"] if entry else "")
            self._update_status()

        elif event.select.id == "lang-select":
            if event.value is Select.BLANK:
                return
            self._flush_sentence_buffer()
            self._target_lang = str(event.value)
            # Pre-download argostranslate package if needed
            detected = self._detected_lang if self._detected_lang != "?" else "en"
            self._translator.ensure_package(detected, self._target_lang)
            self._update_status()

        elif event.select.id == "model-select":
            if event.value is Select.BLANK:
                return
            new_model = str(event.value)
            if new_model != self._model_name:
                self._model_name = new_model
                self._restart_transcriber()

        elif event.select.id == "engine-select":
            if event.value is Select.BLANK:
                return
            # push_screen_wait requires a worker context.
            self.run_worker(
                self._select_engine(str(event.value), event.select),
                group="engine-select",
                exclusive=True,
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear-btn":
            self.action_clear()

    # ------------------------------------------------------------------
    # Translation engine selection
    # ------------------------------------------------------------------

    async def _select_engine(self, option: str, select: Select) -> None:
        engine, _, model = option.partition(":")
        # Availability is an HTTP/cache check; keep it off the event loop so
        # a slow Ollama server cannot freeze the TUI.
        available = await asyncio.to_thread(
            self._translator.is_model_available, engine, model
        )
        if not available:
            if not await self._confirm_and_pull(engine, model, option):
                select.value = self._engine_option
                return
        self._activate_engine(engine, model, option)

    async def _confirm_and_pull(
        self, engine: str, model: str, option: str
    ) -> bool:
        confirmed = await self.push_screen_wait(
            ConfirmDownload(model or engine, self._engine_size(option))
        )
        if not confirmed:
            return False
        self._set_activity(f"Downloading {model or engine}…")
        ok = await asyncio.to_thread(
            self._translator.pull_model, engine, model, self._pull_progress
        )
        if not ok:
            self._set_activity(
                f"Download failed: {model or engine}", "status-error", duration=8.0
            )
        else:
            self._set_activity(
                f"{model or engine} ready", "status-listening", duration=4.0
            )
        return ok

    def _activate_engine(self, engine: str, model: str, option: str) -> None:
        if option == self._engine_option:
            return
        self._flush_sentence_buffer()
        self._translator.set_engine(engine)
        if engine == ENGINE_OLLAMA and model:
            self._translator.set_ollama_model(model)
        if engine == ENGINE_NLLB and model:
            self._translator.set_nllb_model(model)
        self._engine = engine
        self._engine_option = option
        self._set_activity(
            f"Translation: {model or engine}", "status-listening", duration=4.0
        )

    def _pull_progress(self, message: str) -> None:
        try:
            self.call_from_thread(self._set_activity, message)
        except Exception:
            pass  # app shutting down

    def _engine_size(self, option: str) -> str:
        for label, value in TRANSLATION_ENGINES:
            if value == option:
                start = label.rfind("(")
                if start != -1 and label.endswith(")"):
                    return label[start + 1 : -1]
        return ""

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_clear(self) -> None:
        self.query_one("#source-log", RichLog).clear()
        self.query_one("#translation-log", RichLog).clear()
        self._pending_row_id = None
        self._source_strip_start = 0
        self._trans_strip_start = 0
        self._trans_slots.clear()
        self._trans_slot_order.clear()
        self._row_count = 0
        self._last_source_text = ""
        self._sentence_buffer.clear()

    async def action_focus_source(self) -> None:
        # PipeWire nodes and their indices churn, so rebuild before showing.
        # PortAudio enumeration blocks; keep it off the event loop.
        devices = await asyncio.to_thread(list_input_devices)
        self._apply_devices(devices)
        self.query_one("#source-select").focus()

    def action_focus_target(self) -> None:
        self.query_one("#lang-select").focus()

    def action_focus_model(self) -> None:
        self.query_one("#model-select").focus()

    def action_focus_engine(self) -> None:
        self.query_one("#engine-select").focus()

    # ------------------------------------------------------------------
    # Status bar helpers
    # ------------------------------------------------------------------

    def _update_status(self) -> None:
        # A background activity (model load, download, device wait) has
        # priority over the listening line until it finishes or expires.
        if self._activity is not None:
            if self._activity_until and time.monotonic() >= self._activity_until:
                self._activity = None
            else:
                self._set_status(self._activity, self._activity_class)
                return

        if not self._model_ready:
            self._set_status(
                f"● Loading Whisper model ({self._model_name})…", "status-loading"
            )
            return

        device_name = self._device_name(self._current_device_index)
        target_name = LANGUAGE_NAMES.get(self._target_lang, self._target_lang)
        lang_info = (
            f"detected: {LANGUAGE_NAMES.get(self._detected_lang, self._detected_lang)}"
            if self._detected_lang != "?"
            else "detecting…"
        )
        self._set_status(
            f"● Listening — {device_name} — {lang_info} → {target_name}"
            f" — {self._transcriber.backend_name if self._transcriber else 'CPU'}",
            "status-listening",
        )

    def _set_activity(
        self,
        message: str,
        css_class: str = "status-loading",
        duration: Optional[float] = None,
    ) -> None:
        """Show background work in the status bar.

        duration=None keeps the message until replaced or cleared; a value
        expires it and falls back to the listening line.
        """
        self._activity = message
        self._activity_class = css_class
        self._activity_until = (
            time.monotonic() + duration if duration is not None else 0.0
        )
        self._update_status()
        if duration is not None:
            try:
                self.set_timer(duration + 0.05, self._update_status)
            except Exception:
                pass  # app shutting down

    def _clear_activity(self) -> None:
        if self._activity is None:
            return
        self._activity = None
        self._update_status()

    def _apply_translator_status(self, message: str) -> None:
        """Classify engine messages: long operations persist, notes expire."""
        lowered = message.lower()
        if any(
            word in lowered
            for word in ("loading", "downloading", "pulling", "unloading", "releasing")
        ):
            self._set_activity(message)
        elif "failed" in lowered or "unavailable" in lowered or "error" in lowered:
            self._set_activity(message, "status-error", duration=8.0)
        else:
            self._set_activity(message, "status-listening", duration=4.0)

    def _apply_capture_status(self, message: str) -> None:
        lowered = message.lower()
        if "waiting" in lowered:
            self._set_activity(message, "status-loading")
        elif "unavailable" in lowered:
            self._set_activity(message, "status-error", duration=8.0)
        else:
            self._set_activity(message, "status-listening", duration=4.0)

    def _set_status(self, message: str, css_class: str = "status-listening") -> None:
        bar = self.query_one("#status-bar", Label)
        bar.update(message)
        bar.remove_class("status-loading", "status-listening", "status-error")
        bar.add_class(css_class)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _device_options(self) -> list[tuple[str, int]]:
        """Build grouped (prompt, value) pairs for GroupedSelect.

        Group headers have negative sentinel values and will be rendered as
        disabled (non-selectable) rows.  Device entries use their sounddevice
        index as the value.
        """
        # Bucket devices by group
        buckets: dict[str, list[dict]] = {"app": [], "monitor": [], "sound": []}
        for d in self._devices:
            g = d.get("group", "sound")
            if g in buckets:
                buckets[g].append(d)

        sentinels = {"app": _GRP_APP, "monitor": _GRP_MONITOR, "sound": _GRP_INPUT}
        options: list[tuple[str, int]] = []
        for group in _GROUP_ORDER:
            devs = buckets[group]
            if not devs:
                continue
            options.append((_GROUP_LABELS[group], sentinels[group]))
            for d in devs:
                options.append((d["name"], d["index"]))
        return options

    def _device_name(self, index: int) -> str:
        for d in self._devices:
            if d["index"] == index:
                # Truncate long names
                name = d["name"]
                return name[:30] + "…" if len(name) > 30 else name
        return f"device {index}"

    def _device_entry(self, index: int) -> Optional[dict]:
        return next((d for d in self._devices if d["index"] == index), None)

    def _refresh_devices(self) -> None:
        """Rebuild the source list; blocking, so callers use a thread."""
        self._apply_devices(list_input_devices())

    def _apply_devices(self, devices: list[dict]) -> None:
        self._devices = devices
        try:
            select = self.query_one("#source-select", GroupedSelect)
        except Exception:
            return
        select.set_options(self._device_options())
        if self._device_entry(self._current_device_index):
            select.value = self._current_device_index


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = PolyglotApp()
    app.run()


if __name__ == "__main__":
    main()
