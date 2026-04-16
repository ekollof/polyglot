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

import concurrent.futures
import logging
import queue
import threading
from typing import Optional

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
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
from textual import work

from audio import AudioCapture, list_input_devices, default_device
from transcriber import Transcriber
from translator import (
    Translator,
    ENGINE_GOOGLE,
    ENGINE_MYMEMORY,
    ENGINE_LIBRETRANSLATE,
    ENGINE_OLLAMA,
    ENGINE_ARGOS,
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

# Translation engines available in the UI.
TRANSLATION_ENGINES: list[tuple[str, str]] = [
    ("Google", ENGINE_GOOGLE),
    ("MyMemory", ENGINE_MYMEMORY),
    ("LibreTranslate", ENGINE_LIBRETRANSLATE),
    ("Ollama (local)", ENGINE_OLLAMA),
    ("Argos (offline)", ENGINE_ARGOS),
]

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
        self._translator = Translator(on_status=self._on_translator_status)
        self._translation_worker: Optional[TranslationWorker] = None
        self._current_device_index: int = default_device(self._devices)
        self._target_lang: str = "en"
        self._detected_lang: str = "?"
        self._model_name: str = "turbo"
        self._engine: str = ENGINE_GOOGLE
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
                    value=ENGINE_GOOGLE,
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
        self._capture = AudioCapture(
            device_index=self._current_device_index,
            audio_queue=self._audio_queue,
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
            device="cuda",
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
            device="cuda",
        )
        self._transcriber.start()
        self._poll_model_ready()

    # ------------------------------------------------------------------
    # Transcription callbacks (called from transcriber thread)
    # ------------------------------------------------------------------

    def _on_partial_result(
        self, detected_lang: str, partial_text: str, confidence: Optional[float]
    ) -> None:
        """Called by Transcriber thread with each interim partial transcription."""
        self._detected_lang = detected_lang
        row_id = self._next_row_id  # reuse the same row until finalised
        self.call_from_thread(self._show_partial, row_id, partial_text, detected_lang)

        # Partial translations are intentionally NOT submitted to the translation
        # worker. Submitting partials causes slot/strip position races between
        # _show_partial and _finalise_source on the main thread. The translation
        # pane shows "…" (dim) while an utterance is in progress; the final
        # translation arrives after the utterance is committed.

    def _on_transcription_result(
        self,
        detected_lang: str,
        source_text: str,
        confidence: Optional[float],
    ) -> None:
        """Called by Transcriber thread when an utterance is fully transcribed."""
        self._detected_lang = detected_lang
        target = self._target_lang

        row_id = self._next_row_id
        self._next_row_id += 1

        self.call_from_thread(self._finalise_source, row_id, source_text, detected_lang)

        # Empty source_text means _commit failed (inference error) — the partial
        # has been cleared; nothing to translate.
        if not source_text:
            return

        self._last_source_text = source_text

        assert self._translation_worker is not None
        self._translation_worker.submit(
            row_id=row_id,
            text=source_text,
            from_lang=detected_lang,
            to_lang=target,
            is_final=True,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Speaker-change / non-speech callbacks (called from transcriber thread)
    # ------------------------------------------------------------------

    def _on_speaker_change(self) -> None:
        """Called by Transcriber when a speaker change is detected."""
        self.call_from_thread(self._insert_speaker_separator)

    def _on_non_speech(self, reason: str) -> None:
        """Called by Transcriber when an utterance is classified as non-speech."""
        logger.debug("Non-speech utterance suppressed (%s).", reason)

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
        source_log = self.query_one("#source-log", RichLog)
        trans_log = self.query_one("#translation-log", RichLog)

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
        source_log = self.query_one("#source-log", RichLog)
        trans_log = self.query_one("#translation-log", RichLog)

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
        # slot_start now holds the translation; tail starts at slot_start + 1.

        # Re-append the tail lines.
        if tail_lines:
            for strip in tail_lines:
                trans_log.lines.append(strip)
            trans_log._line_cache.clear()  # type: ignore[attr-defined]
            trans_log.virtual_size = trans_log.virtual_size._replace(
                height=len(trans_log.lines)
            )
            trans_log.refresh()

            # Update every downstream slot index.  Each slot that was after
            # slot_start was shifted: old position → same absolute position
            # (the translation replaced the "…" 1-for-1, so no net shift).
            # But if we removed N tail lines and re-appended N tail lines the
            # absolute positions are unchanged — no adjustment needed for those.
            # The only slot whose index changes is the one we just resolved
            # (row_id itself), which we remove below.

            # Update _trans_strip_start if the dim partial was in the tail.
            # The partial's position relative to slot_start is preserved.
            if self._trans_strip_start > slot_start:
                # Partial was in the tail; its absolute index is unchanged
                # because we replaced the "…" 1-for-1 (no net line count shift).
                pass  # _trans_strip_start stays correct

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
        import threading

        if threading.current_thread() is threading.main_thread():
            self._set_status(message, "status-loading")
        else:
            self.call_from_thread(self._set_status, message, "status-loading")

    # ------------------------------------------------------------------
    # Widget event handlers
    # ------------------------------------------------------------------

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "source-select":
            if event.value is Select.BLANK:
                return
            new_idx = int(str(event.value))
            # Ignore clicks on group-header sentinel options
            if new_idx < 0:
                return
            self._current_device_index = new_idx
            if self._capture:
                self._capture.set_device(new_idx)
            self._update_status()

        elif event.select.id == "lang-select":
            if event.value is Select.BLANK:
                return
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
            new_engine = str(event.value)
            if new_engine != self._engine:
                self._engine = new_engine
                self._translator.set_engine(new_engine)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear-btn":
            self.action_clear()

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

    def action_focus_source(self) -> None:
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
            f"● Listening — {device_name} — {lang_info} → {target_name}",
            "status-listening",
        )

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app = PolyglotApp()
    app.run()


if __name__ == "__main__":
    main()
