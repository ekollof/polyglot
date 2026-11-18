"""
transcriber.py — VAD-gated utterance transcription with Whisper.

Architecture
------------
Audio chunks (float32 mono 16kHz) arrive from the queue continuously.
Each chunk is fed to a Silero-VAD iterator that detects speech/silence
boundaries in real time (~32ms latency per chunk).

There are three processing threads:

  1. VAD thread  (_run):
       Reads audio chunks, runs Silero-VAD, tracks utterance state, and
       posts work items onto _infer_queue.  Never touches the Whisper model.

  2. Inference thread  (_inference_loop):
       The ONLY thread that calls the Whisper model.  Pulls items from
       _infer_queue one at a time.  Because it is the sole consumer there
       are NO locks — _detected_lang is a plain attribute written and read
       exclusively by this thread.

  3. LangID thread (_run_langid):
       Classifies timestamped audio with SpeechBrain and publishes immutable
       evidence snapshots. It never calls Whisper or changes detected language.

Work items posted onto _infer_queue
------------------------------------
  ("detect", audio_np, captured_at) — detect language on recent confirmed speech
  ("partial", audio_np, lang) — run greedy transcription, fire on_partial
  ("commit", audio_np, lang, captured_at, boundary) — decode then signal a pause
  ("reset_lang", captured_at) — clear old speech and language context

The queue is a standard queue.Queue which is thread-safe by itself.
No additional locking is needed.

VAD parameters
--------------
- threshold: 0.3
- min_silence_duration_ms: 600 ms
- speech_pad_ms: 100 ms
- max_utterance_seconds: 25 s

Silero-VAD
----------
Loaded from the installed silero-vad package.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import torch

from sentence_buffer import completed_sentences
from speech_quality import (
    MIN_ROLLING_PAUSE_SECONDS,
    LanguageEvidence,
    closest_language_evidence,
    coalesce_inference_work,
    fuse_language_probabilities,
    rolling_audio_boundary,
    segment_confidence_reason,
)

logger = logging.getLogger(__name__)

WHISPER_RATE = 16_000  # Hz required by Whisper
VAD_CHUNK = 512  # samples per VAD call (32 ms @ 16 kHz)
MAX_UTTERANCE_SECONDS = 25  # absolute safety cap
# Maximum audio fed to a partial transcription pass.  Capped so that greedy
# inference stays fast even during very long utterances.
MAX_PARTIAL_SECONDS = 8

# Initial continuous-speech window. Scheduling adapts it up to eight seconds
# on slow hardware; fast models can commit without waiting seven seconds.
ROLLING_COMMIT_S = 4.0

# How often to fire interim (partial) results while speech is ongoing.
# turbo runs at ~3× real-time on ROCm, so 1.0 s gives it enough headroom
# to finish a partial pass before the next one is triggered.
INTERIM_INTERVAL_S = 1.0

# Wait for five seconds of speech before the first VAD language snapshot;
# very short fragments do not provide reliable multilingual context.
DETECT_SECONDS = 5.0

# How many seconds of continuous silence before the cached detected language
# is discarded.  This ensures that a long pause between speakers (e.g. video
# cut, topic change) doesn't carry a stale language into the next utterance.
SILENCE_LANG_RESET_S = 3.0

# ---------------------------------------------------------------------------
# Speech buffer for language detection
# ---------------------------------------------------------------------------
# After each commit that produces real speech, the audio is appended to a
# rolling deque.  Language detection runs on the concatenated buffer rather
# than on raw VAD snapshots, giving the detector 5–12 seconds of
# confirmed-speech audio instead of potentially noisy silence snapshots.
#
# This eliminates the main source of false language switches: a single short,
# quiet utterance at the tail of a sentence being mis-labelled by Whisper and
# poisoning _detected_lang for subsequent commits.
LANG_BUFFER_MIN_SECS: float = 5.0  # minimum speech before buffer-based detection
LANG_DETECT_WINDOW_SECS: float = 12.0
LANG_BUFFER_MAX_SECS: float = 30.0  # oldest audio evicted beyond this

# ---------------------------------------------------------------------------
# VAD tuning
# ---------------------------------------------------------------------------
VAD_THRESHOLD = 0.3
VAD_SPEECH_PAD_MS = 100

# Fixed end-of-speech silence threshold.  All three reference engines
# (RealtimeSTT=600ms, whisper_streaming=500ms, WhisperLive≈700ms) use a
# fixed value.  Adaptive thresholds based on utterance duration don't
# generalise well across languages and speaker styles.
VAD_SILENCE_MS = 600

# Pre-roll: include this many seconds of audio *before* the VAD "start" event.
# RealtimeSTT uses 1 s; 0.5 s is enough to capture the onset phoneme without
# adding much silence to the front of each utterance.
VAD_PREROLL_S = 0.5

# Content-based early commit: if the partial transcript has been identical for
# this many consecutive inference runs, treat the utterance as complete and
# commit without waiting for VAD silence.  Mirrors WhisperLive's
# same_output_threshold (7–10 cycles); 3 cycles at INTERIM_INTERVAL_S=1.0s
# means we commit ~3 s after the speaker stops changing their words.
CONTENT_COMMIT_STABLE_RUNS = 3


ResultCallback = Callable[[str, str, Optional[float]], None]
# (detected_lang, source_text, confidence)  confidence is None if unknown

PartialCallback = Callable[[str, str, Optional[float]], None]
# (detected_lang, partial_text, confidence) — fired periodically during active speech

SpeakerChangeCallback = Callable[[], None]
# Fired when cosine similarity between consecutive utterance encoder
# embeddings falls below SPEAKER_CHANGE_THRESHOLD.

NonSpeechCallback = Callable[[str], None]
# (reason) — fired when an utterance is classified as non-speech
# (e.g. "no_speech", "silence").

# ---------------------------------------------------------------------------
# Speaker-change detection parameters
# ---------------------------------------------------------------------------
# Cosine similarity threshold below which consecutive utterance encoder
# embeddings are considered to be from different speakers.
# Range: [-1, 1].  Empirically, same-speaker successive utterances tend to
# score 0.85–0.98; cross-speaker transitions tend to score 0.50–0.75.
# 0.75 is a conservative starting point — increases recall at the cost of
# slightly higher false-positive rate.
SPEAKER_CHANGE_THRESHOLD: float = 0.75

# Segments whose text is entirely Whisper noise/music tokens should be
# suppressed.  The regex matches the *whole* stripped text.
import re as _re

_NOISE_TOKEN_RE = _re.compile(
    r"^[\s♪♫♬♩]*"  # leading music notes / whitespace
    r"(\[(?:"
    r"music|applause|laughter|noise|silence|blank_audio"
    r"|inaudible|cheering|crowd|background music|singing"
    r")\]"
    r"|♪[^♪]*♪"  # ♪ … ♪ pairs
    r"|♪"
    r"|\((?:music|applause|laughter|inaudible|singing)\)"
    r")+"
    r"[\s♪♫♬♩.!,]*$",  # trailing punctuation
    _re.IGNORECASE,
)

# Seconds since the last confident language confirmation before we treat
# _detected_lang as stale and clear it.  Prevents a wrong language from
# lingering when subsequent detections keep falling below the confidence bar.
LANG_STALE_SECONDS: float = 30.0

# Confidence required before the rolling language detection is used as a
# per-commit decoding hint.  Below this, Whisper auto-detects as before.
COMMIT_LANGUAGE_HINT_MIN_CONFIDENCE: float = 0.8

# Quality below which a hinted decode is compared against auto-detection, so
# a stale hint (speaker switch, music) cannot force the wrong language.
COMMIT_LANGUAGE_RETRY_QUALITY: float = -0.65


def _is_noise_token(text: str) -> bool:
    """Return True if *text* consists entirely of Whisper noise/music tokens."""
    return bool(_NOISE_TOKEN_RE.match(text.strip()))


# Languages that use Arabic / Persian script.
_ARABIC_SCRIPT_LANGS: frozenset[str] = frozenset(
    {"fa", "ar", "ur", "ps", "ug", "ku", "sd"}
)


def _partial_script_mismatch(text: str, expected_lang: Optional[str]) -> bool:
    """Return True when text's dominant script doesn't match expected_lang.

    Used to suppress partial hallucinations that switch language.  Only
    triggers when expected_lang is a known Arabic-script language and the
    partial is predominantly Latin, or vice-versa (Latin-primary language but
    partial has predominantly Arabic script).  Does nothing when expected_lang
    is unknown/None — better to show than to over-suppress.
    """
    if not expected_lang or not text:
        return False
    latin_count = sum(
        1 for c in text if "\u0041" <= c <= "\u007a" or "\u00c0" <= c <= "\u024f"
    )
    arabic_count = sum(
        1
        for c in text
        if "\u0600" <= c <= "\u06ff"
        or "\u0750" <= c <= "\u077f"
        or "\ufb50" <= c <= "\ufdff"
    )
    total = latin_count + arabic_count
    if total < 4:
        return False  # too short to judge
    if expected_lang in _ARABIC_SCRIPT_LANGS:
        # Expect Arabic script; suppress if mostly Latin.
        return latin_count / total > 0.6
    else:
        # Expect Latin script; suppress if mostly Arabic.
        return arabic_count / total > 0.6


def _is_repetition_loop(text: str, min_repeats: int = 3) -> bool:
    """Return True if text is a Whisper hallucination repetition loop.

    Checks whether any word-level n-gram (n=1..5) appears >= min_repeats
    times consecutively.  For example:
        "کنم کنم کنم کنم" → True
        "این مجموعه از این مجموعه از این مجموعه" → True
        "واقعا فکر میکردم" → False
    """
    words = text.split()
    n = len(words)
    for gram_size in range(1, min(6, n // min_repeats + 1)):
        for start in range(n - gram_size * min_repeats + 1):
            gram = words[start : start + gram_size]
            count = 1
            pos = start + gram_size
            while pos + gram_size <= n and words[pos : pos + gram_size] == gram:
                count += 1
                pos += gram_size
            if count >= min_repeats:
                return True
    return False


# ---------------------------------------------------------------------------
# Lingua text-based language detection (module-level lazy singleton)
# ---------------------------------------------------------------------------
# Used as a cross-check on committed transcript text.  Lingua's rule-based
# engine is script-aware: Persian/Arabic characters trigger immediate high-
# confidence detection of 'fa'/'ar', making it an excellent veto for the
# case where Whisper correctly transcribes the script but mislabels it.
# Thread-safe: Lingua documentation explicitly guarantees this.


def _split_sentences(text: str) -> list[str]:
    """Emit completed sentences plus an unfinished tail for UI assembly."""
    complete, pending = completed_sentences(text)
    return complete + ([pending] if pending else [])


_lingua_detector = None  # lazily initialised on first call


def _lingua_detect(text: str) -> Optional[str]:
    """Return ISO 639-1 language code for *text*, or None if uncertain.

    Uses Lingua's full-language detector (75 languages).  The detector is
    built lazily on first call and cached for the life of the process.
    Requires twelve letters, a top confidence of at least 0.80, and a
    0.20 margin over the runner-up; audio evidence must corroborate overrides.
    """
    if sum(character.isalpha() for character in text) < 12:
        return None
    global _lingua_detector
    try:
        if _lingua_detector is None:
            from lingua import LanguageDetectorBuilder

            _lingua_detector = (
                LanguageDetectorBuilder.from_all_languages()
                .with_preloaded_language_models()
                .build()
            )
        values = _lingua_detector.compute_language_confidence_values(text)
        if not values:
            return None
        top = values[0]  # sorted descending by confidence
        if top.value < 0.80 or (len(values) > 1 and top.value - values[1].value < 0.20):
            return None
        # iso_code_639_1 is an enum; .name is e.g. "EN", "FA", "FR", "JA"
        return top.language.iso_code_639_1.name.lower()
    except Exception as exc:
        logger.debug("Lingua detection error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Silero-VAD loader
# ---------------------------------------------------------------------------


def _load_silero_vad():
    """Load the packaged Silero-VAD model and VADIterator."""
    from silero_vad import VADIterator, load_silero_vad  # type: ignore

    model = load_silero_vad(onnx=False)
    return model, VADIterator


# ---------------------------------------------------------------------------
# Transcriber
# ---------------------------------------------------------------------------


class Transcriber:
    """
    Runs Silero-VAD + Whisper inference in background threads.

    There are exactly two background threads:
      - VAD thread   : reads audio, runs VAD, enqueues work items
      - Inference thread : the sole Whisper caller; no locking needed

    Parameters
    ----------
    audio_queue:
        Queue fed by AudioCapture, containing float32 mono 16kHz chunks.
    on_result:
        Callback called with (detected_language, source_text, confidence) when
        an utterance is finalised.
    on_partial:
        Optional callback called periodically while speech is ongoing, with
        (detected_language, partial_text, confidence).
    on_speaker_change:
        Optional callback fired when the encoder-embedding cosine similarity
        between consecutive utterances falls below SPEAKER_CHANGE_THRESHOLD.
        Called with no arguments — the UI inserts a visual separator.
    on_non_speech:
        Optional callback fired when an utterance is classified as non-speech
        (all segments suppressed or no_speech_prob too high).  Called with a
        string reason, e.g. "no_speech".
    model_name:
        Whisper model identifier (e.g. "turbo", "small", "medium").
    device:
        PyTorch device string, or None to select a validated accelerator.
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        on_result: ResultCallback,
        on_partial: Optional[PartialCallback] = None,
        on_speaker_change: Optional[SpeakerChangeCallback] = None,
        on_non_speech: Optional[NonSpeechCallback] = None,
        model_name: str = "turbo",
        device: Optional[str] = None,
    ):
        self.audio_queue = audio_queue
        self.on_result = on_result
        self.on_partial = on_partial
        self.on_speaker_change = on_speaker_change
        self.on_non_speech = on_non_speech
        self.model_name = model_name
        self._requested_device = device
        self._backend_name = "Detecting accelerator…"
        self._fp16 = False
        self._rolling_commit_seconds = ROLLING_COMMIT_S
        self._partials_enabled = True
        self._quality_retry_allowed = True
        self._last_language_inference_at = 0.0
        self._last_transcribed_lang = None
        self.device = (
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._langid_device = self.device if self.device.startswith("cuda") else "cpu"

        self._stop_event = threading.Event()
        self._vad_thread: Optional[threading.Thread] = None
        self._infer_thread: Optional[threading.Thread] = None
        self._langid_thread: Optional[threading.Thread] = None
        self._whisper_model = None
        self._ready = threading.Event()

        # Set by the inference thread when a partial transcript has stabilised
        # (CONTENT_COMMIT_STABLE_RUNS identical consecutive runs).  The VAD
        # thread polls this and triggers a commit + buffer reset, matching
        # WhisperLive's same_output_threshold mechanism.
        self._commit_requested = threading.Event()

        # Work queue between VAD thread and inference thread.
        # Items: ("detect", audio) | ("partial", audio, lang) | ("commit", audio, lang)
        self._infer_queue: queue.Queue = queue.Queue()

        # Separate queue for the SpeechBrain LangID thread.  VAD thread posts
        # the same detection snapshot here; the LangID thread processes it
        # independently without blocking Whisper inference.
        # Items: np.ndarray audio snapshot, or None (sentinel to stop thread).
        self._langid_queue: queue.Queue = queue.Queue()

        # _detected_lang/_detected_confidence are ONLY read and written by the
        # inference thread.  The VAD thread reads lang via _vad_lang (a plain
        # Optional[str] written by the inference thread).  Python's GIL makes
        # single-reference writes/reads of simple objects atomic, so no lock needed.
        self._detected_lang: Optional[str] = None  # inference thread only
        self._detected_confidence: Optional[float] = None  # inference thread only
        self._detected_lang_set_at: float = (
            0.0  # monotonic time when _detected_lang was last confirmed
        )
        self._vad_lang: Optional[str] = None  # shared, GIL-atomic

        # SpeechBrain LangID results — written by LangID thread, read by
        # inference thread.  Plain attribute writes are GIL-atomic in CPython.
        self._sb_lang: Optional[str] = None  # LangID thread → inference thread
        self._sb_confidence: Optional[float] = None  # LangID thread → inference thread
        self._sb_history: tuple[LanguageEvidence, ...] = ()
        self._language_reset_at = 0.0
        self._language_candidate: Optional[str] = None
        self._language_candidate_hits = 0
        self._last_detect_evidence = None

        # Rolling speech buffer for language detection.
        # Each entry is a numpy audio chunk from a confirmed-speech commit.
        # Language detection runs on the concatenation of all chunks when the
        # buffer contains at least LANG_BUFFER_MIN_SECS of audio.
        # Only touched by the inference thread — no lock needed.
        self._speech_buffer: collections.deque = collections.deque()
        self._speech_buffer_secs: float = 0.0
        self._speech_buffer_revision = 0
        self._speech_buffer_at = 0.0

        # Last committed text, fed back as Whisper's initial_prompt to keep
        # context consistent across rolling-commit window boundaries.
        # Capped at ~200 chars (~224 tokens).  Inference thread only.
        # Only used when the commit language matches the prompt language —
        # a mismatched prompt biases the decoder toward the prompt's language
        # and causes Whisper to translate instead of transcribe.
        self._initial_prompt: str = ""
        self._initial_prompt_lang: Optional[str] = None  # language of _initial_prompt

        # Stuck-partial detection: if on_partial emits the same text N times
        # consecutively, suppress further emissions until the text changes.
        # Inference thread only.
        self._last_partial_text: str = ""
        self._stuck_partial_count: int = 0

        # Speaker embedding from the previous utterance (inference thread only).
        # Normalised L2 unit vector, shape [n_state].  None until at least one
        # utterance has been successfully encoded.
        self._last_speaker_embed: Optional[torch.Tensor] = None  # inference thread only

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        return self._ready.is_set()

    def start(self) -> None:
        """Start the VAD, inference, and LangID threads (loads models asynchronously)."""
        if self._vad_thread and self._vad_thread.is_alive():
            return
        self._stop_event.clear()
        # Inference thread starts first; VAD thread waits for _ready.
        self._infer_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="Inference"
        )
        self._infer_thread.start()
        self._vad_thread = threading.Thread(target=self._run, daemon=True, name="VAD")
        self._vad_thread.start()
        # LangID thread loads SpeechBrain independently — does not block VAD/Whisper.
        self._langid_thread = threading.Thread(
            target=self._run_langid, daemon=True, name="LangID"
        )
        self._langid_thread.start()

    def stop(self) -> None:
        """Stop all threads."""
        self._stop_event.set()
        # Unblock the inference thread if it is waiting on an empty queue.
        self._infer_queue.put(("stop",))
        # Unblock the LangID thread.
        self._langid_queue.put(None)
        for t in (self._vad_thread, self._infer_thread, self._langid_thread):
            if t:
                try:
                    t.join(timeout=5)
                except KeyboardInterrupt, SystemExit:
                    pass

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        from inference_backend import backend_name

        return backend_name(self._whisper_model, self._backend_name)

    def _load_models(self):
        from inference_backend import load_whisper

        model, self._backend_name = load_whisper(
            self.model_name, device=self._requested_device
        )
        self.device = str(model.device)
        self._fp16 = model.inference_fp16
        logger.info("Whisper model loaded — %s", self._backend_name)
        return model

    # ------------------------------------------------------------------
    # LangID thread  (SpeechBrain VoxLingua107, runs in parallel to Whisper)
    # ------------------------------------------------------------------

    def _run_langid(self) -> None:
        """Dedicated thread for SpeechBrain spoken-language identification.

        Loads the VoxLingua107 ECAPA-TDNN model from HuggingFace (cached to
        ~/.cache/speechbrain/).  Processes audio snapshots posted by the VAD
        thread onto _langid_queue independently of the Whisper inference thread
        — no GPU contention because SpeechBrain inference is fast (~10-30 ms).

        Results (_sb_lang, _sb_confidence) are written as plain attribute
        assignments; CPython's GIL makes these atomic so no lock is needed.

        VoxLingua107 label quirks (fixed in _SB_LANG_FIXES):
          "iw" → "he" (Hebrew — obsolete ISO code)
          "jw" → "jv" (Javanese — incorrect ISO code)
        """
        _SB_LANG_FIXES = {"iw": "he", "jw": "jv"}

        try:
            from pathlib import Path as _Path

            from speechbrain.inference.classifiers import EncoderClassifier

            logger.info("LangID — loading SpeechBrain VoxLingua107 model…")
            _savedir = str(
                _Path.home() / ".cache" / "speechbrain" / "lang-id-voxlingua107-ecapa"
            )
            language_id = EncoderClassifier.from_hparams(
                source="speechbrain/lang-id-voxlingua107-ecapa",
                savedir=_savedir,
                run_opts={"device": self._langid_device},
            )
            logger.info("LangID — SpeechBrain VoxLingua107 loaded.")
        except Exception as exc:
            logger.error(
                "LangID — failed to load SpeechBrain model; "
                "falling back to Whisper-only detection: %s",
                exc,
            )
            return

        while not self._stop_event.is_set():
            try:
                item = self._langid_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:  # stop sentinel
                break

            captured_at, audio = item
            try:
                import torch as _torch

                # SpeechBrain expects [batch, time] float32 tensor at 16 kHz.
                sig = _torch.from_numpy(audio).unsqueeze(0).float()
                prediction = language_id.classify_batch(sig)
                # prediction[3]: list of label strings, e.g. ["fa: Persian"]
                # prediction[1]: tensor of top-1 log-likelihoods; .exp() → prob
                raw_label: str = prediction[3][0]
                iso: str = raw_label.split(":")[0].strip()
                iso = _SB_LANG_FIXES.get(iso, iso)
                prob: float = float(prediction[1].exp().item())
                logger.debug("LangID SpeechBrain: %s (p=%.2f)", iso, prob)
                # GIL-atomic writes — no lock needed
                self._sb_lang = iso
                self._sb_confidence = prob
                scores = prediction[0].reshape(-1).exp().cpu()
                labels = language_id.hparams.label_encoder.decode_torch(
                    _torch.arange(len(scores))
                )
                probabilities = {}
                for label, score in zip(labels, scores.tolist()):
                    code = label.split(":")[0].strip()
                    code = _SB_LANG_FIXES.get(code, code)
                    probabilities[code] = probabilities.get(code, 0.0) + score
                evidence = LanguageEvidence(captured_at, probabilities)
                # Publish one coherent immutable history snapshot without locks.
                self._sb_history = (*self._sb_history, evidence)[-8:]
            except Exception as exc:
                logger.error("LangID inference error: %s", exc, exc_info=True)

    # ------------------------------------------------------------------
    # Inference thread  (the ONLY Whisper caller)
    # ------------------------------------------------------------------

    def _inference_loop(self) -> None:
        """Single thread that serialises all Whisper model calls."""
        # Load only Whisper here; VAD thread loads Silero-VAD separately.
        whisper_model = self._load_models()
        self._whisper_model = whisper_model
        self._ready.set()

        # _detected_lang lives exclusively on this thread — no lock needed.
        # We also expose it as _vad_lang for the VAD thread to read cheaply.

        pending = collections.deque()
        while not self._stop_event.is_set():
            if not pending:
                try:
                    pending.append(self._infer_queue.get(timeout=0.2))
                except queue.Empty:
                    continue
            while True:
                try:
                    pending.append(self._infer_queue.get_nowait())
                except queue.Empty:
                    break
            pending = collections.deque(coalesce_inference_work(pending))
            item = pending.popleft()
            kind = item[0]
            if kind == "stop":
                break
            if kind == "reset_lang":
                self._reset_language_context(item[1] if len(item) > 1 else None)
            elif kind == "detect":
                self._infer_detect(
                    whisper_model, item[1], item[2] if len(item) > 2 else None
                )
            elif kind == "partial":
                if self._partials_enabled:
                    started = time.monotonic()
                    self._infer_partial(whisper_model, item[1], item[2])
                    if len(item[1]) >= 2 * WHISPER_RATE:
                        self._partials_enabled = (
                            time.monotonic() - started
                            < self._rolling_commit_seconds * 0.25
                        )
            elif kind == "commit":
                captured_at = item[3] if len(item) > 3 else time.monotonic()
                started = time.monotonic()
                queue_delay = max(0.0, started - captured_at)
                self._quality_retry_allowed = queue_delay < 1.0 and not any(
                    work[0] == "commit" for work in pending
                )
                revision = self._speech_buffer_revision
                self._infer_commit(whisper_model, item[1], item[2], captured_at)
                if len(item) > 4 and item[4] and self.on_non_speech:
                    # Deliver the boundary after its transcript, preserving FIFO.
                    self.on_non_speech("utterance_end")
                if self._speech_buffer_revision != revision:
                    # Detection owns language state and sees newly confirmed speech,
                    # even when VAD detection requests were coalesced under load.
                    self._infer_detect(whisper_model, item[1], captured_at)
                processing = time.monotonic() - started
                audio_seconds = len(item[1]) / WHISPER_RATE
                self._partials_enabled = processing < audio_seconds * 0.25
                # Amortize fixed encoder cost on slow hardware without letting
                # continuous speech wait the old seven seconds on fast models.
                target = min(8.0, max(ROLLING_COMMIT_S, processing * 1.25))
                self._rolling_commit_seconds = (
                    0.7 * self._rolling_commit_seconds + 0.3 * target
                )
                logger.debug(
                    "Realtime: audio=%.2fs processing=%.2fs queue=%.2fs lag=%.2fs window=%.1fs partials=%s",
                    audio_seconds,
                    processing,
                    queue_delay,
                    max(0.0, time.monotonic() - captured_at),
                    self._rolling_commit_seconds,
                    self._partials_enabled,
                )

    def _reset_language_context(self, captured_at=None):
        logger.debug("Resetting language context after silence")
        self._language_reset_at = (
            captured_at if captured_at is not None else time.monotonic()
        )
        self._detected_lang = None
        self._detected_confidence = None
        self._detected_lang_set_at = 0.0
        self._vad_lang = None
        self._language_candidate = None
        self._language_candidate_hits = 0
        self._last_detect_evidence = None
        self._last_language_inference_at = 0.0
        self._last_transcribed_lang = None
        self._last_speaker_embed = None
        self._initial_prompt = ""
        self._initial_prompt_lang = None
        self._last_partial_text = ""
        self._stuck_partial_count = 0
        self._commit_requested.clear()
        self._speech_buffer.clear()
        self._speech_buffer_secs = 0.0
        self._speech_buffer_at = 0.0
        self._speech_buffer_revision += 1

        if self.on_non_speech:
            self.on_non_speech("silence")

    # ------------------------------------------------------------------

    def _push_speech_buffer(self, audio: np.ndarray, captured_at=None) -> None:
        """Append confirmed-speech audio to the rolling language-detection buffer.

        Called by _infer_commit after at least one segment is emitted.
        Evicts the oldest chunks when the buffer exceeds LANG_BUFFER_MAX_SECS.
        Inference thread only — no locking required.
        """
        self._speech_buffer_revision += 1
        self._speech_buffer_at = (
            captured_at if captured_at is not None else time.monotonic()
        )
        secs = len(audio) / WHISPER_RATE
        self._speech_buffer.append(audio)
        self._speech_buffer_secs += secs
        while self._speech_buffer_secs > LANG_BUFFER_MAX_SECS and self._speech_buffer:
            oldest = self._speech_buffer.popleft()
            self._speech_buffer_secs -= len(oldest) / WHISPER_RATE

    def _buffer_audio_for_detect(self, fallback: np.ndarray) -> np.ndarray:
        """Return audio to use for language detection.

        If the speech buffer has at least LANG_BUFFER_MIN_SECS of confirmed
        speech, return the most recent LANG_DETECT_WINDOW_SECS of that buffer.
        Otherwise return the raw VAD snapshot passed as fallback.
        """
        if self._speech_buffer_secs >= LANG_BUFFER_MIN_SECS:
            combined = np.concatenate(list(self._speech_buffer))
            # Prefer the newest utterance. Only borrow older confirmed speech
            # to reach five seconds, rather than letting old speakers dominate.
            max_samples = max(
                int(LANG_BUFFER_MIN_SECS * WHISPER_RATE),
                min(
                    int(LANG_DETECT_WINDOW_SECS * WHISPER_RATE),
                    len(self._speech_buffer[-1]),
                ),
            )
            if len(combined) > max_samples:
                combined = combined[-max_samples:]
            logger.debug(
                "Lang detect on recent confirmed speech (%.1fs)",
                len(combined) / WHISPER_RATE,
            )
            return combined
        return fallback

    def _speechbrain_evidence(self, captured_at):
        return closest_language_evidence(
            self._sb_history, captured_at, self._language_reset_at
        )

    def _result_confidence(self, language, captured_at=None):
        if language == self._detected_lang:
            return self._detected_confidence
        evidence = self._speechbrain_evidence(
            captured_at if captured_at is not None else time.monotonic()
        )
        if evidence and evidence.probabilities:
            top = max(evidence.probabilities, key=evidence.probabilities.get)
            if top == language:
                return evidence.probabilities[top]
        return None

    def _infer_detect(self, model, audio: np.ndarray, captured_at=None) -> None:
        """Detect from recent confirmed speech, with aligned audio evidence."""
        now = time.monotonic()
        captured_at = captured_at if captured_at is not None else now
        if (
            self._detected_lang is not None
            and now - self._detected_lang_set_at > LANG_STALE_SECONDS
        ):
            self._detected_lang = None
            self._detected_confidence = None
            self._vad_lang = None
        use_buffer = self._speech_buffer_secs >= LANG_BUFFER_MIN_SECS
        audio = self._buffer_audio_for_detect(audio)
        if use_buffer:
            captured_at = self._speech_buffer_at
        evidence_key = (
            ("speech", self._speech_buffer_revision)
            if use_buffer
            else ("snapshot", captured_at)
        )
        if evidence_key == self._last_detect_evidence:
            return
        sb = self._speechbrain_evidence(captured_at)
        if (
            sb
            and self._detected_lang is not None
            and self._last_transcribed_lang == self._detected_lang
            and (self._detected_confidence or 0.0) >= 0.8
            and now - self._last_language_inference_at < 10.0
            and max(sb.probabilities, key=sb.probabilities.get) == self._detected_lang
            and sb.probabilities[self._detected_lang] >= 0.8
        ):
            return
        if len(audio) < WHISPER_RATE:
            return
        try:
            import whisper as _whisper

            if not use_buffer:
                audio = _normalize_audio(audio, float(np.sqrt(np.mean(audio**2))))
            # Match Whisper.transcribe's padding exactly, allowing encoder reuse.
            mel = _whisper.log_mel_spectrogram(
                torch.from_numpy(audio),
                n_mels=model.dims.n_mels,
                padding=_whisper.audio.N_SAMPLES,
            )
            mel = (
                _whisper.pad_or_trim(mel, _whisper.audio.N_FRAMES)
                .to(
                    device=model.device,
                    dtype=torch.float16 if self._fp16 else torch.float32,
                )
                .unsqueeze(0)
            )
            _, probabilities = model.detect_language(mel)
            self._last_language_inference_at = time.monotonic()
            whisper_probs = probabilities[0]
            sb = self._speechbrain_evidence(captured_at)
            lang, confidence, margin, agreement = fuse_language_probabilities(
                whisper_probs, sb.probabilities if sb else None
            )
            self._last_detect_evidence = evidence_key
            logger.debug(
                "Language evidence: lang=%s confidence=%.2f margin=%.2f agreement=%s sb_age=%s",
                lang,
                confidence,
                margin,
                agreement,
                round(captured_at - sb.captured_at, 1) if sb else None,
            )
            if lang is None or confidence < 0.65 or margin < 0.15:
                self._language_candidate = None
                self._language_candidate_hits = 0
                return
            accept = self._detected_lang in (None, lang)
            accept = accept or (confidence >= 0.85 and agreement) or confidence >= 0.95
            if not accept:
                if self._language_candidate == lang:
                    self._language_candidate_hits += 1
                else:
                    self._language_candidate = lang
                    self._language_candidate_hits = 1
                accept = self._language_candidate_hits >= 2
            if accept:
                self._detected_lang = lang
                self._detected_confidence = confidence
                self._detected_lang_set_at = now
                self._vad_lang = lang
                self._language_candidate = None
                self._language_candidate_hits = 0
        except Exception as exc:
            logger.error("Language detection error: %s", exc, exc_info=True)

    def _infer_partial(self, model, audio: np.ndarray, lang: Optional[str]) -> None:
        """Run a fast greedy pass and fire on_partial."""
        if not self.on_partial:
            return
        if len(audio) < 2 * WHISPER_RATE:
            logger.debug("Partial skipped — audio too short (%d samples).", len(audio))
            return
        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 1e-4:
            logger.debug("Partial skipped — silence (rms too low).")
            return
        if lang is None:
            logger.debug(
                "Partial — lang not yet detected, letting Whisper auto-detect."
            )

        # Never force language on partials — let Whisper auto-detect per window.
        # Forcing a stale or wrong detected language causes Whisper to translate
        # instead of transcribe, or output garbage when the language changes.
        audio = _normalize_audio(audio, rms)
        try:
            result = model.transcribe(
                audio,
                task="transcribe",
                language=None,
                fp16=self._fp16,
                temperature=0.0,
                beam_size=1,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                initial_prompt=None,
                condition_on_previous_text=False,
                verbose=False,
            )
        except Exception as exc:
            logger.error("Partial transcription error: %s", exc, exc_info=True)
            return

        # Use Whisper's detected language if we didn't have one.
        result_lang: str = str(result.get("language") or lang or "?")
        segments = result.get("segments", [])

        # Filter segments the same way commits do — drop no_speech, noise
        # tokens, and low-logprob segments so partials aren't polluted by
        # hallucinations during quiet patches.
        filtered_parts: list[str] = []
        for seg in segments:
            seg_text = str(seg.get("text", "")).strip()
            if not seg_text:
                continue
            if segment_confidence_reason(seg) is not None:
                continue
            if _is_noise_token(seg_text) or _is_repetition_loop(seg_text):
                continue
            filtered_parts.append(seg_text)

        text: str = " ".join(filtered_parts).strip()
        if not text and not segments:
            # Fall back to raw text only if there are no segments (shouldn't
            # normally happen) and the raw text passes basic checks.
            raw = str(result.get("text", "")).strip()
            if raw and not _is_noise_token(raw):
                text = raw
        if text:
            if _is_repetition_loop(text):
                logger.debug(
                    "Partial suppressed — repetition loop detected: %r", text[:60]
                )
                return
            # Content-based early commit (WhisperLive same_output_threshold).
            # If the partial text hasn't changed for CONTENT_COMMIT_STABLE_RUNS
            # consecutive inference runs, the speaker has likely stopped — signal
            # the VAD thread to commit now rather than waiting for silence.
            if text == self._last_partial_text:
                self._stuck_partial_count += 1
                if self._stuck_partial_count >= CONTENT_COMMIT_STABLE_RUNS:
                    logger.debug(
                        "Partial stable for %d runs — requesting early commit: %r",
                        self._stuck_partial_count,
                        text[:60],
                    )
                    self._commit_requested.set()
                    return  # suppress the duplicate partial emission too
            else:
                self._stuck_partial_count = 0
                self._last_partial_text = text
            logger.debug("Partial: %s", text[:80])
            # Suppress partial if its script is inconsistent with the currently
            # detected language (e.g. Korean/German/Latin text when fa is active).
            if _partial_script_mismatch(text, self._detected_lang):
                logger.debug(
                    "Partial suppressed — script mismatch (detected=%s): %r",
                    self._detected_lang,
                    text[:60],
                )
                return
            self.on_partial(result_lang, text, self._result_confidence(result_lang))
        else:
            logger.debug("Partial returned no usable segments.")

    def _compute_speaker_embed(
        self, model, audio: np.ndarray, reuse_last=False
    ) -> Optional[torch.Tensor]:
        """Return a normalised mean-pooled encoder embedding for the audio.

        After decoding, reuse its encoder features. Standalone callers can
        still request a new encoder pass for the supplied audio.

        Returns None if the encoding fails.
        """
        try:
            import whisper as _whisper

            enc = getattr(model.encoder, "last_features", None) if reuse_last else None
            if not isinstance(enc, torch.Tensor):
                audio_t = torch.from_numpy(audio)
                mel = _whisper.log_mel_spectrogram(
                    audio_t, n_mels=model.dims.n_mels, padding=_whisper.audio.N_SAMPLES
                )
                mel = (
                    _whisper.pad_or_trim(mel, _whisper.audio.N_FRAMES)
                    .to(model.device)
                    .unsqueeze(0)
                )
                with torch.no_grad():
                    enc = model.embed_audio(mel)
            enc = enc.float()

            # Mean-pool over time dimension, then L2-normalise.
            pooled = enc.mean(dim=1)  # [1, n_state]
            norm = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
            return norm[0].cpu()  # [n_state]
        except Exception as exc:
            logger.warning("Speaker embed failed: %s", exc)
            return None

    def _transcription_quality(self, result):
        weighted, total = 0.0, 0
        for segment in result.get("segments", []):
            text = str(segment.get("text", "")).strip()
            reason = segment_confidence_reason(segment)
            if not text or _is_noise_token(text) or _is_repetition_loop(text):
                continue
            if reason not in (None, "low_confidence"):
                continue
            weight = max(1, len(segment.get("tokens", [])) or len(text))
            weighted += float(segment.get("avg_logprob", 0.0)) * weight
            total += weight
        return weighted / total if total else float("-inf")

    def _commit_language_hint(self, captured_at) -> Optional[str]:
        """Language to decode the commit in, or None to auto-detect.

        Whisper's per-utterance detection is unreliable on 1–2 s VAD commits:
        Persian speech decodes as Turkish, English, or French, and the
        confidence filter then drops it. The rolling detection already knows
        the language from ~5 s of confirmed speech, so use it as a hint.
        _decode_commit falls back to auto-detection when the hint scores
        worse, so a speaker switch is still handled.
        """
        confidence = self._detected_confidence or 0.0
        if self._detected_lang and confidence >= COMMIT_LANGUAGE_HINT_MIN_CONFIDENCE:
            return self._detected_lang
        evidence = self._speechbrain_evidence(
            captured_at if captured_at is not None else time.monotonic()
        )
        if evidence and evidence.probabilities:
            top = max(evidence.probabilities, key=evidence.probabilities.get)
            if evidence.probabilities[top] >= COMMIT_LANGUAGE_HINT_MIN_CONFIDENCE:
                return top
        return None

    def _decode_commit(self, model, audio, language: Optional[str] = None):
        options = dict(
            task="transcribe",
            language=language,
            fp16=self._fp16,
            temperature=0.0,
            beam_size=1,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            initial_prompt=None,
            condition_on_previous_text=False,
            verbose=False,
        )
        started = time.monotonic()
        result = model.transcribe(audio, **options)
        elapsed = time.monotonic() - started
        quality = self._transcription_quality(result)

        # A hint from an older utterance, music, or a speaker switch decodes
        # worse than auto-detection. Compare before spending a beam retry.
        if language is not None and self._quality_retry_allowed:
            text = str(result.get("text", ""))
            if quality < COMMIT_LANGUAGE_RETRY_QUALITY or _partial_script_mismatch(
                text, language
            ):
                auto_options = dict(options, language=None)
                candidate = model.transcribe(audio, **auto_options)
                candidate_quality = self._transcription_quality(candidate)
                if candidate_quality > quality + 0.05:
                    logger.debug(
                        "Auto-detect beat the %s hint (%.2f > %.2f)",
                        language,
                        candidate_quality,
                        quality,
                    )
                    result, quality, options = candidate, candidate_quality, auto_options

        if (
            self._quality_retry_allowed
            and elapsed <= len(audio) / WHISPER_RATE * 0.35
            and len(audio) >= int(1.5 * WHISPER_RATE)
            and -1.5 <= quality < -0.65
        ):
            # Spend extra decoding work only on uncertain speech, never on
            # noise-only output or every already-confident utterance.
            options["beam_size"] = 3
            try:
                candidate = model.transcribe(audio, **options)
                improved = self._transcription_quality(candidate)
                if improved > quality + 0.05:
                    logger.debug(
                        "Beam retry improved transcript score %.3f → %.3f",
                        quality,
                        improved,
                    )
                    result = candidate
            except Exception:
                logger.warning(
                    "Quality retry failed; keeping the first decode", exc_info=True
                )
        return result

    def _infer_commit(
        self, model, audio: np.ndarray, lang: Optional[str], captured_at=None
    ) -> None:
        """Run a final Whisper pass, detect speaker change, and fire on_result.

        Speaker-change detection
        ------------------------
        After a successful transcribe() call, its encoder features are reused
        to obtain a mean-pooled embedding without another inference pass.
        Cosine similarity is computed against the embedding from the previous
        utterance.  If it falls below SPEAKER_CHANGE_THRESHOLD the
        on_speaker_change callback is fired *before* on_result so that the UI
        can insert a visual separator above the new utterance.

        Non-speech classification
        -------------------------
        If all segments are suppressed by no_speech_prob filtering (or the
        transcription returns empty text), on_non_speech is fired instead of
        on_result.
        """
        # Trim to last 30 s to keep encoder shape sane on ROCm.
        MAX_WHISPER_SAMPLES = 30 * WHISPER_RATE
        if len(audio) > MAX_WHISPER_SAMPLES:
            audio = audio[-MAX_WHISPER_SAMPLES:]

        if len(audio) < WHISPER_RATE // 2:
            return

        rms = float(np.sqrt(np.mean(audio**2)))
        if rms < 1e-4:
            logger.debug("Utterance is silence (rms=%.6f), skipping.", rms)
            if self.on_non_speech:
                self.on_non_speech("silence")
            return

        audio = _normalize_audio(audio, rms)

        # Fall back to the most recently detected language if none was
        # captured at VAD-end time (i.e. the utterance ended before the
        # async detect pass could finish).
        if lang is None:
            lang = self._detected_lang

        # The rolling detector knows the language from several seconds of
        # confirmed speech, and short granular commits (~2 s) are exactly where
        # Whisper's own detection fails: Persian speech was decoded as Turkish,
        # English, or French and then dropped as low-confidence.  The hint is
        # therefore taken from live detection, never from configuration, and
        # _decode_commit reverts to auto-detection whenever the hinted decode
        # scores worse.  A hard pin (language="en" on Persian audio) would
        # translate instead of transcribe, which this avoids.
        logger.debug("Commit: language hint=%s (auto-detect fallback on)", lang)

        # Never pass initial_prompt to commits.  With multilingual audio the
        # prompt is often in the wrong language (e.g. Arabic after a single
        # Arabic phrase) and biases Whisper's decoder away from the actual
        # speech, producing hallucinations or low-logprob garbage.  The cost
        # (slightly less inter-utterance context) is far outweighed by the
        # elimination of hallucination cascades.

        logger.debug(
            "Commit: audio shape=%s dtype=%s rms=%.4f lang=%s",
            audio.shape,
            audio.dtype,
            rms,
            lang,
        )

        try:
            t0 = time.monotonic()
            hint = self._commit_language_hint(captured_at)
            if hint:
                logger.debug("Commit: decoding with detected language %s", hint)
            source_result = self._decode_commit(model, audio, language=hint)

            detected_lang: str = str(source_result.get("language") or "?")
            self._last_transcribed_lang = detected_lang
            raw_text: str = str(source_result.get("text", ""))
            source_text: str = raw_text.strip()

            # Text can corroborate an audio decision, but fluent generated
            # text alone is not evidence that this language was actually spoken.
            if len(source_text) >= 15:
                lingua_lang = _lingua_detect(source_text)
                evidence = self._speechbrain_evidence(
                    captured_at if captured_at is not None else time.monotonic()
                )
                if lingua_lang and lingua_lang != detected_lang and evidence:
                    audio_lang = max(
                        evidence.probabilities, key=evidence.probabilities.get
                    )
                    if (
                        audio_lang == lingua_lang
                        and evidence.probabilities[audio_lang] >= 0.8
                    ):
                        logger.debug(
                            "Audio and text corroborate language %s instead of %s",
                            lingua_lang,
                            detected_lang,
                        )
                        detected_lang = lingua_lang

            segments = source_result.get("segments", [])

            # Log segment-level no_speech_prob so we can diagnose suppression.
            for seg in segments:
                logger.debug(
                    "  segment [%.1f-%.1fs] no_speech_prob=%.3f text=%r",
                    seg.get("start", 0),
                    seg.get("end", 0),
                    seg.get("no_speech_prob", -1),
                    seg.get("text", "")[:80],
                )

            if not source_text:
                logger.debug(
                    "Commit returned empty text (lang=%s, raw=%r, no_speech suppressed?).",
                    detected_lang,
                    raw_text[:40],
                )
                if self.on_non_speech:
                    self.on_non_speech("no_speech")
                return

            # _detected_lang is now managed exclusively by _infer_detect, which
            # runs on the rolling speech buffer.  Commits no longer update the
            # cache — doing so was the root cause of language flips from short,
            # quiet tail utterances being mis-labelled by Whisper.

            elapsed = time.monotonic() - t0
            logger.debug(
                "Transcribed %.1fs utterance in %.2fs — lang=%s",
                len(audio) / WHISPER_RATE,
                elapsed,
                detected_lang,
            )

            # ── Speaker-change detection ────────────────────────────────
            # Reuse the current decode's features before any later language
            # request changes the encoder cache's most recent entry.
            if self.on_speaker_change:
                current_embed = self._compute_speaker_embed(
                    model, audio, reuse_last=True
                )
                if current_embed is not None:
                    if self._last_speaker_embed is not None:
                        sim = float(
                            torch.dot(self._last_speaker_embed, current_embed).item()
                        )
                        logger.debug(
                            "Speaker similarity: %.3f (threshold=%.2f)",
                            sim,
                            SPEAKER_CHANGE_THRESHOLD,
                        )
                        if sim < SPEAKER_CHANGE_THRESHOLD:
                            logger.debug(
                                "Speaker change detected (sim=%.3f < %.2f).",
                                sim,
                                SPEAKER_CHANGE_THRESHOLD,
                            )
                            self.on_speaker_change()
                    self._last_speaker_embed = current_embed

            # Collect all passing segments and join them into a single emission
            # per commit.  Emitting each Whisper segment as its own row caused
            # sentence fragmentation (e.g. "carpet. So" / "means carpet." as
            # two separate rows).  Joining preserves natural sentence flow.
            any_emitted = False
            kept_texts: list[str] = []
            if segments:
                for seg in segments:
                    seg_text = str(seg.get("text", "")).strip()
                    if not seg_text:
                        continue
                    reason = segment_confidence_reason(seg)
                    if reason is None and _is_noise_token(seg_text):
                        reason = "noise"
                    if reason is None and _is_repetition_loop(seg_text):
                        reason = "repetition"
                    if reason:
                        logger.debug(
                            "  segment skipped — %s avg_logprob=%.3f text=%r",
                            reason,
                            seg.get("avg_logprob", 0.0),
                            seg_text[:60],
                        )
                        continue
                    logger.debug(
                        "  segment kept — avg_logprob=%.3f no_speech_prob=%.3f text=%r",
                        seg.get("avg_logprob", 0.0),
                        seg.get("no_speech_prob", 0.0),
                        seg_text[:60],
                    )
                    kept_texts.append(seg_text)

                if kept_texts:
                    joined = " ".join(kept_texts)
                    sentences = _split_sentences(joined)
                    logger.debug(
                        "  emitting %d text part(s) from %d segment(s): %r",
                        len(sentences),
                        len(kept_texts),
                        joined[:80],
                    )
                    for sent in sentences:
                        self.on_result(
                            detected_lang,
                            sent,
                            self._result_confidence(detected_lang, captured_at),
                        )
                    any_emitted = True
            else:
                if not _is_repetition_loop(source_text) and not _is_noise_token(
                    source_text
                ):
                    self.on_result(
                        detected_lang,
                        source_text,
                        self._result_confidence(detected_lang, captured_at),
                    )
                    any_emitted = True
                else:
                    logger.debug(
                        "Commit suppressed — repetition loop: %r", source_text[:60]
                    )

            if not any_emitted and self.on_non_speech:
                # All segments were filtered — classify as non-speech.
                self.on_non_speech("no_speech")
            elif any_emitted:
                # Push confirmed speech into the rolling language-detection buffer.
                self._push_speech_buffer(audio, captured_at)
                self._last_partial_text = ""
                self._stuck_partial_count = 0

        except Exception as exc:
            logger.error("Whisper inference error: %s", exc, exc_info=True)
            # Reset through the inference work channel, never mutate detection
            # state from a commit (including its error path).
            self._infer_queue.put(("reset_lang", time.monotonic()))
            if self.on_non_speech:
                self.on_non_speech("inference_error")

    # ------------------------------------------------------------------
    # VAD thread  (never touches Whisper)
    # ------------------------------------------------------------------

    def _queue_commit(self, audio, lang, boundary=False, captured_at=None):
        """Give both detectors matching audio/time and preserve pause boundaries."""
        captured_at = captured_at if captured_at is not None else time.monotonic()
        if len(audio) >= 2 * WHISPER_RATE:
            self._langid_queue.put((captured_at, audio))
        self._infer_queue.put(("commit", audio, lang, captured_at, boundary))

    def _run(self) -> None:
        """VAD loop: reads audio, posts work items to _infer_queue."""
        # Wait until the inference thread has loaded the models.
        self._ready.wait()

        _vad_model, VADIterator = self._load_vad_only()

        vad_iter = VADIterator(
            _vad_model,
            threshold=VAD_THRESHOLD,
            sampling_rate=WHISPER_RATE,
            min_silence_duration_ms=VAD_SILENCE_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        )

        # Drain audio that accumulated during model load.
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        vad_iter.reset_states()

        utterance: list[np.ndarray] = []
        utterance_samples: int = 0
        remainder = np.zeros(0, dtype=np.float32)

        _speech_active: bool = False
        _utterance_start_time: float = 0.0
        _last_interim_time: float = 0.0
        _detection_triggered: bool = False
        _speech_sample_offset: int = 0
        _silence_start_time: float = time.monotonic()
        _silence_reset_sent = True
        _last_pause_cut = None
        _last_pause_seconds = 0.0

        _vad_frame_count = 0
        _vad_prob_log_interval = 50

        while not self._stop_event.is_set():
            # Capture continues delivering zero-valued chunks during silence;
            # queue.Empty is not a reliable indication that speech has stopped.
            now = time.monotonic()
            if (
                not _speech_active
                and not _silence_reset_sent
                and now - _silence_start_time >= SILENCE_LANG_RESET_S
            ):
                self._infer_queue.put(("reset_lang", now))
                _silence_reset_sent = True
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            utterance.append(chunk)
            utterance_samples += len(chunk)

            audio_for_vad = np.concatenate([remainder, chunk])
            i = 0
            while i + VAD_CHUNK <= len(audio_for_vad):
                window = audio_for_vad[i : i + VAD_CHUNK]
                window_t = torch.from_numpy(window)
                event = vad_iter(window_t, return_seconds=False)
                i += VAD_CHUNK

                _vad_frame_count += 1
                if _vad_frame_count % _vad_prob_log_interval == 0:
                    elapsed = (
                        time.monotonic() - _utterance_start_time
                        if _speech_active
                        else 0.0
                    )
                    logger.debug(
                        "VAD state: triggered=%s, temp_end=%d, current_sample=%d, "
                        "utterance_samples=%d, speech_offset=%d, elapsed=%.1fs, event=%s",
                        vad_iter.triggered,
                        vad_iter.temp_end,
                        vad_iter.current_sample,
                        utterance_samples,
                        _speech_sample_offset,
                        elapsed,
                        event,
                    )

                if event is None:
                    pass
                elif "start" in event:
                    logger.debug("VAD start at sample %d", utterance_samples)
                    _speech_active = True
                    # Include a short pre-roll so the onset phoneme is not
                    # clipped.  We already have the audio in utterance[]; just
                    # move the speech-start pointer back by VAD_PREROLL_S.
                    preroll = int(VAD_PREROLL_S * WHISPER_RATE)
                    _speech_sample_offset = max(0, utterance_samples - preroll)
                    _utterance_start_time = time.monotonic()
                    _last_interim_time = _utterance_start_time
                    _detection_triggered = False
                    _last_pause_cut = None
                    _last_pause_seconds = 0.0

                elif "end" in event:
                    logger.debug("VAD end at sample %d", utterance_samples)
                    _speech_active = False
                    _silence_start_time = time.monotonic()
                    _silence_reset_sent = False
                    _detection_triggered = False

                    speech_chunks = _speech_chunks(utterance, _speech_sample_offset)
                    audio = (
                        np.concatenate(speech_chunks)
                        if speech_chunks
                        else np.zeros(0, dtype=np.float32)
                    )
                    captured_lang = self._vad_lang  # GIL-atomic read

                    self._queue_commit(audio, captured_lang, boundary=True)
                    logger.debug(
                        "VAD end — queued commit (lang=%s, samples=%d).",
                        captured_lang,
                        len(audio),
                    )

                    utterance = []
                    utterance_samples = 0
                    _speech_sample_offset = 0

                    # Clear so next utterance re-detects.
                    self._vad_lang = None

                    vad_iter.reset_states()
                    _last_pause_cut = None
                    _last_pause_seconds = 0.0

                if _speech_active and vad_iter.temp_end:
                    quiet_samples = vad_iter.current_sample - vad_iter.temp_end
                    if quiet_samples >= int(MIN_ROLLING_PAUSE_SECONDS * WHISPER_RATE):
                        # Map Silero's sample clock into the retained utterance.
                        processed_end = utterance_samples - (len(audio_for_vad) - i)
                        cut = processed_end - quiet_samples + int(0.06 * WHISPER_RATE)
                        if cut > _speech_sample_offset:
                            _last_pause_cut = cut
                            _last_pause_seconds = quiet_samples / WHISPER_RATE

            remainder = audio_for_vad[i:]

            # ── Kick off language detection once we have enough audio ───
            speech_samples = utterance_samples - _speech_sample_offset
            if (
                _speech_active
                and not _detection_triggered
                and speech_samples >= int(DETECT_SECONDS * WHISPER_RATE)
            ):
                speech_chunks = _speech_chunks(utterance, _speech_sample_offset)
                snapshot = np.concatenate(speech_chunks)[
                    : int(DETECT_SECONDS * WHISPER_RATE)
                ]
                captured_at = time.monotonic()
                self._infer_queue.put(("detect", snapshot, captured_at))
                self._langid_queue.put(
                    (captured_at, snapshot)
                )  # also feed SpeechBrain LangID thread
                _detection_triggered = True
                logger.debug(
                    "VAD — queued language detection (%d samples).", len(snapshot)
                )

            # ── Periodic partial result ─────────────────────────────────
            if _speech_active and utterance and self.on_partial:
                now = time.monotonic()
                if now - _last_interim_time >= INTERIM_INTERVAL_S:
                    _last_interim_time = now
                    lang = self._vad_lang  # GIL-atomic read
                    # Pass lang=None when detection hasn't finished yet —
                    # Whisper will auto-detect on the partial pass, giving
                    # the user visible text within INTERIM_INTERVAL_S of
                    # speech start instead of waiting for a confirmed language.
                    speech_chunks = _speech_chunks(utterance, _speech_sample_offset)
                    audio = np.concatenate(speech_chunks)
                    max_samples = int(MAX_PARTIAL_SECONDS * WHISPER_RATE)
                    if len(audio) > max_samples:
                        audio = audio[-max_samples:]
                    self._infer_queue.put(("partial", audio, lang))
                    logger.debug(
                        "VAD — queued partial (elapsed=%.1fs, samples=%d, lang=%s).",
                        now - _utterance_start_time,
                        len(audio),
                        lang,
                    )

            # A stable partial alone is not an acoustic boundary. Only let
            # that hint advance a cut when Silero also observed a short pause.
            content_commit = _speech_active and self._commit_requested.is_set()
            if content_commit:
                self._commit_requested.clear()
            target_seconds = self._rolling_commit_seconds
            if content_commit and _last_pause_cut is not None:
                target_seconds = min(target_seconds, speech_samples / WHISPER_RATE)

            # Prefer a word/phrase pause around the rolling deadline. If no
            # pause arrives within 600 ms, keep the bounded realtime hard cut.
            boundary = (
                rolling_audio_boundary(
                    utterance_samples,
                    _speech_sample_offset,
                    target_seconds,
                    _last_pause_cut,
                    _last_pause_seconds,
                    sample_rate=WHISPER_RATE,
                )
                if _speech_active
                else None
            )
            if boundary is not None:
                cut, utterance_end = boundary
                buffered = np.concatenate(utterance)
                audio = buffered[_speech_sample_offset:cut]
                tail = buffered[cut:]
                captured_at = time.monotonic() - len(tail) / WHISPER_RATE
                self._queue_commit(audio, self._vad_lang, utterance_end, captured_at)
                logger.debug(
                    "Rolling commit — %.1fs, pause=%s, retained=%.2fs, utterance_end=%s",
                    len(audio) / WHISPER_RATE,
                    round(_last_pause_seconds, 3) if cut != len(buffered) else None,
                    len(tail) / WHISPER_RATE,
                    utterance_end,
                )
                # Carry the suffix intact: never discard audio captured after
                # the selected pause, or repeat it in the committed prefix.
                utterance = [tail] if len(tail) else []
                utterance_samples = len(tail)
                _speech_sample_offset = 0
                _utterance_start_time = captured_at
                _last_interim_time = time.monotonic()
                _detection_triggered = False
                _last_pause_cut = None
                _last_pause_seconds = 0.0

            # ── Safety cap (wall-clock) ─────────────────────────────────
            elif (
                _speech_active
                and (time.monotonic() - _utterance_start_time) >= MAX_UTTERANCE_SECONDS
            ):
                logger.debug(
                    "Utterance reached %ds wall-clock cap, force-emitting.",
                    MAX_UTTERANCE_SECONDS,
                )
                # Do NOT reset _detection_triggered here. During continuous speech
                # cut by the wall-clock cap, the known language is still valid.
                # Resetting would cause an immediate re-detect on prosodic boundary
                # audio that may misidentify the language (e.g. fa→en flip).
                # _detection_triggered stays True until genuine VAD silence.

                speech_chunks = _speech_chunks(utterance, _speech_sample_offset)
                audio = (
                    np.concatenate(speech_chunks)
                    if speech_chunks
                    else np.zeros(0, dtype=np.float32)
                )
                captured_lang = self._vad_lang

                self._queue_commit(audio, captured_lang)
                logger.debug(
                    "Force-emit — queued commit (lang=%s, samples=%d).",
                    captured_lang,
                    len(audio),
                )

                # Reset buffers and VAD state, but keep _speech_active=True
                # because audio is still flowing — VAD never fired "end".
                # Reset vad_iter so it re-triggers cleanly on the next chunk
                # and fires a fresh "start" event.
                utterance = []
                utterance_samples = 0
                _speech_sample_offset = 0
                _last_pause_cut = None
                _last_pause_seconds = 0.0
                _utterance_start_time = time.monotonic()
                _last_interim_time = _utterance_start_time
                # Keep _vad_lang: detection result still valid for next segment.
                vad_iter.reset_states()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_vad_only(self) -> tuple:
        """Load only Silero-VAD (Whisper already loaded by inference thread)."""
        logger.info("Loading Silero-VAD for VAD thread…")
        vad_model, VADIterator = _load_silero_vad()
        logger.info("Silero-VAD ready.")
        return vad_model, VADIterator


# ---------------------------------------------------------------------------
# Module-level helper (no self — used by both threads via plain function call)
# ---------------------------------------------------------------------------


def _speech_chunks(
    utterance: list[np.ndarray], speech_sample_offset: int
) -> list[np.ndarray]:
    """Return the subset of utterance chunks that begin at or after speech_sample_offset."""
    if speech_sample_offset <= 0:
        return list(utterance)
    result: list[np.ndarray] = []
    cumulative = 0
    for chunk in utterance:
        if cumulative + len(chunk) > speech_sample_offset:
            result.append(chunk)
        cumulative += len(chunk)
    return result if result else list(utterance)


def _normalize_audio(audio: np.ndarray, rms: float) -> np.ndarray:
    """Apply bounded gain without flattening speech peaks through clipping."""
    clean = np.nan_to_num(
        np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if not len(clean):
        return clean
    if not np.isfinite(rms) or rms <= 0:
        rms = float(np.sqrt(np.mean(clean**2)))
    peak = float(np.max(np.abs(clean)))
    if rms <= 0 or peak <= 0:
        return clean
    gain = min(0.2 / rms, 8.0, 0.98 / peak)
    return (clean * gain).astype(np.float32)
