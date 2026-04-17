"""
transcriber.py — VAD-gated utterance transcription with Whisper.

Architecture
------------
Audio chunks (float32 mono 16kHz) arrive from the queue continuously.
Each chunk is fed to a Silero-VAD iterator that detects speech/silence
boundaries in real time (~32ms latency per chunk).

There are exactly TWO threads:

  1. VAD thread  (_run):
       Reads audio chunks, runs Silero-VAD, tracks utterance state, and
       posts work items onto _infer_queue.  Never touches the Whisper model.

  2. Inference thread  (_inference_loop):
       The ONLY thread that calls the Whisper model.  Pulls items from
       _infer_queue one at a time.  Because it is the sole consumer there
       are NO locks — _detected_lang is a plain attribute written and read
       exclusively by this thread.

Work items posted onto _infer_queue
------------------------------------
  ("detect",  audio_np)          — run detect_language, update _detected_lang
  ("partial", audio_np, lang)    — run transcribe (greedy), fire on_partial
  ("commit",  audio_np, lang)    — drain stale partials, run transcribe, fire on_result

The queue is a standard queue.Queue which is thread-safe by itself.
No additional locking is needed.

VAD parameters
--------------
- threshold: 0.3
- min_silence_duration_ms: adaptive (300/400/600 ms based on speaker pace)
- speech_pad_ms: 100 ms
- max_utterance_seconds: 25 s

Silero-VAD
----------
Loaded from the cached torch.hub snapshot
(~/.cache/torch/hub/snakers4_silero-vad_master).  torchaudio is NOT required.
"""

from __future__ import annotations

import logging
import collections
import queue
import sys
import threading
import time
import types
from typing import Callable, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

WHISPER_RATE = 16_000  # Hz required by Whisper
VAD_CHUNK = 512  # samples per VAD call (32 ms @ 16 kHz)
MAX_UTTERANCE_SECONDS = 25  # absolute safety cap
# Maximum audio fed to a partial transcription pass.  Capped so that greedy
# inference stays fast even during very long utterances.
MAX_PARTIAL_SECONDS = 8

# For continuous speech with no VAD silence (fast speakers, monologues),
# force-commit after this many seconds so each commit contains 1–3 sentences
# rather than a wall of text.  The buffer is reset and VAD state continues
# so the next commit picks up seamlessly.
ROLLING_COMMIT_S = 7.0

# How often to fire interim (partial) results while speech is ongoing.
# turbo runs at ~3× real-time on ROCm, so 1.0 s gives it enough headroom
# to finish a partial pass before the next one is triggered.
INTERIM_INTERVAL_S = 1.0

# How many seconds of audio to feed to detect_language().
# 3.0 s gives Whisper enough audio to overcome zero-padding dominance in the
# 30 s mel spectrogram window. With only 1 s of speech + 29 s of zeros the
# silence bins swamp the signal and Whisper reliably misdetects short Farsi
# (and other non-Latin-script) clips as English. At 3 s the real phonemes
# dominate and detection accuracy is much higher (fa p=0.98 observed vs
# en p=0.83 at 1 s).
DETECT_SECONDS = 3.0

# How many seconds of continuous silence before the cached detected language
# is discarded.  This ensures that a long pause between speakers (e.g. video
# cut, topic change) doesn't carry a stale language into the next utterance.
SILENCE_LANG_RESET_S = 3.0

# ---------------------------------------------------------------------------
# Speech buffer for language detection
# ---------------------------------------------------------------------------
# After each commit that produces real speech, the audio is appended to a
# rolling deque.  Language detection runs on the concatenated buffer rather
# than on the raw 3-second VAD window, giving the detector 5–30 seconds of
# confirmed-speech audio instead of potentially noisy silence snapshots.
#
# This eliminates the main source of false language switches: a single short,
# quiet utterance at the tail of a sentence being mis-labelled by Whisper and
# poisoning _detected_lang for subsequent commits.
LANG_BUFFER_MIN_SECS: float = 5.0  # minimum speech before buffer-based detection
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

# ---------------------------------------------------------------------------
# Language ID fusion weights
# ---------------------------------------------------------------------------
# SpeechBrain VoxLingua107 runs in its own dedicated thread (LangID) and
# produces audio-domain language probabilities independently of Whisper.
# When both detectors agree, confidence is a weighted average biased toward
# SpeechBrain (purpose-built audio LID, 6.7% WER on 107 languages).
# When they disagree, the higher-confidence detector wins but is discounted.
LANGID_WEIGHT_SB: float = 0.60  # SpeechBrain share when detectors agree
LANGID_WEIGHT_WH: float = 0.40  # Whisper share when detectors agree
LANGID_DISAGREE_DISCOUNT: float = 0.80  # multiply winner confidence on disagree


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

# Segments whose avg_logprob is below this are likely music/noise
# hallucinations.  Whisper's own threshold is -1.0; music hallucinations
# typically cluster in [-0.65, -1.0] while real speech sits above -0.6.
# Log every skipped segment so the threshold can be tuned from polyglot.log.
# Set to -0.8 (was -0.65) so that foreign words embedded in an otherwise-English
# utterance are not dropped: forced language="en" makes those segments score
# poorly even when they contain real speech.
AVG_LOGPROB_THRESHOLD: float = -0.95
# Segments that fall below AVG_LOGPROB_THRESHOLD but whose text is confidently
# identified by Lingua as a *different* language than Whisper's detected_lang
# are rescued up to this looser threshold.  The rationale: when Whisper runs
# in the wrong language (e.g. en hint on fa audio), logprob is artificially
# low because the decoder is fighting against the wrong phoneme distribution.
# Lingua agrees the text is real language but just a different one.
AVG_LOGPROB_RESCUE_THRESHOLD: float = -1.5

# Seconds since the last confident language confirmation before we treat
# _detected_lang as stale and clear it.  Prevents a wrong language from
# lingering when subsequent detections keep falling below the confidence bar.
LANG_STALE_SECONDS: float = 30.0


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

_lingua_detector = None  # lazily initialised on first call


def _lingua_detect(text: str) -> Optional[str]:
    """Return ISO 639-1 language code for *text*, or None if uncertain.

    Uses Lingua's full-language detector (75 languages).  The detector is
    built lazily on first call and cached for the life of the process.
    Only returns a result when Lingua's top confidence value exceeds 0.70,
    i.e. the library is reasonably sure.
    """
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
        if top.value < 0.70:
            return None
        # iso_code_639_1 is an enum; .name is e.g. "EN", "FA", "FR", "JA"
        return top.language.iso_code_639_1.name.lower()
    except Exception as exc:
        logger.debug("Lingua detection error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Silero-VAD loader (no torchaudio required)
# ---------------------------------------------------------------------------

_SILERO_SRC = "/home/andrath/.cache/torch/hub/snakers4_silero-vad_master/src"


def _load_silero_vad():
    """Load Silero-VAD model and VADIterator without requiring torchaudio."""
    if "torchaudio" not in sys.modules:
        stub = types.ModuleType("torchaudio")
        stub.__spec__ = type(sys)("torchaudio")  # type: ignore[arg-type]
        setattr(stub, "__version__", "0.0.0")
        sys.modules["torchaudio"] = stub

    if _SILERO_SRC not in sys.path:
        sys.path.insert(0, _SILERO_SRC)

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
        PyTorch device string ("cuda", "cpu").
    """

    def __init__(
        self,
        audio_queue: queue.Queue,
        on_result: ResultCallback,
        on_partial: Optional[PartialCallback] = None,
        on_speaker_change: Optional[SpeakerChangeCallback] = None,
        on_non_speech: Optional[NonSpeechCallback] = None,
        model_name: str = "turbo",
        device: str = "cuda",
    ):
        self.audio_queue = audio_queue
        self.on_result = on_result
        self.on_partial = on_partial
        self.on_speaker_change = on_speaker_change
        self.on_non_speech = on_non_speech
        self.model_name = model_name
        self.device = device

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

        # Rolling speech buffer for language detection.
        # Each entry is a numpy audio chunk from a confirmed-speech commit.
        # Language detection runs on the concatenation of all chunks when the
        # buffer contains at least LANG_BUFFER_MIN_SECS of audio.
        # Only touched by the inference thread — no lock needed.
        self._speech_buffer: collections.deque = collections.deque()
        self._speech_buffer_secs: float = 0.0

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

    def _load_models(self):
        logger.info("Loading Whisper model '%s' on %s…", self.model_name, self.device)
        import whisper
        from whisper.model import MultiHeadAttention

        whisper_model = whisper.load_model(self.model_name, device=self.device)

        # ROCm's scaled_dot_product_attention kernel crashes with a shape
        # mismatch when Whisper's kv_cache hooks are active.  Disabling SDPA
        # forces the manual matmul path, which works correctly and is fast
        # enough (turbo: ~3× real-time).  The additional qkv_attention monkey-
        # patch is NOT applied — it was written to fix a mask-slicing bug in
        # the manual path, but benchmarking shows it makes turbo 10× slower
        # and causes empty decoder output; use_sdpa=False alone is sufficient.
        MultiHeadAttention.use_sdpa = False
        logger.info("SDPA disabled — using manual attention (ROCm workaround).")

        logger.info("Whisper model loaded.")
        return whisper_model

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
                run_opts={"device": self.device},
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

            audio: np.ndarray = item
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

        while not self._stop_event.is_set():
            try:
                item = self._infer_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            kind = item[0]

            match kind:
                case "stop":
                    break

                case "detect":
                    _, audio = item
                    self._infer_detect(whisper_model, audio)

                case "reset_lang":
                    # Silence expired — forget cached language so the next
                    # utterance re-detects from scratch without the two-pass
                    # threshold working against a stale value.
                    logger.debug(
                        "Inference: resetting _detected_lang (was %s) on silence expiry.",
                        self._detected_lang,
                    )
                    self._detected_lang = None
                    self._detected_confidence = None
                    self._last_speaker_embed = None  # speaker context also stale
                    self._initial_prompt = (
                        ""  # context no longer relevant after long silence
                    )
                    self._last_partial_text = ""
                    self._stuck_partial_count = 0
                    self._commit_requested.clear()
                    # Clear speech buffer so the next speaker starts fresh
                    # without the previous speaker's audio biasing detection.
                    self._speech_buffer.clear()
                    self._speech_buffer_secs = 0.0

                case "partial":
                    _, audio, lang = item
                    # Skip this partial only if a commit is already queued
                    # (it would be stale by the time we finish inference).
                    # Drain any intermediate partial/detect items to find out.
                    commit_pending = None
                    while True:
                        try:
                            nxt = self._infer_queue.get_nowait()
                        except queue.Empty:
                            break
                        match nxt[0]:
                            case "partial" | "detect":
                                # Discard older partials — superseded by this one
                                # or by the upcoming commit.
                                logger.debug(
                                    "Inference: discarding stale %s item.", nxt[0]
                                )
                            case _:
                                # commit or stop — put it back and stop draining
                                commit_pending = nxt
                                self._infer_queue.put(nxt)
                                break

                    if commit_pending is None:
                        # No commit waiting — run this partial.
                        self._infer_partial(whisper_model, audio, lang)
                    else:
                        logger.debug(
                            "Inference: skipping partial — commit already queued."
                        )

                case "commit":
                    _, audio, lang = item
                    # Drain all stale partial/detect items ahead of next commit.
                    while True:
                        try:
                            nxt = self._infer_queue.get_nowait()
                        except queue.Empty:
                            break
                        match nxt[0]:
                            case "partial" | "detect":
                                logger.debug(
                                    "Inference: discarding stale %s before commit.",
                                    nxt[0],
                                )
                            case _:
                                # another commit or stop — put it back
                                self._infer_queue.put(nxt)
                                break
                    self._infer_commit(whisper_model, audio, lang)

    # ------------------------------------------------------------------

    def _push_speech_buffer(self, audio: np.ndarray) -> None:
        """Append confirmed-speech audio to the rolling language-detection buffer.

        Called by _infer_commit after at least one segment is emitted.
        Evicts the oldest chunks when the buffer exceeds LANG_BUFFER_MAX_SECS.
        Inference thread only — no locking required.
        """
        secs = len(audio) / WHISPER_RATE
        self._speech_buffer.append(audio)
        self._speech_buffer_secs += secs
        while self._speech_buffer_secs > LANG_BUFFER_MAX_SECS and self._speech_buffer:
            oldest = self._speech_buffer.popleft()
            self._speech_buffer_secs -= len(oldest) / WHISPER_RATE

    def _buffer_audio_for_detect(self, fallback: np.ndarray) -> np.ndarray:
        """Return audio to use for language detection.

        If the speech buffer has at least LANG_BUFFER_MIN_SECS of confirmed
        speech, concatenate and return it (trimmed to LANG_BUFFER_MAX_SECS).
        Otherwise return the raw VAD snapshot passed as fallback.
        """
        if self._speech_buffer_secs >= LANG_BUFFER_MIN_SECS:
            combined = np.concatenate(list(self._speech_buffer))
            max_samples = int(LANG_BUFFER_MAX_SECS * WHISPER_RATE)
            if len(combined) > max_samples:
                combined = combined[-max_samples:]
            logger.debug(
                "Lang detect on speech buffer (%.1fs)", self._speech_buffer_secs
            )
            return combined
        return fallback

    def _infer_detect(self, model, audio: np.ndarray) -> None:
        """Run detect_language and update _detected_lang / _vad_lang.

        A staleness check at the top clears _detected_lang if it has not been
        confirmed by a confident detection for more than LANG_STALE_SECONDS.
        This prevents a wrong language (e.g. 'uk' detected on noisy audio)
        from persisting indefinitely when subsequent detections lack confidence.

        Confidence thresholds
        ---------------------
        >= 0.80  — accept immediately (high confidence).
        0.65–0.80 — ambiguous zone: run a second detect pass on a shifted
                    time window of the same audio clip.  Accept the new
                    language only if both passes agree (same lang, both
                    >= 0.65).  This catches genuine language switches that
                    fall just below the single-pass threshold (e.g. the
                    first 1.5 s of Farsi after a long English utterance)
                    while still rejecting noisy low-confidence flips.
        < 0.65   — reject outright; keep current language.
        """
        import time as _time

        # Use the rolling speech buffer when it has enough confirmed speech.
        # This replaces the raw 3-second VAD snapshot with 5–30 seconds of
        # actual spoken audio, making detection far more reliable.
        audio = self._buffer_audio_for_detect(audio)

        # Staleness check: if _detected_lang hasn't been confirmed by a
        # confident detection within LANG_STALE_SECONDS, clear it so the next
        # commit auto-detects from scratch instead of being forced into the
        # wrong language.
        if (
            self._detected_lang is not None
            and self._detected_lang_set_at > 0.0
            and (_time.monotonic() - self._detected_lang_set_at) > LANG_STALE_SECONDS
        ):
            logger.debug(
                "Language '%s' stale (>%.0fs without confident confirmation) — clearing.",
                self._detected_lang,
                LANG_STALE_SECONDS,
            )
            self._detected_lang = None
            self._detected_confidence = None
            self._detected_lang_set_at = 0.0
            self._vad_lang = None

        try:
            import whisper as _whisper

            def _detect_on_slice(sliced: np.ndarray) -> tuple:
                """Return (lang, confidence) for a given audio slice."""
                mel = _whisper.log_mel_spectrogram(
                    torch.from_numpy(sliced).to(model.device),
                    n_mels=model.dims.n_mels,
                )
                mel = _whisper.pad_or_trim(mel, _whisper.audio.N_FRAMES)
                mel = mel.unsqueeze(0)
                _, probs_list = model.detect_language(mel)
                probs = probs_list[0]
                top = max(probs, key=probs.get)
                return top, probs[top]

            lang, confidence = _detect_on_slice(audio)
            logger.debug("Language detection: %s (p=%.2f)", lang, confidence)

            # ── Fuse with SpeechBrain LangID result (if available) ──────────
            # _sb_lang/_sb_confidence are written by the LangID thread; reads
            # here are GIL-atomic in CPython (no lock needed).
            # SpeechBrain is purpose-built for spoken language ID (6.7% error
            # on VoxLingua107 dev set, 107 languages) so it gets a higher
            # weight when both detectors agree.
            sb_lang = self._sb_lang
            sb_conf = self._sb_confidence
            if sb_lang is not None and sb_conf is not None:
                if sb_lang == lang:
                    # Agreement: weighted average, biased toward SpeechBrain.
                    combined = (
                        LANGID_WEIGHT_WH * confidence + LANGID_WEIGHT_SB * sb_conf
                    )
                    logger.debug(
                        "LangID fusion (agree %s): wh=%.2f sb=%.2f → %.2f",
                        lang,
                        confidence,
                        sb_conf,
                        combined,
                    )
                    confidence = combined
                else:
                    # Disagreement: take the higher-confidence result but
                    # discount it to reflect the uncertainty.
                    wh_lang_orig, wh_conf_orig = lang, confidence
                    if sb_conf > confidence:
                        lang = sb_lang
                        confidence = sb_conf * LANGID_DISAGREE_DISCOUNT
                    else:
                        confidence = confidence * LANGID_DISAGREE_DISCOUNT
                    logger.debug(
                        "LangID fusion (disagree): wh=%s(%.2f) sb=%s(%.2f) → %s(%.2f)",
                        wh_lang_orig,
                        wh_conf_orig,
                        sb_lang,
                        sb_conf,
                        lang,
                        confidence,
                    )

            MIN_CONFIDENT = 0.80
            MIN_AMBIGUOUS = 0.65

            if self._detected_lang is None or lang == self._detected_lang:
                # No existing language, or same language confirmed — accept if
                # confidence meets the minimum bar.
                if confidence >= MIN_AMBIGUOUS:
                    self._detected_lang = lang
                    self._detected_confidence = confidence
                    self._detected_lang_set_at = _time.monotonic()
                    self._vad_lang = lang
                elif confidence < 0.40:
                    # Very low confidence even for the cached language means
                    # Whisper is >60% sure this is NOT the cached language.
                    # Clear the cache so the next commit auto-detects on the
                    # full audio rather than being forced into the wrong language.
                    logger.debug(
                        "Confidence %.2f very low for cached lang '%s' — "
                        "clearing cache to force auto-detect on next commit.",
                        confidence,
                        self._detected_lang,
                    )
                    self._detected_lang = None
                    self._detected_confidence = None
                    self._vad_lang = None
                else:
                    logger.debug(
                        "Language detection rejected — confidence %.2f < %.2f; keeping %s.",
                        confidence,
                        MIN_AMBIGUOUS,
                        self._detected_lang,
                    )
            elif confidence >= MIN_CONFIDENT:
                # High-confidence switch to a different language — accept.
                self._detected_lang = lang
                self._detected_confidence = confidence
                self._detected_lang_set_at = _time.monotonic()
                self._vad_lang = lang
            elif confidence >= MIN_AMBIGUOUS:
                # Ambiguous zone: the detected lang differs from current and
                # confidence is 0.65–0.80.  Run a second pass on a shifted
                # window (skip first 0.5 s) to get a second opinion.
                offset = int(0.5 * WHISPER_RATE)
                if len(audio) > offset + WHISPER_RATE // 4:
                    audio2 = audio[offset:]
                    lang2, conf2 = _detect_on_slice(audio2)
                    logger.debug(
                        "Language detection (2nd pass): %s (p=%.2f)", lang2, conf2
                    )
                    if lang2 == lang and conf2 >= MIN_AMBIGUOUS:
                        # Both passes agree on the new language — accept.
                        combined = (confidence + conf2) / 2
                        logger.debug(
                            "Language switch confirmed by 2nd pass: %s→%s "
                            "(p1=%.2f p2=%.2f avg=%.2f).",
                            self._detected_lang,
                            lang,
                            confidence,
                            conf2,
                            combined,
                        )
                        self._detected_lang = lang
                        self._detected_confidence = combined
                        self._detected_lang_set_at = _time.monotonic()
                        self._vad_lang = lang
                    else:
                        logger.debug(
                            "Language detection rejected — ambiguous and 2nd pass "
                            "disagrees (%s p=%.2f vs %s p=%.2f); keeping %s.",
                            lang,
                            confidence,
                            lang2,
                            conf2,
                            self._detected_lang,
                        )
                        # Detector saw a different language twice — reduce cached
                        # confidence so the next commit uses Whisper auto-detect
                        # on the full utterance audio instead of forcing the stale
                        # language.  Auto-detect on 5-7 s is far more reliable than
                        # the 3 s snapshot used here.
                        if self._detected_confidence is not None:
                            self._detected_confidence = min(
                                self._detected_confidence, 0.69
                            )
                            logger.debug(
                                "Confidence capped to %.2f — next commit will auto-detect.",
                                self._detected_confidence,
                            )
                else:
                    logger.debug(
                        "Language detection rejected — ambiguous (%.2f) and audio "
                        "too short for 2nd pass; keeping %s.",
                        confidence,
                        self._detected_lang,
                    )
                    if self._detected_confidence is not None:
                        self._detected_confidence = min(self._detected_confidence, 0.69)
                        logger.debug(
                            "Confidence capped to %.2f — next commit will auto-detect.",
                            self._detected_confidence,
                        )
            else:
                logger.debug(
                    "Language detection rejected — confidence %.2f < %.2f; keeping %s.",
                    confidence,
                    MIN_AMBIGUOUS,
                    self._detected_lang,
                )
                # Even a below-threshold signal for a *different* language is a
                # reason to be less certain.  Cap confidence so the commit path
                # falls back to Whisper auto-detect on the full audio.
                if self._detected_confidence is not None and confidence >= 0.45:
                    self._detected_confidence = min(self._detected_confidence, 0.69)
                    logger.debug(
                        "Confidence capped to %.2f — next commit will auto-detect.",
                        self._detected_confidence,
                    )

        except Exception as exc:
            logger.error("Language detection error: %s", exc, exc_info=True)

    def _infer_partial(self, model, audio: np.ndarray, lang: Optional[str]) -> None:
        """Run a fast greedy pass and fire on_partial."""
        if not self.on_partial:
            return
        if len(audio) < WHISPER_RATE // 2:
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
        partial_lang: Optional[str] = None
        use_initial_prompt = False

        audio = _normalize_audio(audio, rms)
        try:
            result = model.transcribe(
                audio,
                task="transcribe",
                language=partial_lang,
                fp16=self.device != "cpu",
                temperature=(0.0, 0.2),
                beam_size=1,
                no_speech_threshold=1.0,  # disable internal suppression; our own filter handles it
                logprob_threshold=-1.0,
                compression_ratio_threshold=1.8,
                initial_prompt=self._initial_prompt if use_initial_prompt else None,
                condition_on_previous_text=use_initial_prompt,
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
            if seg.get("no_speech_prob", 0.0) > 0.7:
                logger.debug(
                    "Partial segment skipped — no_speech_prob=%.3f text=%r",
                    seg.get("no_speech_prob", 0.0),
                    seg_text[:60],
                )
                continue
            if _is_noise_token(seg_text):
                continue
            if seg.get("avg_logprob", 0.0) < AVG_LOGPROB_THRESHOLD:
                continue
            filtered_parts.append(seg_text)

        text: str = " ".join(filtered_parts).strip()
        if not text:
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
            self.on_partial(result_lang, text, self._detected_confidence)
        else:
            logger.debug("Partial returned no usable segments.")

    def _compute_speaker_embed(
        self, model, audio: np.ndarray
    ) -> Optional[torch.Tensor]:
        """Return a normalised mean-pooled encoder embedding for the audio.

        Uses model.embed_audio() which runs only the encoder (no decoder),
        so it adds minimal latency on top of the transcribe() call.

        Returns None if the encoding fails.
        """
        try:
            import whisper as _whisper

            audio_t = torch.from_numpy(audio).to(model.device)
            mel = _whisper.log_mel_spectrogram(audio_t, n_mels=model.dims.n_mels)
            mel = _whisper.pad_or_trim(mel, _whisper.audio.N_FRAMES)
            mel = mel.unsqueeze(0)  # [1, n_mels, N_FRAMES]

            with torch.no_grad():
                enc = model.embed_audio(mel)  # [1, T, n_state]

            # Mean-pool over time dimension, then L2-normalise.
            pooled = enc.mean(dim=1)  # [1, n_state]
            norm = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
            return norm[0].cpu()  # [n_state]
        except Exception as exc:
            logger.warning("Speaker embed failed: %s", exc)
            return None

    def _infer_commit(self, model, audio: np.ndarray, lang: Optional[str]) -> None:
        """Run a final Whisper pass, detect speaker change, and fire on_result.

        Speaker-change detection
        ------------------------
        After a successful transcribe() call, the encoder is run a second time
        on the same (normalised) audio to obtain a mean-pooled embedding.
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

        # If we still have no language (very short first utterance before any
        # detection has ever run), do a quick inline detect now so that
        # transcribe() gets a language hint rather than having to guess.
        if lang is None:
            try:
                import whisper as _whisper

                detect_audio = audio[: int(DETECT_SECONDS * WHISPER_RATE)]
                mel = _whisper.log_mel_spectrogram(
                    torch.from_numpy(detect_audio).to(model.device),
                    n_mels=model.dims.n_mels,
                )
                mel = _whisper.pad_or_trim(mel, _whisper.audio.N_FRAMES)
                mel = mel.unsqueeze(0)
                _, probs_list = model.detect_language(mel)
                probs = probs_list[0]
                lang = max(probs, key=probs.get)
                confidence = probs[lang]
                self._detected_lang = lang
                self._detected_confidence = confidence
                self._vad_lang = lang
                logger.debug("Commit inline detect: %s (p=%.2f)", lang, confidence)
            except Exception as exc:
                logger.warning("Inline detect failed: %s", exc)

        # Never force the language on commits.  With task="transcribe" and
        # language=None, Whisper auto-detects the language per commit window
        # and transcribes in that language — it does NOT translate.  Forcing
        # language="en" on audio that contains Arabic/Farsi words causes
        # Whisper to either translate semantically (fluent English output that
        # is actually a translation) or produce low-logprob phonetic garbage
        # that our filter then drops.  Both outcomes are worse than letting
        # Whisper pick the language itself from the actual audio.
        commit_lang: Optional[str] = None
        logger.debug("Commit: using Whisper auto-detect (lang hint=%s)", lang)

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
            source_result = model.transcribe(
                audio,
                task="transcribe",
                language=commit_lang,
                fp16=self.device != "cpu",
                temperature=(0.0, 0.2),
                beam_size=1,
                no_speech_threshold=1.0,  # disable internal suppression; our own filter handles it
                logprob_threshold=-1.0,
                compression_ratio_threshold=1.8,
                initial_prompt=None,
                condition_on_previous_text=False,
                verbose=False,
            )

            detected_lang: str = str(source_result.get("language") or "?")
            raw_text: str = str(source_result.get("text", ""))
            source_text: str = raw_text.strip()

            # ── Lingua text-based language cross-check ───────────────────
            # Lingua's rule engine recognises script characters immediately
            # (e.g. Arabic/Persian script → 'fa'/'ar' at confidence 1.0).
            # This catches the edge case where Whisper correctly transcribes
            # the script but still labels the segment with the wrong language
            # tag.  Only override when text is long enough to be reliable and
            # Lingua is confident (>= 0.70 — enforced inside _lingua_detect).
            if len(source_text) >= 15:
                lingua_lang = _lingua_detect(source_text)
                if lingua_lang is not None and lingua_lang != detected_lang:
                    logger.info(
                        "Lingua overrides Whisper lang %s → %s for text %r",
                        detected_lang,
                        lingua_lang,
                        source_text[:50],
                    )
                    detected_lang = lingua_lang
                elif lingua_lang:
                    logger.debug(
                        "Lingua confirms lang=%s for %r", lingua_lang, source_text[:30]
                    )

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
            # Compute encoder embedding for this utterance and compare to
            # the previous one.  We do this AFTER transcribe() so the
            # encoder result can be reused (embed_audio is a separate call
            # but re-uses the already-loaded model weights on device).
            if self.on_speaker_change:
                current_embed = self._compute_speaker_embed(model, audio)
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

            # Emit each segment as a separate result so the UI shows one
            # sentence per row instead of a wall of text.  Fall back to the
            # full text as a single result if there are no segments (shouldn't
            # happen with Whisper, but be defensive).
            any_emitted = False
            if segments:
                for seg in segments:
                    seg_text = str(seg.get("text", "")).strip()
                    if not seg_text:
                        continue
                    # Skip segments Whisper flagged as likely silence/noise.
                    if seg.get("no_speech_prob", 0) > 0.7:
                        logger.debug(
                            "  segment skipped — no_speech_prob=%.3f text=%r",
                            seg.get("no_speech_prob", 0),
                            seg_text[:60],
                        )
                        continue
                    # Skip segments that are pure music/noise bracket tokens
                    # (e.g. "[Music]", "♪ … ♪", "[Applause]").
                    if _is_noise_token(seg_text):
                        logger.debug(
                            "  segment skipped — noise token: %r", seg_text[:60]
                        )
                        continue
                    # Skip low-confidence segments likely produced by music or
                    # background noise.  avg_logprob below AVG_LOGPROB_THRESHOLD
                    # (-0.65) indicates the decoder was uncertain — hallucinated
                    # speech during music falls in this range.
                    seg_logprob = seg.get("avg_logprob", 0.0)
                    if seg_logprob < AVG_LOGPROB_THRESHOLD:
                        # Before discarding, try a per-segment Lingua rescue.
                        # If Lingua confidently identifies the segment text as a
                        # *different* language from Whisper's detected_lang, the
                        # low logprob is an artefact of wrong-language mode —
                        # rescue it up to a looser threshold.
                        rescued = False
                        if (
                            len(seg_text) >= 8
                            and seg_logprob >= AVG_LOGPROB_RESCUE_THRESHOLD
                        ):
                            seg_lingua = _lingua_detect(seg_text)
                            if seg_lingua is not None and seg_lingua != detected_lang:
                                logger.debug(
                                    "  segment rescued — Lingua=%s≠Whisper=%s avg_logprob=%.3f text=%r",
                                    seg_lingua,
                                    detected_lang,
                                    seg_logprob,
                                    seg_text[:60],
                                )
                                rescued = True
                        if not rescued:
                            logger.debug(
                                "  segment skipped — avg_logprob=%.3f (<%.2f) text=%r",
                                seg_logprob,
                                AVG_LOGPROB_THRESHOLD,
                                seg_text[:60],
                            )
                            continue
                    # Skip hallucination repetition loops.
                    if _is_repetition_loop(seg_text):
                        logger.debug(
                            "  segment skipped — repetition loop: %r", seg_text[:60]
                        )
                        continue
                    logger.debug(
                        "  segment emitted — avg_logprob=%.3f no_speech_prob=%.3f text=%r",
                        seg.get("avg_logprob", 0.0),
                        seg.get("no_speech_prob", 0.0),
                        seg_text[:60],
                    )
                    self.on_result(detected_lang, seg_text, self._detected_confidence)
                    any_emitted = True
            else:
                if not _is_repetition_loop(source_text):
                    self.on_result(
                        detected_lang, source_text, self._detected_confidence
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
                self._push_speech_buffer(audio)
                self._last_partial_text = ""
                self._stuck_partial_count = 0

        except Exception as exc:
            logger.error("Whisper inference error: %s", exc, exc_info=True)
            # Reset so the next utterance re-detects from scratch.
            self._detected_lang = None
            self._detected_confidence = None
            self._vad_lang = None
            self.on_result("?", "", None)

    # ------------------------------------------------------------------
    # VAD thread  (never touches Whisper)
    # ------------------------------------------------------------------

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

        _vad_frame_count = 0
        _vad_prob_log_interval = 50

        while not self._stop_event.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                # During silence: expire stale language cache.
                if not _speech_active:
                    now = time.monotonic()
                    if now - _silence_start_time >= SILENCE_LANG_RESET_S:
                        if self._vad_lang is not None:
                            logger.debug(
                                "Silence %.1fs — clearing cached lang '%s'.",
                                now - _silence_start_time,
                                self._vad_lang,
                            )
                            self._vad_lang = None
                            # Tell the inference thread to forget its cached
                            # language too, so the next utterance re-detects
                            # without the two-pass threshold fighting a stale
                            # value.  This is the only cross-thread write to
                            # _detected_lang — done via the queue, not directly.
                            self._infer_queue.put(("reset_lang",))
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

                elif "end" in event:
                    logger.debug("VAD end at sample %d", utterance_samples)
                    _speech_active = False
                    _silence_start_time = time.monotonic()
                    _detection_triggered = False

                    speech_chunks = _speech_chunks(utterance, _speech_sample_offset)
                    audio = (
                        np.concatenate(speech_chunks)
                        if speech_chunks
                        else np.zeros(0, dtype=np.float32)
                    )
                    captured_lang = self._vad_lang  # GIL-atomic read

                    self._infer_queue.put(("commit", audio, captured_lang))
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
                self._infer_queue.put(("detect", snapshot))
                self._langid_queue.put(snapshot)  # also feed SpeechBrain LangID thread
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

            # ── Content-based early commit ───────────────────────────────────
            # Inference thread signals this when the partial transcript has
            # been identical for CONTENT_COMMIT_STABLE_RUNS consecutive runs,
            # meaning the speaker has stopped changing their words.
            if _speech_active and self._commit_requested.is_set():
                self._commit_requested.clear()
                speech_chunks = _speech_chunks(utterance, _speech_sample_offset)
                if speech_chunks:
                    audio = np.concatenate(speech_chunks)
                    captured_lang = self._vad_lang
                    self._infer_queue.put(("commit", audio, captured_lang))
                    logger.debug(
                        "Content commit — stable partial triggered early commit, lang=%s, samples=%d.",
                        captured_lang,
                        len(audio),
                    )
                # Reset buffer same as rolling commit; keep VAD state running.
                utterance = []
                utterance_samples = 0
                _speech_sample_offset = 0
                _utterance_start_time = time.monotonic()
                _last_interim_time = _utterance_start_time

            # ── Rolling commit (continuous speech) ──────────────────────
            # If speech has been active for ROLLING_COMMIT_S without a VAD
            # silence end event (fast speaker / monologue), commit what we
            # have and restart the buffer.  This keeps each commit to ~1–3
            # sentences rather than a wall of text at the 25 s cap.
            elapsed_speech = time.monotonic() - _utterance_start_time
            if _speech_active and elapsed_speech >= ROLLING_COMMIT_S:
                speech_chunks = _speech_chunks(utterance, _speech_sample_offset)
                if speech_chunks:
                    audio = np.concatenate(speech_chunks)
                    captured_lang = self._vad_lang
                    self._infer_queue.put(("commit", audio, captured_lang))
                    logger.debug(
                        "Rolling commit — %.1fs elapsed, lang=%s, samples=%d.",
                        elapsed_speech,
                        captured_lang,
                        len(audio),
                    )

                # Reset buffer; keep _speech_active=True and VAD state running.
                # Reset _detection_triggered so the next rolling window re-detects
                # after DETECT_SECONDS of speech.  This is essential for language
                # switching during continuous speech (e.g. Arabic → English) — the
                # detection from the first window cannot be assumed valid for all
                # subsequent rolling windows.
                utterance = []
                utterance_samples = 0
                _speech_sample_offset = 0
                _detection_triggered = False
                _utterance_start_time = time.monotonic()
                _last_interim_time = _utterance_start_time
                # Keep _vad_lang: detection result still valid until overridden.

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

                self._infer_queue.put(("commit", audio, captured_lang))
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
    """Scale audio so its RMS is ~0.2, clamped to [-1, 1].

    Whisper's no-speech classifier and decoder both perform best when speech
    is at a moderate amplitude (roughly rms=0.1–0.3).  Audio captured via a
    line input at low gain (rms~0.04) is quiet enough that the encoder output
    distribution shifts and the decoder produces empty results.  A simple RMS
    normalisation to a fixed target level fixes this without distortion for
    speech-range signals.
    """
    TARGET_RMS = 0.2
    if rms <= 0:
        return audio
    return np.clip(audio * (TARGET_RMS / rms), -1.0, 1.0).astype(np.float32)
