"""
audio.py — sounddevice-based capture thread with hot-switchable source.

Uses sounddevice (which wraps libportaudio) instead of PyAudio.  sounddevice
does not expose Pa_Initialize/Pa_Terminate to Python, which avoids the JACK
assertion crash that PyAudio triggered on Pa_Terminate under PipeWire.

Audio is resampled to 16 kHz mono float32 and pushed into a queue for the
transcriber to consume.
"""

from __future__ import annotations

import logging
import queue
import threading
from math import gcd
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

logger = logging.getLogger(__name__)

WHISPER_RATE = 16_000  # Hz required by Whisper
CHUNK_FRAMES = 1024  # frames per callback
TARGET_CHANNELS = 1  # mono output

# ---------------------------------------------------------------------------
# Device filtering
# ---------------------------------------------------------------------------

_NEVER_DEFAULT = {"pipewire", "jack", "pulse", "default"}
_BAD_SUBSTRINGS = ("(hw:", "hw:")
_BAD_KEYWORDS_DEFAULT = ("input", "webcam", "brio", "c920", "split", " 0")


def _device_is_usable(info: dict) -> bool:
    """Return True if this device is safe to open under PipeWire."""
    if int(info["max_input_channels"]) < 1:
        return False
    name_lower = info["name"].lower()
    # Exclude raw hw: ALSA devices (claimed by PipeWire, cause conflicts)
    if any(s in name_lower for s in _BAD_SUBSTRINGS):
        return False
    return True


def list_input_devices() -> list[dict]:
    """Return usable input devices as a list of dicts."""
    devices = []
    for i, info in enumerate(sd.query_devices()):
        if not _device_is_usable(info):
            continue
        devices.append(
            {
                "index": i,
                "name": info["name"],
                "rate": int(info["default_samplerate"]),
                "channels": int(info["max_input_channels"]),
            }
        )
    return devices


def default_device(devices: list[dict]) -> int:
    """Pick the best default input device using a 4-pass priority scheme."""
    # Pass 1: "Line A" on an audio interface
    for d in devices:
        if "line a" in d["name"].lower():
            return d["index"]
    # Pass 2: "Line B"
    for d in devices:
        if "line b" in d["name"].lower():
            return d["index"]
    # Pass 3: named app capture sources (Brave, Firefox, mpv, …)
    for d in devices:
        lower = d["name"].lower()
        if lower in _NEVER_DEFAULT:
            continue
        if any(k in lower for k in _BAD_KEYWORDS_DEFAULT):
            continue
        return d["index"]
    # Pass 4: anything not broken
    for d in devices:
        if d["name"].lower() not in _NEVER_DEFAULT:
            return d["index"]
    return devices[0]["index"] if devices else 0


# ---------------------------------------------------------------------------
# AudioCapture
# ---------------------------------------------------------------------------


class AudioCapture:
    """
    Captures audio from a sounddevice input device in a background thread.

    Audio is resampled to WHISPER_RATE (16 kHz) mono float32 and placed
    into `self.queue` as numpy arrays of shape (N,).

    Supports hot-switching the source via `set_device(index)`.
    """

    def __init__(self, device_index: int, audio_queue: Optional[queue.Queue] = None):
        self.queue: queue.Queue[np.ndarray] = audio_queue or queue.Queue(maxsize=200)
        self._device_index = device_index
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._switch_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="AudioCapture"
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._stop_event.set()
        self._switch_event.set()
        if self._thread:
            try:
                self._thread.join(timeout=3)
            except (KeyboardInterrupt, SystemExit):
                pass

    def set_device(self, device_index: int) -> None:
        """Hot-switch to a different input device."""
        with self._lock:
            self._device_index = device_index
        self._switch_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                idx = self._device_index
            self._switch_event.clear()
            try:
                self._stream_device(idx)
            except Exception as exc:
                logger.warning(
                    "AudioCapture error on device %d: %s — retrying in 1 s", idx, exc
                )
                self._stop_event.wait(1.0)

    def _stream_device(self, device_index: int) -> None:
        info = sd.query_devices(device_index)
        native_rate = int(info["default_samplerate"])
        native_channels = min(int(info["max_input_channels"]), 2)

        # Compute resample ratio once.
        g = gcd(WHISPER_RATE, native_rate)
        up = WHISPER_RATE // g
        down = native_rate // g
        need_resample = native_rate != WHISPER_RATE

        logger.info(
            "Opened device %d (%s) — %d Hz, %d ch",
            device_index,
            info["name"],
            native_rate,
            native_channels,
        )

        def callback(indata: np.ndarray, frames: int, time_info, status) -> None:
            if status:
                logger.debug("sounddevice status: %s", status)
            if self._switch_event.is_set():
                raise sd.CallbackStop()

            # indata shape: (frames, channels) float32
            pcm: np.ndarray = (
                indata[:, 0].copy()
                if native_channels == 1
                else indata.mean(axis=1).astype(np.float32)
            )

            if need_resample:
                pcm = resample_poly(pcm, up, down).astype(np.float32)

            try:
                self.queue.put_nowait(pcm)
            except queue.Full:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass
                self.queue.put_nowait(pcm)

        with sd.InputStream(
            device=device_index,
            channels=native_channels,
            samplerate=native_rate,
            dtype="float32",
            blocksize=CHUNK_FRAMES,
            callback=callback,
        ):
            # Block until stop or device switch.
            self._switch_event.wait()
