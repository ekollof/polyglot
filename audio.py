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
import pulsectl
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

_VIRTUAL_NAMES = {"pipewire", "jack", "pulse", "default"}


def _pulse_source_classes() -> dict:
    """Return {lowercased_description: 'monitor'|'sound'} for all PulseAudio sources.

    Monitor sources are also keyed by their stripped name (without the
    "Monitor of " prefix) since sounddevice exposes them under that shorter name.
    """
    result: dict = {}
    try:
        with pulsectl.Pulse("polyglot-probe") as pulse:
            for src in pulse.source_list():
                desc = src.description or ""
                cls = src.proplist.get("device.class", "sound")
                result[desc.lower()] = cls
                if desc.lower().startswith("monitor of "):
                    stripped = desc[len("Monitor of ") :].lower()
                    result[stripped] = "monitor"
    except Exception as exc:
        logger.warning("pulsectl source_list failed: %s", exc)
    logger.debug("pulse source classes: %s", result)
    return result


def _pulse_app_names() -> set:
    """Return a set of lowercased application names currently playing audio.

    Uses sink_input_list() which lists active audio streams from apps
    (Brave, Firefox, mpv, etc.) via the PulseAudio compatibility layer.
    """
    names: set = set()
    try:
        with pulsectl.Pulse("polyglot-probe") as pulse:
            for si in pulse.sink_input_list():
                app = si.proplist.get("application.name") or si.proplist.get(
                    "node.name", ""
                )
                if app:
                    names.add(app.lower())
    except Exception as exc:
        logger.warning("pulsectl sink_input_list failed: %s", exc)
    logger.debug("pulse app names: %s", names)
    return names


def _classify_device(name: str, source_classes: dict, app_names: set) -> str:
    """Return 'app' | 'monitor' | 'sound' for a sounddevice input name."""
    key = name.lower()
    cls = source_classes.get(key)
    if cls == "monitor":
        return "monitor"
    if cls == "sound":
        return "sound"
    # Only confirmed active sink inputs are real app streams.
    # Everything else (Split nodes, numbered virtual nodes, etc.) → hardware input.
    if key in app_names:
        return "app"
    return "sound"


def list_input_devices() -> list[dict]:
    """Return usable input devices as a list of dicts with a 'group' field.

    group is one of: 'app' | 'monitor' | 'sound'
    """
    source_classes = _pulse_source_classes()
    app_names = _pulse_app_names()
    devices = []
    for i, info in enumerate(sd.query_devices()):
        if int(info["max_input_channels"]) < 1:
            continue
        name = info["name"]
        name_lower = name.lower()
        # Exclude raw hw: ALSA devices
        if any(s in name_lower for s in _BAD_SUBSTRINGS):
            continue
        # Exclude generic virtual sinks
        if name_lower in _VIRTUAL_NAMES:
            continue
        group = _classify_device(name, source_classes, app_names)
        devices.append(
            {
                "index": i,
                "name": name,
                "rate": int(info["default_samplerate"]),
                "channels": int(info["max_input_channels"]),
                "group": group,
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
        self._device_name: str = ""
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
            except KeyboardInterrupt, SystemExit:
                pass

    def set_device(self, device_index: int) -> None:
        """Hot-switch to a different input device."""
        with self._lock:
            self._device_index = device_index
            self._device_name = ""  # will be resolved in _stream_device
        self._switch_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                idx = self._device_index
                name = self._device_name
            self._switch_event.clear()
            try:
                self._stream_device(idx)
            except Exception as exc:
                logger.warning(
                    "AudioCapture error on device %d (%s): %s", idx, name, exc
                )
                # If the device has a known name (e.g. an app stream that
                # de-registered), poll until it reappears under any index.
                # For a user-requested switch (_switch_event already set) or
                # a permanent device, fall back to a simple 1 s delay.
                if name and not self._switch_event.is_set():
                    self._reconnect_by_name(name)
                else:
                    self._stop_event.wait(1.0)

    def _reconnect_by_name(self, name: str) -> None:
        """Block until a device with *name* reappears, then update _device_index."""
        logger.info("Waiting for device '%s' to reappear…", name)
        while not self._stop_event.is_set() and not self._switch_event.is_set():
            try:
                for i, info in enumerate(sd.query_devices()):
                    if info["name"] == name and int(info["max_input_channels"]) > 0:
                        with self._lock:
                            self._device_index = i
                        logger.info(
                            "Device '%s' reappeared at index %d — reconnecting",
                            name,
                            i,
                        )
                        return
            except Exception as exc:
                logger.debug("query_devices error while waiting: %s", exc)
            self._stop_event.wait(1.0)

    def _stream_device(self, device_index: int) -> None:
        info = sd.query_devices(device_index)
        native_rate = int(info["default_samplerate"])
        native_channels = min(int(info["max_input_channels"]), 2)

        # Record the name so _run() can reconnect by name if the device drops.
        with self._lock:
            self._device_name = info["name"]

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
