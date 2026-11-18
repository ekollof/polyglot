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
from typing import Callable, Optional

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


def _shared_hostapis() -> set[int]:
    """Host APIs that tap PipeWire's graph without claiming the hardware.

    A raw ALSA PCM (hw:, sysdefault, vdownmix, front, …) can be opened
    exclusively and take the card away from every other application.  The
    JACK host API connects a client to PipeWire's server instead, so capture
    stays passive and other apps keep playing.
    """
    return {
        index
        for index, api in enumerate(sd.query_hostapis())
        if "JACK" in str(api.get("name", ""))
    }


def list_input_devices() -> list[dict]:
    """Return usable input devices as a list of dicts with a 'group' field.

    group is one of: 'app' | 'monitor' | 'sound'

    Only PipeWire-backed (JACK) devices are listed.  Polyglot must never lock
    the audio bus: opening an ALSA PCM directly can steal the card from the
    browser or any other player.  When no shared host API exists (no PipeWire),
    the list falls back to whatever PortAudio offers.
    """
    source_classes = _pulse_source_classes()
    app_names = _pulse_app_names()
    shared = _shared_hostapis()
    if not shared:
        logger.warning(
            "No PipeWire/JACK host API found; falling back to ALSA devices, "
            "which may hold the audio device exclusively"
        )
    devices = []
    for i, info in enumerate(sd.query_devices()):
        if int(info["max_input_channels"]) < 1:
            continue
        if shared and int(info["hostapi"]) not in shared:
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


class AudioDeviceLost(RuntimeError):
    """The JACK/PipeWire node behind an open stream disappeared."""


class AudioCapture:
    """
    Captures audio from a sounddevice input device in a background thread.

    Audio is resampled to WHISPER_RATE (16 kHz) mono float32 and placed
    into `self.queue` as numpy arrays of shape (N,).

    Supports hot-switching the source via `set_device(index)`.
    """

    # Rates tried when a device rejects its advertised default (see
    # _rate_candidates for why JACK devices need this).
    _RATE_CANDIDATES = (44100, 48000, 88200, 96000, 176400, 192000)

    # How often an open JACK/PipeWire stream checks that its node still exists.
    _DEVICE_WATCH_S = 1.0

    def __init__(
        self,
        device_index: int,
        audio_queue: Optional[queue.Queue] = None,
        device_name: str = "",
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.queue: queue.Queue[np.ndarray] = audio_queue or queue.Queue(maxsize=200)
        self._device_index = device_index
        self._device_name = device_name
        self._on_status = on_status or (lambda message: None)
        self._last_good_rate: int = 0
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

    def set_device(self, device_index: int, device_name: str = "") -> None:
        """Hot-switch to a different input device.

        The name pins the selection: device indices are not stable while
        PipeWire nodes appear and disappear, so an index alone can silently
        point at a different device by the time the stream reopens.
        """
        with self._lock:
            self._device_index = device_index
            self._device_name = device_name
            self._last_good_rate = 0  # a different device may need another rate
        self._switch_event.set()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run(self) -> None:
        failures = 0
        while not self._stop_event.is_set():
            with self._lock:
                idx = self._device_index
                name = self._device_name
            self._switch_event.clear()
            try:
                self._stream_device(idx)
                failures = 0
            except AudioDeviceLost as exc:
                if self._stop_event.is_set():
                    break
                # The stream stayed open but its JACK/PipeWire node is gone, so
                # reopen it under the new index once the node comes back.
                with self._lock:
                    lost = self._device_name or name
                logger.info("%s; waiting for it to return", exc)
                if lost:
                    self._on_status(f"Audio: waiting for {lost} to return")
                    self._reconnect_by_name(lost)
                failures = 0
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                failures += 1
                with self._lock:
                    current = self._device_name or name
                # Only the first failure of a streak warns; a device that
                # stays broken must not flood the log.
                log = logger.warning if failures == 1 else logger.debug
                log("AudioCapture error on device %d (%s): %s", idx, current, exc)
                if failures == 1 and current:
                    self._on_status(f"Audio: {current} unavailable — retrying")
                # Reconnecting only makes sense when the device is really gone.
                if (
                    current
                    and not self._switch_event.is_set()
                    and not self._device_present(current)
                ):
                    self._reconnect_by_name(current)
                # Back off so a permanently failing device cannot spin.
                self._stop_event.wait(min(5.0, float(failures)))

    @staticmethod
    def _device_present(name: str) -> bool:
        try:
            return any(
                info["name"] == name and int(info["max_input_channels"]) > 0
                for info in sd.query_devices()
            )
        except Exception:
            return False

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
                        self._on_status(f"Audio: reconnected to {name}")
                        return
            except Exception as exc:
                logger.debug("query_devices error while waiting: %s", exc)
            self._stop_event.wait(1.0)

    def _rate_candidates(self, default_rate: int) -> list[int]:
        """Rates to try, advertised default and last known good first.

        PortAudio reports a JACK device's hardware rate, but the JACK graph
        runs at the server rate (PipeWire is often at 44100 while the USB
        interface advertises 48000), so the first open fails with
        PaErrorCode -9997 and the real rate has to be discovered.
        """
        ordered: list[int] = []
        for rate in (default_rate, self._last_good_rate, *self._RATE_CANDIDATES):
            if rate > 0 and rate not in ordered:
                ordered.append(rate)
        return ordered

    @staticmethod
    def _is_rate_error(exc: Exception) -> bool:
        text = str(exc)
        return "Invalid sample rate" in text or "PaErrorCode -9997" in text

    @staticmethod
    def _locate_device(device_index: int, name: str) -> Optional[int]:
        """Index of the named input device, or None when it is gone.

        JACK/PipeWire app nodes shift index whenever any other node appears or
        disappears, so a moved index is not a lost device.
        """
        try:
            info = sd.query_devices(device_index)
            if info["name"] == name and int(info["max_input_channels"]) > 0:
                return device_index
        except Exception:
            pass
        try:
            for index, info in enumerate(sd.query_devices()):
                if info["name"] == name and int(info["max_input_channels"]) > 0:
                    return index
        except Exception:
            pass
        return None

    def _stream_device(self, device_index: int) -> None:
        info = sd.query_devices(device_index)
        with self._lock:
            expected = self._device_name
        if expected and info["name"] != expected:
            # The index now names another device.  Follow the selected name
            # instead of capturing whatever moved into its old slot.
            located = self._locate_device(device_index, expected)
            if located is None:
                raise AudioDeviceLost(
                    f"Device {device_index} ({expected}) disappeared"
                )
            logger.info(
                "Device '%s' moved from index %d to %d", expected, device_index, located
            )
            device_index = located
            info = sd.query_devices(device_index)
        name = info["name"]
        native_channels = min(int(info["max_input_channels"]), 2)
        # JACK/PipeWire app nodes disappear while the app is paused; PortAudio
        # keeps such a stream open and silently delivers zeros, so watch them.
        hostapi = int(info["hostapi"])
        watch_node = "JACK" in sd.query_hostapis(hostapi)["name"]

        # Record the name so _run() can reconnect by name if the device drops.
        with self._lock:
            self._device_name = name

        last_error: Optional[Exception] = None
        for rate in self._rate_candidates(int(info["default_samplerate"])):
            try:
                self._stream_until_switched(
                    device_index, name, rate, native_channels, watch_node
                )
                self._last_good_rate = rate
                return
            except sd.PortAudioError as exc:
                if not self._is_rate_error(exc):
                    raise
                last_error = exc
                logger.debug("Device %d (%s) rejected %d Hz", device_index, name, rate)
        raise last_error or RuntimeError("no usable sample rate")

    def _stream_until_switched(
        self,
        device_index: int,
        name: str,
        native_rate: int,
        native_channels: int,
        watch_node: bool = False,
    ) -> None:
        # Compute resample ratio once.
        g = gcd(WHISPER_RATE, native_rate)
        up = WHISPER_RATE // g
        down = native_rate // g
        need_resample = native_rate != WHISPER_RATE

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
            # Log only after PortAudio accepted the stream, so a rejected
            # sample rate is never reported as an open device.
            logger.info(
                "Opened device %d (%s) — %d Hz, %d ch",
                device_index,
                name,
                native_rate,
                native_channels,
            )
            # Block until stop or device switch, checking that a JACK/PipeWire
            # node has not disappeared underneath the still-open stream.
            while not self._switch_event.wait(self._DEVICE_WATCH_S):
                if not watch_node:
                    continue
                located = self._locate_device(device_index, name)
                if located is None:
                    raise AudioDeviceLost(
                        f"Device {device_index} ({name}) disappeared"
                    )
                if located != device_index:
                    logger.debug(
                        "Device %d (%s) moved to index %d", device_index, name, located
                    )
                    device_index = located
                    with self._lock:
                        self._device_index = located
