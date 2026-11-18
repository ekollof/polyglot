"""Sample-rate negotiation and reconnect backoff in AudioCapture."""

import queue
import unittest
from unittest.mock import patch

import numpy as np
import sounddevice as sd

from audio import AudioCapture, AudioDeviceLost, list_input_devices

UMC = {
    "name": "UMC404HD 192k Pro",
    "max_input_channels": 8,
    "default_samplerate": 48000.0,
    "hostapi": 0,
}


class _Stream:
    """Minimal stand-in for sd.InputStream that accepts any opener kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _rejecting_stream(accepted_rate: int):
    def opener(**kwargs):
        if kwargs["samplerate"] != accepted_rate:
            raise sd.PortAudioError(
                "Error opening InputStream: Invalid sample rate [PaErrorCode -9997]"
            )
        return _Stream(**kwargs)

    return opener


class AudioRateTests(unittest.TestCase):
    def _capture(self) -> AudioCapture:
        capture = AudioCapture(12, queue.Queue())
        # _stream_device blocks until a switch is requested; make that immediate.
        capture._switch_event.set()
        return capture

    def test_device_advertising_48000_opens_at_44100(self):
        capture = self._capture()
        with patch("audio.sd.query_devices", return_value=UMC), patch(
            "audio.sd.InputStream", side_effect=_rejecting_stream(44100)
        ):
            capture._stream_device(12)
        self.assertEqual(capture._last_good_rate, 44100)

    def test_last_good_rate_is_tried_before_the_advertised_default(self):
        capture = self._capture()
        capture._last_good_rate = 44100
        rates = capture._rate_candidates(48000)
        self.assertEqual(rates[:3], [48000, 44100, 88200])

    def test_rate_candidates_are_unique_and_positive(self):
        capture = self._capture()
        rates = capture._rate_candidates(48000)
        self.assertEqual(len(rates), len(set(rates)))
        self.assertTrue(all(rate > 0 for rate in rates))

    def test_non_rate_errors_are_not_swallowed(self):
        capture = self._capture()

        def busy(**kwargs):
            raise sd.PortAudioError(
                "Error opening InputStream: Device unavailable [PaErrorCode -9985]"
            )

        with patch("audio.sd.query_devices", return_value=UMC), patch(
            "audio.sd.InputStream", side_effect=busy
        ):
            with self.assertRaises(sd.PortAudioError):
                capture._stream_device(12)
        self.assertEqual(capture._last_good_rate, 0)

    def test_callback_resamples_the_accepted_rate(self):
        capture = AudioCapture(12, queue.Queue())
        frames_seen = []

        def opener(**kwargs):
            if kwargs["samplerate"] != 44100:
                raise sd.PortAudioError(
                    "Error opening InputStream: Invalid sample rate [PaErrorCode -9997]"
                )
            stream = _Stream(**kwargs)
            # Deliver one block, then allow the stream block to return.
            stream.kwargs["callback"](
                np.full((1024, 2), 0.5, dtype=np.float32), 1024, None, None
            )
            frames_seen.append(stream.kwargs["samplerate"])
            capture._switch_event.set()
            return stream

        with patch("audio.sd.query_devices", return_value=UMC), patch(
            "audio.sd.InputStream", side_effect=opener
        ):
            capture._stream_device(12)
        # 44.1 kHz is resampled to Whisper's 16 kHz.
        self.assertEqual(frames_seen, [44100])
        self.assertAlmostEqual(capture.queue.get_nowait().shape[0], 372, delta=4)


BRAVE = {
    "name": "Brave",
    "max_input_channels": 2,
    "default_samplerate": 48000.0,
    "hostapi": 2,
}


class NodeWatchTests(unittest.TestCase):
    def setUp(self):
        self.capture = AudioCapture(15, queue.Queue())
        self.capture._DEVICE_WATCH_S = 0.01

    def test_locate_device_finds_the_current_index(self):
        with patch("audio.sd.query_devices", return_value=BRAVE):
            self.assertEqual(self.capture._locate_device(15, "Brave"), 15)

    def test_locate_device_reports_a_moved_index(self):
        def query(index=None):
            return UMC if index == 15 else [UMC, BRAVE]

        with patch("audio.sd.query_devices", side_effect=query):
            self.assertEqual(self.capture._locate_device(15, "Brave"), 1)

    def test_locate_device_returns_none_when_the_node_is_gone(self):
        def query(index=None):
            if index is None:
                return [UMC]
            raise sd.PortAudioError("PaErrorCode -9999")

        with patch("audio.sd.query_devices", side_effect=query):
            self.assertIsNone(self.capture._locate_device(15, "Brave"))

    def test_open_stream_raises_when_the_watched_node_disappears(self):
        with patch("audio.sd.query_devices", return_value=BRAVE), patch(
            "audio.sd.InputStream", side_effect=lambda **kwargs: _Stream(**kwargs)
        ), patch.object(AudioCapture, "_locate_device", return_value=None):
            with self.assertRaises(AudioDeviceLost):
                self.capture._stream_until_switched(
                    15, "Brave", 44100, 2, watch_node=True
                )

    def test_open_stream_is_not_watched_for_non_jack_devices(self):
        with patch("audio.sd.InputStream", side_effect=lambda **kwargs: _Stream(**kwargs)):
            self.capture._switch_event.set()
            self.capture._stream_until_switched(
                12, "UMC404HD 192k Pro", 48000, 2, watch_node=False
            )

    def test_jack_devices_are_watched_and_alsa_devices_are_not(self):
        hosts = {2: {"name": "JACK Audio Connection Kit"}, 0: {"name": "ALSA"}}
        seen = []

        def stream(device_index, name, rate, channels, watch_node=False):
            seen.append(watch_node)
            self.capture._switch_event.set()

        with patch("audio.sd.query_devices", return_value=BRAVE), patch(
            "audio.sd.query_hostapis", side_effect=lambda i: hosts[i]
        ), patch.object(AudioCapture, "_stream_until_switched", side_effect=stream):
            self.capture._stream_device(15)
        alsa = dict(BRAVE, hostapi=0)
        with patch("audio.sd.query_devices", return_value=alsa), patch(
            "audio.sd.query_hostapis", side_effect=lambda i: hosts[i]
        ), patch.object(AudioCapture, "_stream_until_switched", side_effect=stream):
            self.capture._stream_device(15)
        self.assertEqual(seen, [True, False])

    def test_run_reconnects_after_the_node_disappears(self):
        calls = []

        def fake_stream(index):
            calls.append(index)
            self.capture._device_name = "Brave"
            if len(calls) == 1:
                raise AudioDeviceLost("Device 15 (Brave) disappeared")
            self.capture._stop_event.set()

        with patch.object(
            self.capture, "_stream_device", side_effect=fake_stream
        ), patch.object(self.capture, "_reconnect_by_name") as reconnect:
            self.capture._run()
        reconnect.assert_called_once_with("Brave")
        self.assertEqual(calls, [15, 15])


class DeviceIdentityTests(unittest.TestCase):
    def test_set_device_keeps_the_selected_name(self):
        capture = AudioCapture(0, queue.Queue())
        capture._last_good_rate = 44100
        capture.set_device(21, "UMC404HD 192k Pro")
        self.assertEqual(capture._device_name, "UMC404HD 192k Pro")
        self.assertEqual(capture._last_good_rate, 0)

    def test_stream_device_follows_a_moved_index(self):
        listing = [dict(BRAVE) for _ in range(22)]
        listing[12] = UMC

        def query(index=None):
            return listing if index is None else listing[index]

        capture = AudioCapture(21, queue.Queue(), device_name=UMC["name"])
        opened = []

        def stream(device_index, name, rate, channels, watch_node=False):
            opened.append((device_index, name))
            capture._switch_event.set()

        with patch("audio.sd.query_devices", side_effect=query), patch.object(
            AudioCapture, "_stream_until_switched", side_effect=stream
        ):
            capture._stream_device(21)
        self.assertEqual(opened, [(12, UMC["name"])])

    def test_stream_device_reports_a_missing_name(self):
        capture = AudioCapture(9, queue.Queue(), device_name="sysdefault")

        def query(index=None):
            return BRAVE if index is None else BRAVE

        with patch("audio.sd.query_devices", side_effect=query):
            with self.assertRaises(AudioDeviceLost):
                capture._stream_device(9)


class DeviceFilterTests(unittest.TestCase):
    HOSTAPIS = [{"name": "ALSA"}, {"name": "OSS"}, {"name": "JACK Audio Connection Kit"}]

    def _list(self, devices, hostapis=HOSTAPIS):
        with patch("audio.sd.query_devices", return_value=devices), patch(
            "audio.sd.query_hostapis", return_value=hostapis
        ), patch("audio._pulse_source_classes", return_value={}), patch(
            "audio._pulse_app_names", return_value=set()
        ):
            return list_input_devices()

    def test_raw_alsa_devices_are_not_offered(self):
        devices = [
            dict(UMC, name="sysdefault", hostapi=0),
            dict(UMC, name="vdownmix", hostapi=0),
            dict(BRAVE),
        ]
        listed = self._list(devices)
        self.assertEqual([d["name"] for d in listed], ["Brave"])

    def test_alsa_is_used_only_when_no_shared_api_exists(self):
        devices = [dict(UMC, name="sysdefault", hostapi=0)]
        listed = self._list(devices, hostapis=[{"name": "ALSA"}])
        self.assertEqual([d["name"] for d in listed], ["sysdefault"])


if __name__ == "__main__":
    unittest.main()
