"""Pause-aware streaming cuts must preserve audio and callback ordering."""

import queue
import unittest
from unittest.mock import Mock, patch

import numpy as np

from speech_quality import rolling_audio_boundary
from transcriber import WHISPER_RATE, Transcriber


class AudioBoundaryTests(unittest.TestCase):
    def test_recent_short_pause_is_preferred_over_a_clock_cut(self):
        self.assertEqual(
            rolling_audio_boundary(4000, 0, 4, 3700, 0.2, 1000), (3700, False)
        )
        self.assertEqual(
            rolling_audio_boundary(4000, 0, 4, 3700, 0.4, 1000), (3700, True)
        )
        self.assertIsNone(rolling_audio_boundary(3900, 0, 4, 3700, 0.4, 1000))

    def test_no_pause_has_a_bounded_wait_and_old_pauses_are_not_reused(self):
        self.assertIsNone(rolling_audio_boundary(4400, 0, 4, None, 0, 1000))
        self.assertEqual(
            rolling_audio_boundary(4600, 0, 4, None, 0, 1000), (4600, False)
        )
        self.assertIsNone(rolling_audio_boundary(4000, 0, 4, 2000, 0.4, 1000))
        self.assertIsNone(rolling_audio_boundary(4000, 0, 4, 3700, 0.05, 1000))

    def test_speech_offset_is_respected(self):
        self.assertIsNone(rolling_audio_boundary(4400, 500, 4, 4000, 0.4, 1000))
        self.assertEqual(
            rolling_audio_boundary(4500, 500, 4, 4100, 0.4, 1000), (4100, True)
        )

    def test_vad_carries_suffix_without_losing_or_repeating_samples(self):
        for with_pause in (True, False):
            with self.subTest(with_pause=with_pause):
                transcriber = Transcriber(queue.Queue(), Mock())
                transcriber._ready.set()
                source = Mock()
                source.empty.return_value = True
                delivered = [0]

                def read_chunk(**kwargs):
                    start = delivered[0]
                    delivered[0] += (
                        342  # Deliberately misaligned to Silero's 512 samples.
                    )
                    return np.arange(start, delivered[0], dtype=np.float32)

                source.get.side_effect = read_chunk
                transcriber.audio_queue = source

                class VAD:
                    triggered = False
                    temp_end = 0
                    current_sample = 0

                    def reset_states(self):
                        self.triggered = False
                        self.temp_end = 0
                        self.current_sample = 0

                    def __call__(self, *args, **kwargs):
                        self.current_sample += 512
                        frame = self.current_sample // 512
                        if frame == 1:
                            self.triggered = True
                            return {"start": 0}
                        if with_pause and 115 <= frame <= 128:
                            self.temp_end = 114 * 512
                        else:
                            self.temp_end = 0
                        if frame == 220:
                            transcriber._stop_event.set()
                            self.triggered = False
                            return {"end": self.current_sample}
                        return None

                transcriber._load_vad_only = Mock(
                    return_value=(None, lambda *args, **kwargs: VAD())
                )
                with patch(
                    "time.monotonic", side_effect=lambda: delivered[0] / WHISPER_RATE
                ):
                    transcriber._run()
                events = []
                while not transcriber._infer_queue.empty():
                    events.append(transcriber._infer_queue.get_nowait())
                commits = [item for item in events if item[0] == "commit"]
                self.assertEqual(len(commits), 2)
                np.testing.assert_array_equal(
                    np.concatenate([item[1] for item in commits]),
                    np.arange(delivered[0], dtype=np.float32),
                )
                self.assertTrue(commits[-1][4])
                if with_pause:
                    self.assertTrue(commits[0][4])
                    self.assertLess(len(commits[0][1]), 4 * WHISPER_RATE)
                    self.assertAlmostEqual(
                        commits[0][3], len(commits[0][1]) / WHISPER_RATE
                    )
                else:
                    self.assertFalse(commits[0][4])
                    self.assertGreaterEqual(len(commits[0][1]) / WHISPER_RATE, 4.6)
                    self.assertLess(len(commits[0][1]) / WHISPER_RATE, 4.63)

    def test_pause_callback_follows_its_transcript_not_the_next_chunk(self):
        seen = []
        transcriber = Transcriber(
            queue.Queue(), Mock(), on_non_speech=lambda reason: seen.append(reason)
        )
        transcriber._load_models = Mock(return_value=object())
        transcriber._infer_commit = lambda *args: seen.append("transcript")
        audio = np.ones(32000, dtype=np.float32)
        for item in [
            ("commit", audio, None, 100, True),
            ("commit", audio, None, 101, False),
            ("stop",),
        ]:
            transcriber._infer_queue.put(item)
        transcriber._inference_loop()
        self.assertEqual(seen, ["transcript", "utterance_end", "transcript"])


if __name__ == "__main__":
    unittest.main()
