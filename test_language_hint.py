"""Per-commit language hinting: short-window auto-detect is unreliable."""

import queue
import unittest
from unittest.mock import Mock

import numpy as np

from speech_quality import LanguageEvidence
from transcriber import Transcriber


def segment(text: str, logprob: float, silent: float = 0.0) -> dict:
    return {
        "text": text,
        "avg_logprob": logprob,
        "no_speech_prob": silent,
        "compression_ratio": 1.0,
        "tokens": [1] * max(1, len(text.split())),
    }


class FakeModel:
    """Return canned results keyed by the requested language (None = auto)."""

    def __init__(self, by_language):
        self.by_language = by_language
        self.calls = []

    def transcribe(self, audio, **kwargs):
        self.calls.append(dict(kwargs))
        return self.by_language[kwargs["language"]]


class CommitLanguageHintTests(unittest.TestCase):
    def setUp(self):
        self.transcriber = Transcriber(queue.Queue(), Mock())
        self.transcriber._ready.set()

    def test_confident_detection_is_used_as_hint(self):
        self.transcriber._detected_lang = "fa"
        self.transcriber._detected_confidence = 0.97
        self.assertEqual(self.transcriber._commit_language_hint(None), "fa")

    def test_uncertain_detection_leaves_auto_detection_alone(self):
        self.transcriber._detected_lang = "fa"
        self.transcriber._detected_confidence = 0.5
        self.assertIsNone(self.transcriber._commit_language_hint(None))

    def test_recent_speechbrain_evidence_is_used_without_rolling_buffer(self):
        now = 1000.0
        self.transcriber._sb_history = (
            LanguageEvidence(now - 0.5, {"fa": 0.95, "tr": 0.05}),
        )
        self.assertIsNone(self.transcriber._detected_lang)
        self.assertEqual(self.transcriber._commit_language_hint(now), "fa")

    def test_hinted_decode_is_used_without_a_second_pass(self):
        hinted = {
            "language": "fa",
            "text": "سلام",
            "segments": [segment("سلام", -0.4)],
        }
        model = FakeModel({"fa": hinted})
        result = self.transcriber._decode_commit(
            model, np.zeros(32000, dtype=np.float32), language="fa"
        )
        self.assertIs(result, hinted)
        self.assertEqual([call["language"] for call in model.calls], ["fa"])

    def test_auto_detection_recovers_when_the_hint_scores_worse(self):
        hinted = {
            "language": "fa",
            "text": "بنم ایشلر",
            "segments": [segment("بنم ایشلر", -1.4)],
        }
        auto = {
            "language": "fa",
            "text": "سلام دنیا",
            "segments": [segment("سلام دنیا", -0.35)],
        }
        model = FakeModel({"fa": hinted, None: auto})
        result = self.transcriber._decode_commit(
            model, np.zeros(32000, dtype=np.float32), language="fa"
        )
        self.assertIs(result, auto)
        self.assertEqual([call["language"] for call in model.calls], ["fa", None])

    def test_latin_text_under_a_persian_hint_is_compared_with_auto(self):
        hinted = {
            "language": "fa",
            "text": "Merci beaucoup",
            "segments": [segment("Merci beaucoup", -0.4)],
        }
        auto = {
            "language": "fr",
            "text": "Merci beaucoup",
            "segments": [segment("Merci beaucoup", -0.3)],
        }
        model = FakeModel({"fa": hinted, None: auto})
        result = self.transcriber._decode_commit(
            model, np.zeros(32000, dtype=np.float32), language="fa"
        )
        self.assertIs(result, auto)


if __name__ == "__main__":
    unittest.main()
