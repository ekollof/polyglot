"""Regressions for queue order, language evidence, and hallucination filtering."""

import queue
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

from speech_quality import (
    LanguageEvidence,
    closest_language_evidence,
    coalesce_inference_work,
    fuse_language_probabilities,
    segment_confidence_reason,
)
from transcriber import WHISPER_RATE, Transcriber, _normalize_audio


def transcript(text="Hello there.", logprob=-0.1, silence=0.1, language="en"):
    return {
        "language": language,
        "text": text,
        "segments": [
            {
                "text": text,
                "avg_logprob": logprob,
                "no_speech_prob": silence,
                "compression_ratio": 1.0,
                "tokens": [1, 2, 3],
            }
        ],
    }


class SpeechQualityTests(unittest.TestCase):
    def test_speaker_embedding_reuses_current_decode_features(self):
        transcriber = self.transcriber()
        model = SimpleNamespace(
            encoder=SimpleNamespace(
                last_features=torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
            ),
            embed_audio=Mock(side_effect=RuntimeError("must not encode again")),
        )
        embedding = transcriber._compute_speaker_embed(
            model, np.ones(16000), reuse_last=True
        )
        self.assertIsNotNone(embedding)
        self.assertAlmostEqual(embedding.norm().item(), 1.0, places=5)
        model.embed_audio.assert_not_called()

    def test_stable_language_skips_expensive_detection_but_a_change_does_not(self):
        transcriber = self.transcriber()
        transcriber._detected_lang = "en"
        transcriber._detected_confidence = 0.99
        transcriber._detected_lang_set_at = 99
        transcriber._last_transcribed_lang = "en"
        transcriber._last_language_inference_at = 99
        transcriber._sb_history = (LanguageEvidence(100, {"en": 0.99, "fr": 0.01}),)
        model = SimpleNamespace(
            device="cpu",
            dims=SimpleNamespace(n_mels=80),
            detect_language=Mock(return_value=(None, [{"fr": 0.99, "en": 0.01}])),
        )
        audio = np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1
        with patch("time.monotonic", return_value=100):
            transcriber._infer_detect(model, audio, 100)
        model.detect_language.assert_not_called()
        transcriber._sb_history = (LanguageEvidence(101, {"en": 0.01, "fr": 0.99}),)
        with (
            patch("time.monotonic", return_value=101),
            patch("whisper.log_mel_spectrogram", return_value=torch.zeros(80, 3000)),
        ):
            transcriber._infer_detect(model, audio, 101)
        model.detect_language.assert_called_once()
        self.assertEqual(transcriber._detected_lang, "fr")

    def test_retry_is_skipped_when_behind_or_when_decode_uses_the_time_budget(self):
        for allowed, end in ((False, 10), (True, 11)):
            with self.subTest(allowed=allowed):
                transcriber = self.transcriber()
                transcriber._quality_retry_allowed = allowed
                model = Mock()
                model.transcribe.return_value = transcript(logprob=-0.8)
                with patch("time.monotonic", side_effect=[10, end]):
                    transcriber._decode_commit(model, np.ones(2 * WHISPER_RATE))
                model.transcribe.assert_called_once()

    def test_slow_hardware_can_skip_previews_without_dropping_commits(self):
        transcriber = self.transcriber()
        transcriber._load_models = Mock(return_value=object())
        transcriber._partials_enabled = False
        transcriber._infer_partial = Mock()
        transcriber._infer_commit = Mock()
        source = Mock()
        source.get.side_effect = [
            ("partial", np.ones(48000), None),
            ("commit", np.ones(64000), None, 100),
            ("stop",),
        ]
        source.get_nowait.side_effect = queue.Empty
        transcriber._infer_queue = source
        transcriber._inference_loop()
        transcriber._infer_partial.assert_not_called()
        transcriber._infer_commit.assert_called_once()

    def transcriber(self):
        return Transcriber(
            queue.Queue(), Mock(), on_partial=Mock(), on_non_speech=Mock()
        )

    def test_coalescing_preserves_commit_and_reset_order_and_detection(self):
        items = [
            ("partial", "old"),
            ("detect", "old"),
            ("detect", "latest"),
            ("commit", "first"),
            ("partial", "discard"),
            ("reset_lang",),
            ("detect", "next"),
            ("commit", "second"),
            ("stop",),
        ]
        self.assertEqual(
            coalesce_inference_work(items),
            [
                ("detect", "latest"),
                ("commit", "first"),
                ("reset_lang",),
                ("detect", "next"),
                ("commit", "second"),
                ("stop",),
            ],
        )

    def test_inference_loop_keeps_commits_fifo_and_detects_new_confirmed_audio(self):
        transcriber = self.transcriber()
        transcriber._load_models = Mock(return_value=object())
        seen = []

        def commit(model, audio, lang, captured_at):
            seen.append(("commit", captured_at))
            transcriber._push_speech_buffer(audio, captured_at)

        transcriber._infer_commit = commit
        transcriber._infer_detect = lambda model, audio, captured_at: seen.append(
            ("detect", captured_at)
        )
        transcriber._infer_partial = Mock()
        audio = np.ones(16000, dtype=np.float32)
        for item in [
            ("partial", audio, None),
            ("detect", audio, 1),
            ("commit", audio, None, 2),
            ("commit", audio, None, 3),
            ("stop",),
        ]:
            transcriber._infer_queue.put(item)
        transcriber._inference_loop()
        self.assertEqual(
            seen,
            [("detect", 1), ("commit", 2), ("detect", 2), ("commit", 3), ("detect", 3)],
        )
        transcriber._infer_partial.assert_not_called()

    def test_silence_reset_works_while_audio_chunks_keep_arriving(self):
        transcriber = self.transcriber()
        transcriber._ready.set()
        clock = [0.0]
        source = Mock()
        source.empty.return_value = True

        def read_chunk(**kwargs):
            clock[0] += 1
            if clock[0] >= 7:
                transcriber._stop_event.set()
            return np.zeros(512, dtype=np.float32)

        source.get.side_effect = read_chunk
        transcriber.audio_queue = source

        class VAD:
            triggered = False
            temp_end = 0
            current_sample = 0

            def reset_states(self):
                pass

            def __call__(self, *args, **kwargs):
                self.current_sample += 512
                if self.current_sample == 512:
                    self.triggered = True
                    return {"start": 0}
                if self.current_sample == 1024:
                    self.triggered = False
                    return {"end": 1024}
                return None

        transcriber._load_vad_only = Mock(return_value=(None, lambda *a, **kw: VAD()))
        with patch("time.monotonic", side_effect=lambda: clock[0]):
            transcriber._run()
        items = []
        while not transcriber._infer_queue.empty():
            items.append(transcriber._infer_queue.get_nowait())
        self.assertEqual(sum(item[0] == "reset_lang" for item in items), 1)

    def test_latest_partial_is_used_when_no_commit_is_pending(self):
        self.assertEqual(
            coalesce_inference_work(
                [("partial", "old"), ("detect", "current"), ("partial", "new")]
            ),
            [("detect", "current"), ("partial", "new")],
        )

    def test_full_utterance_reaches_both_detectors_with_the_same_time(self):
        transcriber = self.transcriber()
        audio = np.ones(2 * WHISPER_RATE, dtype=np.float32)
        with patch("time.monotonic", return_value=123):
            transcriber._queue_commit(audio, None)
        commit = transcriber._infer_queue.get_nowait()
        captured_at, signal = transcriber._langid_queue.get_nowait()
        self.assertIs(commit[1], audio)
        self.assertIs(signal, audio)
        self.assertEqual(commit[3], captured_at)

    def test_uncertain_speechbrain_does_not_dilute_strong_whisper(self):
        language, confidence, _, _ = fuse_language_probabilities(
            {"fa": 0.99, "en": 0.01}, {"fa": 0.5, "en": 0.5}
        )
        self.assertEqual(language, "fa")
        self.assertAlmostEqual(confidence, 0.99)

    def test_strong_disagreement_is_ambiguous(self):
        _, confidence, margin, agreement = fuse_language_probabilities(
            {"fa": 0.99, "en": 0.01}, {"fa": 0.01, "en": 0.99}
        )
        self.assertLess(confidence, 0.65)
        self.assertLess(margin, 0.15)
        self.assertFalse(agreement)

    def test_strong_audio_specialist_can_resolve_weak_whisper(self):
        language, confidence, _, _ = fuse_language_probabilities(
            {"fa": 0.2, "en": 0.4, "ar": 0.4}, {"fa": 0.99, "en": 0.005, "ar": 0.005}
        )
        self.assertEqual(language, "fa")
        self.assertGreater(confidence, 0.65)

    def test_unsupported_language_mass_is_not_renormalized_into_certainty(self):
        language, confidence, _, _ = fuse_language_probabilities(
            {"en": 0.9, "fr": 0.1}, {"unsupported": 0.99, "fr": 0.01}
        )
        self.assertEqual(language, "en")
        self.assertAlmostEqual(confidence, 0.9)

    def test_language_evidence_excludes_future_stale_and_previous_session_audio(self):
        history = tuple(LanguageEvidence(at, {"en": 1.0}) for at in (90, 99, 101))
        self.assertEqual(closest_language_evidence(history, 100, 0).captured_at, 99)
        self.assertIsNone(closest_language_evidence(history, 100, 100))
        self.assertIsNone(closest_language_evidence(history, 110, 0))

    def test_normalization_preserves_peak_shape_and_limits_gain(self):
        audio = np.array([0.9, -0.8] + [0.04, -0.03] * 100, dtype=np.float32)
        normalized = _normalize_audio(audio, float(np.sqrt(np.mean(audio**2))))
        self.assertLessEqual(float(np.abs(normalized).max()), 0.981)
        np.testing.assert_allclose(
            normalized / audio,
            np.full(audio.shape, normalized[0] / audio[0]),
            rtol=1e-6,
        )
        quiet = np.full(100, 1e-5, dtype=np.float32)
        np.testing.assert_allclose(_normalize_audio(quiet, 1e-5), quiet * 8)
        self.assertTrue(
            np.isfinite(
                _normalize_audio(
                    np.array([np.nan, np.inf, 0], dtype=np.float32), np.nan
                )
            ).all()
        )

    def test_recent_confirmed_speech_limits_old_language_bias(self):
        transcriber = self.transcriber()
        transcriber._push_speech_buffer(
            np.ones(20 * WHISPER_RATE, dtype=np.float32), 100
        )
        transcriber._push_speech_buffer(
            np.full(7 * WHISPER_RATE, 2, dtype=np.float32), 107
        )
        selected = transcriber._buffer_audio_for_detect(np.zeros(WHISPER_RATE))
        self.assertEqual(len(selected), 7 * WHISPER_RATE)
        self.assertTrue(np.all(selected[-7 * WHISPER_RATE :] == 2))

    def test_uncertain_switch_needs_distinct_audio_not_repeated_inference(self):
        transcriber = self.transcriber()
        audio = np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1
        model = SimpleNamespace(
            device="cpu", dims=SimpleNamespace(n_mels=80), detect_language=Mock()
        )
        model.detect_language.side_effect = [
            (None, [{"en": 0.99, "it": 0.01}]),
            (None, [{"en": 0.25, "it": 0.75}]),
            (None, [{"en": 0.25, "it": 0.75}]),
        ]
        with (
            patch("whisper.log_mel_spectrogram", return_value=torch.zeros(80, 3000)),
            patch("time.monotonic", return_value=100),
        ):
            transcriber._infer_detect(model, audio, 98)
            transcriber._infer_detect(model, audio, 99)
            self.assertEqual(transcriber._detected_lang, "en")
            transcriber._infer_detect(model, audio, 99)
            self.assertEqual(transcriber._detected_lang, "en")
            transcriber._infer_detect(model, audio, 100)
        self.assertEqual(transcriber._detected_lang, "it")
        self.assertEqual(model.detect_language.call_count, 3)

    def test_pipeline_state_reports_only_changes(self):
        states = []
        transcriber = Transcriber(queue.Queue(), Mock(), on_state=states.append)
        transcriber._set_state("speech")
        transcriber._set_state("speech")
        transcriber._set_state("committing")
        transcriber._set_state("listening")
        self.assertEqual(states, ["speech", "committing", "listening"])

    def test_reset_keeps_the_language_but_invalidates_old_audio_evidence(self):
        transcriber = self.transcriber()
        transcriber._detected_lang = "en"
        transcriber._detected_confidence = 0.95
        transcriber._vad_lang = "en"
        transcriber._sb_history = (LanguageEvidence(99, {"en": 0.99}),)
        transcriber._push_speech_buffer(np.ones(WHISPER_RATE), 99)
        transcriber._reset_language_context(100)
        # The hint survives a pause so the next short commit is not decoded
        # with a fresh, unreliable auto-detection (see _commit_language_hint).
        self.assertEqual(transcriber._detected_lang, "en")
        self.assertEqual(transcriber._detected_confidence, 0.95)
        self.assertIsNone(transcriber._vad_lang)
        self.assertIsNone(transcriber._speechbrain_evidence(101))
        self.assertFalse(transcriber._speech_buffer)

    def test_filtered_partials_cannot_resurface_as_raw_text(self):
        transcriber = self.transcriber()
        model = Mock()
        model.transcribe.return_value = transcript(logprob=-1.4)
        transcriber._infer_partial(
            model, np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1, None
        )
        transcriber.on_partial.assert_not_called()

    def test_confident_speech_is_not_discarded_just_for_high_silence_score(self):
        self.assertIsNone(
            segment_confidence_reason(
                transcript(logprob=-0.2, silence=0.9)["segments"][0]
            )
        )
        self.assertEqual(
            segment_confidence_reason(
                transcript(logprob=-1.1, silence=0.9)["segments"][0]
            ),
            "no_speech",
        )

    def test_uncertain_commit_gets_one_unpinned_deterministic_retry(self):
        transcriber = self.transcriber()
        first, better = (
            transcript(logprob=-0.8),
            transcript(text="Better words.", logprob=-0.3),
        )
        model = Mock()
        model.transcribe.side_effect = [first, better]
        self.assertIs(
            transcriber._decode_commit(model, np.ones(2 * WHISPER_RATE)), better
        )
        self.assertEqual(
            [call.kwargs["beam_size"] for call in model.transcribe.call_args_list],
            [1, 3],
        )
        for call in model.transcribe.call_args_list:
            self.assertIsNone(call.kwargs["language"])
            self.assertIsNone(call.kwargs["initial_prompt"])
            self.assertFalse(call.kwargs["condition_on_previous_text"])
            self.assertEqual(call.kwargs["temperature"], 0.0)

    def test_worse_retry_keeps_original_and_confident_speech_is_not_retried(self):
        transcriber = self.transcriber()
        first = transcript(logprob=-0.8)
        model = Mock()
        model.transcribe.side_effect = [first, transcript(logprob=-1.2)]
        self.assertIs(
            transcriber._decode_commit(model, np.ones(2 * WHISPER_RATE)), first
        )
        model.transcribe.reset_mock(side_effect=True)
        model.transcribe.return_value = transcript()
        transcriber._decode_commit(model, np.ones(2 * WHISPER_RATE))
        model.transcribe.assert_called_once()

    def test_commit_does_not_overwrite_cached_language_or_reuse_its_confidence(self):
        transcriber = self.transcriber()
        transcriber._detected_lang, transcriber._detected_confidence = "fr", 0.99
        model = Mock()
        model.transcribe.return_value = transcript()
        transcriber._infer_commit(
            model, np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1, "fr", 100
        )
        self.assertEqual(transcriber._detected_lang, "fr")
        self.assertEqual(transcriber._detected_confidence, 0.99)
        transcriber.on_result.assert_called_once_with("en", "Hello there.", None)
        self.assertEqual(transcriber._speech_buffer_secs, 2)

    def test_commit_error_schedules_reset_without_mutating_detection_state(self):
        transcriber = self.transcriber()
        transcriber._detected_lang, transcriber._detected_confidence = "fr", 0.99
        model = Mock()
        model.transcribe.side_effect = RuntimeError("decode failed")
        with self.assertLogs("transcriber", level="ERROR"):
            transcriber._infer_commit(
                model, np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1, None
            )
        self.assertEqual(transcriber._detected_lang, "fr")
        self.assertEqual(transcriber._infer_queue.get_nowait()[0], "reset_lang")
        transcriber.on_result.assert_not_called()
        transcriber.on_non_speech.assert_called_once_with("inference_error")

    def test_commit_with_a_contradicting_script_is_vetoed(self):
        transcriber = self.transcriber()
        transcriber._detected_lang = "fa"
        transcriber._detected_confidence = 0.99
        import time

        transcriber._detected_lang_set_at = time.monotonic()
        model = Mock()
        model.transcribe.return_value = transcript(
            text="This is clearly English text.", language="en"
        )
        with patch("transcriber._lingua_detect", return_value=None):
            transcriber._infer_commit(
                model,
                np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1,
                None,
                time.monotonic(),
            )
        transcriber.on_result.assert_not_called()
        transcriber.on_non_speech.assert_called_with("no_speech")

    def test_matching_script_is_not_vetoed(self):
        transcriber = self.transcriber()
        transcriber._detected_lang = "fa"
        transcriber._detected_confidence = 0.99
        import time

        transcriber._detected_lang_set_at = time.monotonic()
        model = Mock()
        model.transcribe.return_value = transcript(
            text="این یک جمله فارسی است.", language="fa"
        )
        with patch("transcriber._lingua_detect", return_value=None):
            transcriber._infer_commit(
                model,
                np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1,
                None,
                time.monotonic(),
            )
        transcriber.on_result.assert_called_once()

    def test_short_commits_are_held_and_merged(self):
        transcriber = self.transcriber()
        model = Mock()
        model.transcribe.return_value = transcript(text="Hello there.")
        short = np.ones(int(0.6 * WHISPER_RATE), dtype=np.float32) * 0.1
        transcriber._infer_commit(model, short, None, 100, boundary=True)
        model.transcribe.assert_not_called()
        self.assertIsNotNone(transcriber._held_audio)
        transcriber._infer_commit(model, short, None, 101, boundary=True)
        model.transcribe.assert_called_once()
        decoded = model.transcribe.call_args.args[0]
        self.assertAlmostEqual(len(decoded) / WHISPER_RATE, 1.2, places=1)
        self.assertIsNone(transcriber._held_audio)

    def test_forced_commit_decodes_a_held_fragment(self):
        transcriber = self.transcriber()
        model = Mock()
        model.transcribe.return_value = transcript(text="Hello there.")
        short = np.ones(int(0.6 * WHISPER_RATE), dtype=np.float32) * 0.1
        transcriber._infer_commit(model, short, None, 100, boundary=True)
        transcriber._infer_commit(model, short, None, 101, boundary=True, force=True)
        model.transcribe.assert_called_once()

    def test_text_language_override_requires_matching_audio_evidence(self):
        for evidence_time, expected in ((99, "fa"), (80, "en")):
            with self.subTest(evidence_time=evidence_time):
                transcriber = self.transcriber()
                transcriber._sb_history = (
                    LanguageEvidence(evidence_time, {"fa": 0.99, "en": 0.01}),
                )
                model = Mock()
                model.transcribe.return_value = transcript(
                    text="این یک متن کاملاً فارسی است.", language="en"
                )
                with patch("transcriber._lingua_detect", return_value="fa"):
                    transcriber._infer_commit(
                        model,
                        np.ones(2 * WHISPER_RATE, dtype=np.float32) * 0.1,
                        None,
                        100,
                    )
                self.assertEqual(transcriber.on_result.call_args.args[0], expected)
                self.assertIsNone(transcriber._detected_lang)


if __name__ == "__main__":
    unittest.main()
