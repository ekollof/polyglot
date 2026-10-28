"""Sentence boundaries across rolling Whisper windows."""

import unittest

from sentence_buffer import SentenceBuffer, completed_sentences


class SentenceBufferTests(unittest.TestCase):
    def test_rolling_ellipses_join_before_translation(self):
        buffer = SentenceBuffer()
        ready = buffer.push(
            "Voorzitter, ik had vragen. Op social media zijn het...", "nl", 0.9
        )
        self.assertEqual([item.text for item in ready], ["Voorzitter, ik had vragen."])
        self.assertEqual(buffer.push("...mensen die laten zien hoe…", "nl", 0.8), [])
        ready = buffer.push("…ze dat doen. Er is nog", "nl", 0.9)
        self.assertEqual(
            [item.text for item in ready],
            ["Op social media zijn het mensen die laten zien hoe ze dat doen."],
        )
        self.assertEqual(buffer.pending, "Er is nog")

    def test_ellipsis_is_not_a_sentence_stop_even_before_a_capital(self):
        complete, pending = completed_sentences("I thought... Then I wondered…")
        self.assertEqual(complete, [])
        self.assertEqual(pending, "I thought... Then I wondered…")

    def test_abbreviations_decimals_and_closing_quotes(self):
        buffer = SentenceBuffer()
        self.assertEqual(buffer.push("Dr.", "en"), [])
        ready = buffer.push(
            'Smith paid 3.14 euros, e.g. today. He said "yes!" A tail', "en"
        )
        self.assertEqual(
            [item.text for item in ready],
            ["Dr. Smith paid 3.14 euros, e.g. today.", 'He said "yes!"'],
        )
        self.assertEqual(buffer.pending, "A tail")

    def test_unicode_sentence_ends_and_cjk_joins(self):
        buffer = SentenceBuffer()
        buffer.push("这是第一…", "zh")
        self.assertEqual(
            [item.text for item in buffer.push("…句话。下一句！", "zh")],
            ["这是第一句话。", "下一句！"],
        )
        self.assertEqual(
            completed_sentences("این درست است؟ بعداً"), (["این درست است؟"], "بعداً")
        )

    def test_pause_and_language_change_flush_without_mixing_languages(self):
        buffer = SentenceBuffer()
        buffer.push("A final unfinished phrase", "en", 0.9)
        ready = buffer.push("Bonjour.", "fr", 0.8)
        self.assertEqual(
            [(item.language, item.text) for item in ready],
            [("en", "A final unfinished phrase"), ("fr", "Bonjour.")],
        )
        buffer.push("Another tail", "en")
        self.assertEqual(len(buffer.flush()), 1)
        self.assertEqual(buffer.flush(), [])

    def test_live_preview_never_becomes_confirmed_translation_text(self):
        buffer = SentenceBuffer()
        buffer.push("This is", "en")
        self.assertEqual(
            buffer.preview("a wrong partial", "en")[0], "This is a wrong partial"
        )
        ready = buffer.push("the final wording.", "en")
        self.assertEqual([item.text for item in ready], ["This is the final wording."])

    def test_clear_and_length_limit_do_not_keep_old_text(self):
        buffer = SentenceBuffer(max_chars=20)
        buffer.push("Old text", "en")
        buffer.clear()
        self.assertEqual(buffer.pending, "")
        ready = buffer.push("A long unpunctuated speech fragment", "en")
        self.assertEqual(len(ready), 1)
        self.assertEqual(buffer.pending, "")


if __name__ == "__main__":
    unittest.main()
