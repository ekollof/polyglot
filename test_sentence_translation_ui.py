"""Exercise real Textual slots with a fake translation worker and audio source."""

import asyncio
import unittest
from unittest.mock import Mock, patch

from textual.widgets import RichLog, Select

from app import PolyglotApp


class SentenceApp(PolyglotApp):
    def _start_pipeline(self):
        self._translation_worker = Mock()


class SentenceTranslationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unpunctuated_persian_waits_for_a_real_audio_boundary(self):
        app = self.make_app()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            app._accept_transcription("fa", "این‌ها می‌گویند که", 0.99)
            app._accept_transcription("fa", "باید یک دولت برای همه باشد", 0.99)
            app._translation_worker.submit.assert_not_called()
            await asyncio.to_thread(app._on_non_speech, "utterance_end")
            app._translation_worker.submit.assert_called_once()
            job = app._translation_worker.submit.call_args.kwargs
            self.assertEqual(job["text"], "این‌ها می‌گویند که باید یک دولت برای همه باشد")
            self.assertEqual(job["from_lang"], "fa")
            self.assertTrue(job["is_final"])
            self.assertEqual(app._sentence_buffer.pending, "")

    async def test_first_wrapped_reply_shifts_the_next_unresolved_slot(self):
        app = self.make_app()
        async with app.run_test(size=(90, 35)) as pilot:
            await pilot.pause()
            app._accept_transcription(
                "en", "First sentence. Second sentence. Tail", 0.9
            )
            jobs = [
                call.kwargs for call in app._translation_worker.submit.call_args_list
            ]
            first, second = [job["row_id"] for job in jobs]
            old_position = app._trans_slots[second]
            app._update_translation_row(first, "FIRST " * 25, True, "en", 0.9)
            self.assertGreater(app._trans_slots[second], old_position)
            app._update_translation_row(second, "SECOND", True, "en", 0.9)
            app._show_live_partial("more words", "en")
            text = "\n".join(
                line.text for line in app.query_one("#translation-log", RichLog).lines
            )
            self.assertEqual(text.count("FIRST"), 25)
            self.assertIn("SECOND", text)

    async def test_speaker_boundary_flushes_tail_before_the_separator(self):
        app = self.make_app()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            app._accept_transcription("en", "An unfinished turn", 0.9)
            await asyncio.to_thread(app._on_speaker_change)
            self.assertEqual(
                app._translation_worker.submit.call_args.kwargs["text"],
                "An unfinished turn",
            )
            self.assertEqual(app._sentence_buffer.pending, "")
            app._accept_transcription("en", "A different speaker.", 0.9)
            self.assertEqual(app._translation_worker.submit.call_count, 2)
            text = "\n".join(
                line.text for line in app.query_one("#source-log", RichLog).lines
            )
            self.assertLess(text.index("An unfinished turn"), text.index("─"))

    def make_app(self):
        with (
            patch(
                "app.list_input_devices",
                return_value=[{"index": 0, "name": "Fixture input", "group": "sound"}],
            ),
            patch("app.default_device", return_value=0),
            patch("app.Translator"),
        ):
            return SentenceApp()

    async def test_only_complete_confirmed_sentences_are_submitted(self):
        app = self.make_app()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            await asyncio.to_thread(
                app._on_transcription_result, "en", "This is...", 0.9
            )
            app._translation_worker.submit.assert_not_called()
            await asyncio.to_thread(
                app._on_partial_result, "en", "a wrong partial", 0.9
            )
            app._translation_worker.submit.assert_not_called()
            await asyncio.to_thread(
                app._on_transcription_result,
                "en",
                "...a complete sentence. Another tail",
                0.9,
            )
            calls = app._translation_worker.submit.call_args_list
            self.assertEqual(
                [call.kwargs["text"] for call in calls],
                ["This is a complete sentence."],
            )
            self.assertIn(calls[0].kwargs["row_id"], app._trans_slots)
            self.assertEqual(app._sentence_buffer.pending, "Another tail")
            await asyncio.to_thread(app._on_non_speech, "silence")
            self.assertEqual(
                [
                    call.kwargs["text"]
                    for call in app._translation_worker.submit.call_args_list
                ],
                ["This is a complete sentence.", "Another tail"],
            )
            self.assertTrue(
                all(
                    call.kwargs["is_final"]
                    for call in app._translation_worker.submit.call_args_list
                )
            )

    async def test_wrapped_out_of_order_translations_preserve_slots_and_preview(self):
        app = self.make_app()
        async with app.run_test(size=(90, 35)) as pilot:
            await pilot.pause()
            app._accept_transcription(
                "en", "First sentence. Second sentence. Pending", 0.9
            )
            jobs = [
                call.kwargs for call in app._translation_worker.submit.call_args_list
            ]
            first, second = [job["row_id"] for job in jobs]
            app._update_translation_row(second, "SECOND " * 25, True, "en", 0.9)
            app._update_translation_row(first, "FIRST " * 25, True, "en", 0.9)
            app._show_live_partial("new live words", "en")
            text = "\n".join(
                line.text for line in app.query_one("#translation-log", RichLog).lines
            )
            self.assertIn("FIRST", text)
            self.assertIn("SECOND", text)
            self.assertLess(text.index("FIRST"), text.index("SECOND"))
            self.assertIn("…", text)
            self.assertEqual(app._trans_slots, {})
            app._accept_transcription("en", "continued.", 0.9)
            self.assertEqual(
                app._translation_worker.submit.call_args.kwargs["text"],
                "Pending continued.",
            )
            row = app._translation_worker.submit.call_args.kwargs["row_id"]
            app._update_translation_row(row, "THIRD", True, "en", 0.9)
            text = "\n".join(
                line.text for line in app.query_one("#translation-log", RichLog).lines
            )
            self.assertIn("FIRST", text)
            self.assertIn("SECOND", text)
            self.assertIn("THIRD", text)

    async def test_target_switch_flushes_under_old_target_and_clear_drops_tail(self):
        app = self.make_app()
        async with app.run_test(size=(140, 40)) as pilot:
            await pilot.pause()
            app._accept_transcription("nl", "Nog niet klaar", 0.9)
            app.on_select_changed(
                Select.Changed(app.query_one("#lang-select", Select), "fr")
            )
            self.assertEqual(
                app._translation_worker.submit.call_args.kwargs["to_lang"], "en"
            )
            app._accept_transcription("nl", "Nieuwe tekst.", 0.9)
            self.assertEqual(
                app._translation_worker.submit.call_args.kwargs["to_lang"], "fr"
            )
            app._accept_transcription("nl", "Discard this tail", 0.9)
            app.action_clear()
            app._flush_sentence_buffer()
            self.assertEqual(app._translation_worker.submit.call_count, 2)


if __name__ == "__main__":
    unittest.main()
