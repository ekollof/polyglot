"""OpenAI-compatible and NLLB engine wiring."""

import threading
import unittest
from unittest.mock import Mock, patch

from translator import (
    ENGINE_ARGOS,
    ENGINE_GOOGLE,
    ENGINE_NLLB,
    ENGINE_OLLAMA,
    ENGINE_OPENAI,
    Translator,
    _NLLB_CODES,
    _clean_llm_output,
    _llm_prompt,
)


def make_translator(**kwargs) -> Translator:
    return Translator(on_status=Mock(), **kwargs)


class OpenAIEngineTests(unittest.TestCase):
    def test_chat_completion_request_and_parse(self):
        tr = make_translator(
            engine=ENGINE_OPENAI,
            openai_url="http://box:1234/v1",
            openai_model="qwen",
            openai_api_key="secret",
        )
        response = Mock()
        response.json.return_value = {
            "choices": [{"message": {"content": '  "سلام"  '}}]
        }
        with patch("requests.post", return_value=response) as post:
            result = tr.translate("hello", "en", "fa")
        self.assertEqual(result, "سلام")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://box:1234/v1/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "qwen")
        self.assertEqual(kwargs["json"]["temperature"], 0)
        self.assertEqual(kwargs["headers"], {"Authorization": "Bearer secret"})
        self.assertIn("English", kwargs["json"]["messages"][1]["content"])

    def test_no_api_key_sends_no_authorization_header(self):
        tr = make_translator(
            engine=ENGINE_OPENAI,
            openai_url="http://localhost:11434/v1",
            openai_api_key="",
        )
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": "Hallo"}}]}
        with patch("requests.post", return_value=response) as post:
            self.assertEqual(tr.translate("hello", "en", "de"), "Hallo")
        self.assertIsNone(post.call_args.kwargs["headers"])

    def test_endpoint_failure_falls_back_to_argos(self):
        tr = make_translator(engine=ENGINE_OPENAI)
        with patch("requests.post", side_effect=OSError("no route")), patch.object(
            tr, "_argo_translate", return_value="offline"
        ) as argo:
            self.assertEqual(tr.translate("hello", "en", "fa"), "offline")
        argo.assert_called_once_with("hello", "en", "fa")


class NLLBTests(unittest.TestCase):
    def test_persian_and_english_codes(self):
        self.assertEqual(_NLLB_CODES["fa"], "pes_Arab")
        self.assertEqual(_NLLB_CODES["en"], "eng_Latn")
        self.assertEqual(_NLLB_CODES["zh"], "zho_Hans")

    def test_generation_uses_the_mapped_codes(self):
        tr = make_translator(engine=ENGINE_NLLB)
        with patch.object(tr, "_nllb_generate", return_value="ترجمه") as generate:
            self.assertEqual(tr.translate("hello", "en", "fa"), "ترجمه")
        generate.assert_called_once_with("hello", "eng_Latn", "pes_Arab")

    def test_unsupported_language_falls_back_to_argos(self):
        tr = make_translator(engine=ENGINE_NLLB)
        with patch.object(tr, "_argo_translate", return_value="offline") as argo:
            self.assertEqual(tr.translate("hello", "xx", "fa"), "offline")
        argo.assert_called_once()

    def test_generation_error_falls_back_to_argos(self):
        tr = make_translator(engine=ENGINE_NLLB)
        with patch.object(
            tr, "_nllb_generate", side_effect=RuntimeError("out of memory")
        ), patch.object(tr, "_argo_translate", return_value="offline"):
            self.assertEqual(tr.translate("hello", "en", "fa"), "offline")

    def test_model_name_is_configurable(self):
        tr = make_translator(engine=ENGINE_NLLB, nllb_model="facebook/nllb-200-1.3B")
        self.assertEqual(tr._nllb_model_name, "facebook/nllb-200-1.3B")


class ArgosDetectionTests(unittest.TestCase):
    def test_installed_package_is_detected_via_to_lang(self):
        from types import SimpleNamespace

        language = SimpleNamespace(
            code="fa",
            translations_to=[SimpleNamespace(to_lang=SimpleNamespace(code="en"))],
        )
        tr = make_translator(engine=ENGINE_ARGOS)
        with patch(
            "argostranslate.translate.get_installed_languages",
            return_value=[language],
        ):
            self.assertTrue(tr._argo_is_installed("fa", "en"))
            self.assertFalse(tr._argo_is_installed("fa", "de"))


class OllamaEngineTests(unittest.TestCase):
    def test_generate_request_is_deterministic(self):
        tr = make_translator(
            engine=ENGINE_OLLAMA,
            ollama_url="http://box:11434",
            ollama_model="aya-expanse:8b",
        )
        response = Mock()
        response.json.return_value = {"response": "Hallo"}
        with patch("requests.post", return_value=response) as post:
            self.assertEqual(tr.translate("hello", "en", "de"), "Hallo")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://box:11434/api/generate")
        self.assertEqual(kwargs["json"]["model"], "aya-expanse:8b")
        self.assertEqual(kwargs["json"]["options"], {"temperature": 0})

    def test_environment_overrides_the_default_model(self):
        with patch.dict("os.environ", {"POLYGLOT_OLLAMA_MODEL": "aya-expanse:8b"}):
            tr = Translator(on_status=Mock())
        self.assertEqual(tr._ollama_model, "aya-expanse:8b")


class NLLBReleaseTests(unittest.TestCase):
    def test_release_does_not_wait_for_an_in_flight_generation(self):
        tr = make_translator(engine=ENGINE_NLLB)
        tr._nllb = ("tokenizer", "model", "cuda")
        tr._nllb_lock.acquire()  # simulate an in-flight generate()
        try:
            thread = threading.Thread(target=tr._release_nllb)
            thread.start()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive(), "release blocked on the lock")
            self.assertIsNone(tr._nllb)
        finally:
            tr._nllb_lock.release()

    def test_set_nllb_model_does_not_block_on_a_running_generation(self):
        tr = make_translator(engine=ENGINE_NLLB)
        tr._nllb = ("tokenizer", "model", "cuda")
        tr._nllb_lock.acquire()
        try:
            thread = threading.Thread(
                target=tr.set_nllb_model,
                args=("facebook/nllb-200-distilled-1.3B",),
            )
            thread.start()
            thread.join(timeout=1.0)
            self.assertFalse(thread.is_alive(), "model switch blocked on the lock")
            self.assertIsNone(tr._nllb)
            self.assertEqual(
                tr._nllb_model_name, "facebook/nllb-200-distilled-1.3B"
            )
        finally:
            tr._nllb_lock.release()

    def test_switching_away_releases_the_model(self):
        tr = make_translator(engine=ENGINE_NLLB)
        tr._nllb = ("tokenizer", "model", "cuda")
        tr.set_engine(ENGINE_ARGOS)
        self.assertIsNone(tr._nllb)

    def test_switching_between_other_engines_keeps_nllb(self):
        tr = make_translator(engine=ENGINE_GOOGLE)
        tr._nllb = ("tokenizer", "model", "cuda")
        tr.set_engine(ENGINE_ARGOS)
        self.assertIsNotNone(tr._nllb)


class ModelAvailabilityTests(unittest.TestCase):
    def test_ollama_model_presence_is_checked_via_tags(self):
        tr = make_translator()
        response = Mock()
        response.json.return_value = {"models": [{"name": "aya-expanse:8b"}]}
        with patch("requests.get", return_value=response):
            self.assertTrue(tr.is_model_available(ENGINE_OLLAMA, "aya-expanse:8b"))
            self.assertFalse(tr.is_model_available(ENGINE_OLLAMA, "qwen2.5:7b"))

    def test_unreachable_ollama_reports_unavailable(self):
        tr = make_translator()
        with patch("requests.get", side_effect=OSError("connection refused")):
            self.assertFalse(tr.is_model_available(ENGINE_OLLAMA, "aya-expanse:8b"))

    def test_nllb_availability_uses_the_hf_cache(self):
        tr = make_translator()
        with patch("translator._nllb_model_cached", return_value=True):
            self.assertTrue(
                tr.is_model_available(
                    ENGINE_NLLB, "facebook/nllb-200-distilled-600M"
                )
            )

    def test_pull_dispatch_selects_the_engine(self):
        tr = make_translator()
        with patch.object(tr, "_pull_ollama", return_value=True) as pull:
            self.assertTrue(tr.pull_model(ENGINE_OLLAMA, "aya-expanse:8b"))
        pull.assert_called_once_with("aya-expanse:8b", None)

    def test_set_nllb_model_releases_the_loaded_model(self):
        tr = make_translator(engine=ENGINE_NLLB)
        tr._nllb = ("tokenizer", "model", "cuda")
        tr.set_nllb_model("facebook/nllb-200-distilled-1.3B")
        self.assertIsNone(tr._nllb)
        self.assertEqual(tr._nllb_model_name, "facebook/nllb-200-distilled-1.3B")

    def test_ollama_failure_falls_back_to_argos(self):
        tr = make_translator(engine=ENGINE_OLLAMA)
        with patch("requests.post", side_effect=OSError("down")), patch.object(
            tr, "_argo_translate", return_value="offline"
        ) as argo:
            self.assertEqual(tr.translate("hello", "en", "fa"), "offline")
        argo.assert_called_once()


class EngineReleaseTests(unittest.TestCase):
    def test_ollama_unload_asks_for_immediate_eviction(self):
        tr = make_translator(ollama_url="http://box:11434")
        with patch("requests.post") as post:
            tr._ollama_unload("aya-expanse:8b")
        args, kwargs = post.call_args
        self.assertEqual(args[0], "http://box:11434/api/generate")
        self.assertEqual(kwargs["json"], {"model": "aya-expanse:8b", "keep_alive": 0})

    def test_switching_away_from_ollama_unloads_its_model(self):
        tr = make_translator(engine=ENGINE_OLLAMA, ollama_model="aya-expanse:8b")
        seen = []
        done = threading.Event()

        def fake_unload(model):
            seen.append(model)
            done.set()

        with patch.object(tr, "_ollama_unload", side_effect=fake_unload):
            tr.set_engine(ENGINE_ARGOS)
        self.assertTrue(done.wait(1.0), "unload did not run")
        self.assertEqual(seen, ["aya-expanse:8b"])

    def test_changing_the_ollama_model_unloads_the_previous_one(self):
        tr = make_translator(engine=ENGINE_OLLAMA, ollama_model="aya-expanse:8b")
        seen = []
        done = threading.Event()

        def fake_unload(model):
            seen.append(model)
            done.set()

        with patch.object(tr, "_ollama_unload", side_effect=fake_unload):
            tr.set_ollama_model("qwen2.5:7b")
        self.assertTrue(done.wait(1.0), "unload did not run")
        self.assertEqual(seen, ["aya-expanse:8b"])
        self.assertEqual(tr._ollama_model, "qwen2.5:7b")

    def test_openai_unload_only_probes_a_local_ollama(self):
        tr = make_translator(
            openai_url="http://localhost:11434/v1", openai_model="aya-expanse:8b"
        )
        with patch("requests.post") as post:
            tr._openai_unload()
        post.assert_called_once()
        self.assertEqual(post.call_args.args[0], "http://localhost:11434/api/generate")

        hosted = make_translator(
            openai_url="https://api.openai.com/v1", openai_model="gpt-4o-mini"
        )
        with patch("requests.post") as post:
            hosted._openai_unload()
        post.assert_not_called()

    def test_switching_between_other_engines_keeps_nllb(self):
        tr = make_translator(engine=ENGINE_GOOGLE)
        tr._nllb = ("tokenizer", "model", "cuda")
        tr.set_engine(ENGINE_ARGOS)
        self.assertIsNotNone(tr._nllb)


class LLMPromptTests(unittest.TestCase):
    def test_prompt_uses_language_names(self):
        prompt = _llm_prompt("hello", "en", "fa")
        self.assertIn("English", prompt)
        self.assertIn("Persian", prompt)
        self.assertTrue(prompt.endswith("hello"))

    def test_quotes_are_stripped_once(self):
        self.assertEqual(_clean_llm_output(' "Hallo" '), "Hallo")
        self.assertEqual(_clean_llm_output("'Hallo'"), "Hallo")
        self.assertEqual(_clean_llm_output('He said "Hallo"'), 'He said "Hallo"')

    def test_argos_constant_is_unchanged(self):
        self.assertEqual(ENGINE_ARGOS, "argos")


if __name__ == "__main__":
    unittest.main()
