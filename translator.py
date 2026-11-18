"""
translator.py — Multi-engine translation with automatic fallback.

Supported engines (selectable at runtime via set_engine()):
  google        — Google Translate via deep-translator (needs internet)
  mymemory      — MyMemory free API via deep-translator (needs internet)
  libretranslate— LibreTranslate REST API; configurable base URL for
                  self-hosted instances (needs internet or local server)
  ollama        — Local LLM inference via Ollama REST API (offline, GPU)
  openai        — Any OpenAI-compatible /chat/completions endpoint:
                  LM Studio, llama.cpp server, vLLM, OpenRouter, Groq, …
  nllb          — NLLB-200 via transformers (offline, GPU, 200 languages)
  argos         — argostranslate fully offline; packages downloaded on demand

Fallback:
  When the active engine fails (network error / timeout / missing model),
  the call falls through to argostranslate automatically.  A background probe
  restores the primary engine after connectivity returns (every 30 s).
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Engine name constants — use these rather than bare strings.
ENGINE_GOOGLE = "google"
ENGINE_MYMEMORY = "mymemory"
ENGINE_LIBRETRANSLATE = "libretranslate"
ENGINE_OLLAMA = "ollama"
ENGINE_OPENAI = "openai"
ENGINE_NLLB = "nllb"
ENGINE_ARGOS = "argos"

_ONLINE_ENGINES = {ENGINE_GOOGLE, ENGINE_MYMEMORY, ENGINE_LIBRETRANSLATE}

# How long to wait before re-probing connectivity after a failure (seconds).
_CONNECTIVITY_RETRY_S = 30
# Host used for the connectivity probe — just a DNS lookup, no HTTP.
_PROBE_HOST = "translate.googleapis.com"

# Default LibreTranslate public instance.
_LIBRETRANSLATE_DEFAULT_URL = "https://libretranslate.com"
# Ollama default endpoint and model.
_OLLAMA_DEFAULT_URL = "http://localhost:11434"
_OLLAMA_DEFAULT_MODEL = "aya-expanse:8b"
# OpenAI-compatible endpoint; Ollama exposes this path as well.
_OPENAI_DEFAULT_URL = "http://localhost:11434/v1"
_OPENAI_DEFAULT_MODEL = "mistral"
# NLLB default model: 600M distilled is fast enough for live sentences.
_NLLB_DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"

# Shared instruction for every LLM-backed engine.
_LLM_SYSTEM_PROMPT = (
    "You are a silent translation engine. "
    "When given text, you output ONLY the translated text — "
    "no explanations, no notes, no prefixes, no quotes, no labels. "
    "Never say 'Here is the translation' or anything similar. "
    "If the input is already in the target language, output it unchanged."
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Whisper uses standard ISO-639-1 codes, but Google Translate's API (via
# deep-translator) uses a slightly different set for a handful of languages.
_GOOGLE_CODE: dict[str, str] = {
    "zh": "zh-CN",
    "he": "iw",
}


def _to_google_lang(code: str) -> str:
    return _GOOGLE_CODE.get(code, code)


def _is_online() -> bool:
    try:
        socket.setdefaulttimeout(2)
        socket.getaddrinfo(_PROBE_HOST, 443)
        return True
    except OSError:
        return False


# ---------------------------------------------------------------------------
# Lazy argostranslate import
# ---------------------------------------------------------------------------

_argo_imported = False


def _import_argo() -> None:
    global _argo_imported
    if _argo_imported:
        return
    import argostranslate.package  # noqa: F401
    import argostranslate.translate  # noqa: F401

    _argo_imported = True


# ---------------------------------------------------------------------------
# Translator
# ---------------------------------------------------------------------------


class Translator:
    """
    Multi-engine translator with automatic online/offline fallback.

    Call set_engine(name) to switch engines at runtime.  Thread-safe.
    """

    def __init__(
        self,
        engine: str = ENGINE_GOOGLE,
        on_status: Optional[StatusCallback] = None,
        libretranslate_url: str = _LIBRETRANSLATE_DEFAULT_URL,
        libretranslate_api_key: str = "",
        ollama_url: Optional[str] = None,
        ollama_model: Optional[str] = None,
        openai_url: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_model: Optional[str] = None,
        nllb_model: Optional[str] = None,
    ):
        self._on_status = on_status or (lambda s: None)
        self._lock = threading.Lock()

        # Engine selection
        self._engine: str = engine

        # Online state (relevant only when engine is an online engine)
        # Assume online; the first failed request marks it offline, so startup
        # never blocks on a DNS lookup in the UI thread.
        self._online: bool = True
        self._last_probe: float = time.monotonic()

        # LibreTranslate config
        self._libretranslate_url = libretranslate_url.rstrip("/")
        self._libretranslate_api_key = libretranslate_api_key

        # Ollama config
        self._ollama_url = (
            ollama_url or os.environ.get("POLYGLOT_OLLAMA_URL") or _OLLAMA_DEFAULT_URL
        ).rstrip("/")
        self._ollama_model = (
            ollama_model
            or os.environ.get("POLYGLOT_OLLAMA_MODEL")
            or _OLLAMA_DEFAULT_MODEL
        )

        # OpenAI-compatible config (env overrides keep keys out of the source)
        self._openai_url = (
            openai_url or os.environ.get("POLYGLOT_OPENAI_URL") or _OPENAI_DEFAULT_URL
        ).rstrip("/")
        self._openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("POLYGLOT_OPENAI_API_KEY", "")
        )
        self._openai_model = (
            openai_model
            or os.environ.get("POLYGLOT_OPENAI_MODEL")
            or _OPENAI_DEFAULT_MODEL
        )

        # NLLB config and lazily loaded model
        self._nllb_model_name = (
            nllb_model or os.environ.get("POLYGLOT_NLLB_MODEL") or _NLLB_DEFAULT_MODEL
        )
        self._nllb_lock = threading.Lock()
        self._nllb = None  # (tokenizer, model, device)

        # argostranslate package tracking
        self._installed_pairs: set[tuple[str, str]] = set()
        self._downloading: set[tuple[str, str]] = set()

        logger.info(
            "Translator initialised — engine=%s, online=%s", engine, self._online
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_engine(self, engine: str) -> None:
        """Switch the active translation engine.  Thread-safe."""
        with self._lock:
            previous = self._engine
            self._engine = engine
        if previous != engine:
            self._release_engine(previous)
        logger.info("Translation engine changed to '%s'.", engine)
        self._on_status(f"Translation engine: {engine}")

    def _release_engine(self, engine: str) -> None:
        """Free whatever *engine* had loaded locally.

        Network unloads run on a daemon thread so a slow local server cannot
        block the UI thread that switches engines.
        """
        if engine == ENGINE_NLLB:
            self._on_status("Releasing NLLB model…")
            self._release_nllb()
        elif engine == ENGINE_OLLAMA:
            model = self._ollama_model
            self._on_status(f"Unloading Ollama model {model}…")
            threading.Thread(
                target=self._ollama_unload, args=(model,), daemon=True
            ).start()
        elif engine == ENGINE_OPENAI:
            self._on_status("Releasing OpenAI-compatible model…")
            threading.Thread(target=self._openai_unload, daemon=True).start()

    def _ollama_unload(self, model: str) -> None:
        """Ask Ollama to evict *model* now instead of after its idle timeout."""
        if not model:
            return
        try:
            import requests

            requests.post(
                f"{self._ollama_url}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=5,
            )
            logger.info("Unloaded Ollama model %s", model)
            self._on_status(f"Ollama model {model} unloaded")
        except Exception as exc:
            logger.debug("Ollama unload failed (%s): %s", model, exc)
            self._on_status(f"Ollama unload failed: {model}")

    def _openai_unload(self) -> None:
        """Best effort: unload when the OpenAI endpoint is a local Ollama."""
        base = self._openai_url
        if not base.endswith("/v1"):
            return
        base = base[:-3]
        if not any(
            host in base for host in ("localhost", "127.0.0.1", "[::1]")
        ):
            return  # hosted providers have no unload endpoint
        try:
            import requests

            requests.post(
                f"{base}/api/generate",
                json={"model": self._openai_model, "keep_alive": 0},
                timeout=5,
            )
            logger.info("Unloaded OpenAI-compatible local model %s", self._openai_model)
            self._on_status(f"{self._openai_model} unloaded")
        except Exception as exc:
            logger.debug("OpenAI-compatible unload failed: %s", exc)

    def _release_nllb(self) -> None:
        """Free the NLLB model when another engine takes over.

        Must never wait for the generation lock: an in-flight translation
        holds it for the whole generate() call, and this runs on the UI
        thread when the user switches engines.  The running call already
        holds its own reference to the weights, so dropping the shared one
        frees the VRAM as soon as it finishes.
        """
        if not self._nllb_lock.acquire(blocking=False):
            self._nllb = None
            logger.info("Released NLLB model (generation still running)")
            self._on_status("NLLB model released")
            return
        try:
            if self._nllb is None:
                return
            self._nllb = None
        finally:
            self._nllb_lock.release()
        try:
            import torch

            if torch.cuda.is_available():
                # empty_cache synchronises the device; never do that on the
                # UI thread while Whisper may be decoding.
                threading.Thread(target=torch.cuda.empty_cache, daemon=True).start()
        except Exception:
            pass
        logger.info("Released NLLB model")
        self._on_status("NLLB model released")

    def set_ollama_model(self, model: str) -> None:
        """Switch the Ollama model used for translation.  Thread-safe."""
        with self._lock:
            previous = self._ollama_model
            self._ollama_model = model
        if previous and previous != model:
            threading.Thread(
                target=self._ollama_unload, args=(previous,), daemon=True
            ).start()
        logger.info("Ollama model changed to '%s'.", model)

    def set_nllb_model(self, model: str) -> None:
        """Switch the NLLB model; the next call reloads it.  Thread-safe."""
        if model == self._nllb_model_name:
            return
        self._release_nllb()
        # Plain attribute write: acquiring _nllb_lock here would block the
        # caller (the UI thread) behind an in-flight generation.
        self._nllb_model_name = model
        logger.info("NLLB model changed to '%s'.", model)

    # ------------------------------------------------------------------
    # Model availability / on-demand download
    # ------------------------------------------------------------------

    def is_model_available(self, engine: str, model: str = "") -> bool:
        """Whether *model* is already present locally for *engine*."""
        if engine == ENGINE_OLLAMA:
            names = self.ollama_models()
            if names is None:
                return False
            return model in names or f"{model}:latest" in names
        if engine == ENGINE_NLLB:
            return _nllb_model_cached(model)
        return True

    def ollama_models(self) -> Optional[set[str]]:
        """Installed Ollama model names, or None when the server is unreachable."""
        try:
            import requests

            resp = requests.get(f"{self._ollama_url}/api/tags", timeout=3)
            resp.raise_for_status()
            return {entry["name"] for entry in resp.json().get("models", [])}
        except Exception as exc:
            logger.debug("Ollama tags query failed: %s", exc)
            return None

    def pull_model(self, engine: str, model: str, on_progress=None) -> bool:
        """Download *model* for *engine*.  Blocking; call from a worker thread."""
        if engine == ENGINE_OLLAMA:
            return self._pull_ollama(model, on_progress)
        if engine == ENGINE_NLLB:
            return _download_nllb(model, on_progress)
        return True

    def _pull_ollama(self, model: str, on_progress=None) -> bool:
        try:
            import json

            import requests

            with requests.post(
                f"{self._ollama_url}/api/pull",
                json={"model": model, "stream": True},
                stream=True,
                timeout=(5, 3600),
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("error"):
                        raise RuntimeError(data["error"])
                    if on_progress:
                        total = data.get("total")
                        completed = data.get("completed")
                        if total and completed:
                            on_progress(f"Pulling {model}… {completed / total * 100:.0f}%")
                        elif data.get("status"):
                            on_progress(f"{data['status']} — {model}")
            logger.info("Ollama model %s pulled", model)
            return True
        except Exception as exc:
            logger.warning("Ollama pull failed (%s): %s", model, exc)
            return False

    @property
    def engine(self) -> str:
        with self._lock:
            return self._engine

    def translate(self, text: str, from_lang: str, to_lang: str, **_kwargs) -> str:
        """
        Translate text from from_lang to to_lang using the active engine.

        Falls back to argostranslate on network failure for online engines.
        Returns original text on unrecoverable failure.
        """
        if not text.strip():
            return text
        if from_lang == to_lang:
            return text

        with self._lock:
            engine = self._engine

        if engine == ENGINE_ARGOS:
            return self._argo_translate(text, from_lang, to_lang)

        if engine == ENGINE_OLLAMA:
            result = self._ollama_translate(text, from_lang, to_lang)
            if result is not None:
                return result
            logger.warning(
                "Ollama failed (%s→%s), falling back to argostranslate.",
                from_lang,
                to_lang,
            )
            self._on_status("ollama unavailable — using offline fallback")
            return self._argo_translate(text, from_lang, to_lang)

        if engine == ENGINE_OPENAI:
            result = self._openai_translate(text, from_lang, to_lang)
            if result is not None:
                return result
            logger.warning(
                "OpenAI-compatible endpoint failed (%s→%s), falling back to argostranslate.",
                from_lang,
                to_lang,
            )
            self._on_status("openai unavailable — using offline fallback")
            return self._argo_translate(text, from_lang, to_lang)

        if engine == ENGINE_NLLB:
            result = self._nllb_translate(text, from_lang, to_lang)
            return result if result is not None else self._argo_translate(
                text, from_lang, to_lang
            )

        # Online engine — try primary, fall back to argos on failure.
        if self._check_online():
            result = self._call_online_engine(engine, text, from_lang, to_lang)
            if result is not None:
                return result
            # Mark offline and fall through.
            with self._lock:
                self._online = False
                self._last_probe = time.monotonic()
            logger.warning(
                "%s failed (%s→%s), falling back to argostranslate.",
                engine,
                from_lang,
                to_lang,
            )
            self._on_status(f"{engine} unavailable — using offline fallback")

        return self._argo_translate(text, from_lang, to_lang)

    def ensure_package(
        self,
        from_lang: str,
        to_lang: str,
        blocking: bool = False,
    ) -> bool:
        if from_lang == to_lang:
            return True
        if self._argo_is_installed(from_lang, to_lang):
            return True
        pair = (from_lang, to_lang)
        with self._lock:
            if pair in self._downloading:
                return False
            self._downloading.add(pair)
        t = threading.Thread(
            target=self._argo_download,
            args=(from_lang, to_lang),
            daemon=True,
            name=f"ArgoDownload-{from_lang}-{to_lang}",
        )
        t.start()
        if blocking:
            t.join()
        return False

    def installed_languages(self) -> list[str]:
        try:
            _import_argo()
            import argostranslate.translate as at

            return [lang.code for lang in at.get_installed_languages()]
        except Exception as exc:
            logger.warning("Could not list installed languages: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Online state
    # ------------------------------------------------------------------

    def _check_online(self) -> bool:
        with self._lock:
            online = self._online
            elapsed = time.monotonic() - self._last_probe
        if not online and elapsed >= _CONNECTIVITY_RETRY_S:
            now_online = _is_online()
            with self._lock:
                self._online = now_online
                self._last_probe = time.monotonic()
            if now_online:
                logger.info(
                    "Connectivity restored — switching back to %s.", self._engine
                )
                self._on_status(f"Online — {self._engine} resumed")
            return now_online
        return online

    # ------------------------------------------------------------------
    # Engine dispatch
    # ------------------------------------------------------------------

    def _call_online_engine(
        self, engine: str, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        if engine == ENGINE_GOOGLE:
            return self._google_translate(text, from_lang, to_lang)
        if engine == ENGINE_MYMEMORY:
            return self._mymemory_translate(text, from_lang, to_lang)
        if engine == ENGINE_LIBRETRANSLATE:
            return self._libretranslate_translate(text, from_lang, to_lang)
        logger.warning("Unknown engine '%s', using Google.", engine)
        return self._google_translate(text, from_lang, to_lang)

    # ------------------------------------------------------------------
    # Google Translate
    # ------------------------------------------------------------------

    def _google_translate(
        self, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        try:
            from deep_translator import GoogleTranslator

            result = GoogleTranslator(
                source=_to_google_lang(from_lang),
                target=_to_google_lang(to_lang),
            ).translate(text)
            return result or None
        except Exception as exc:
            logger.debug("Google Translate error (%s→%s): %s", from_lang, to_lang, exc)
            return None

    # ------------------------------------------------------------------
    # MyMemory
    # ------------------------------------------------------------------

    def _mymemory_translate(
        self, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        try:
            from deep_translator import MyMemoryTranslator

            result = MyMemoryTranslator(
                source=from_lang,
                target=to_lang,
            ).translate(text)
            if isinstance(result, list):
                result = " ".join(result)
            return result or None
        except Exception as exc:
            logger.debug("MyMemory error (%s→%s): %s", from_lang, to_lang, exc)
            return None

    # ------------------------------------------------------------------
    # LibreTranslate
    # ------------------------------------------------------------------

    def _libretranslate_translate(
        self, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        try:
            import requests  # system-wide

            payload: dict = {"q": text, "source": from_lang, "target": to_lang}
            if self._libretranslate_api_key:
                payload["api_key"] = self._libretranslate_api_key
            resp = requests.post(
                f"{self._libretranslate_url}/translate",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("translatedText") or None
        except Exception as exc:
            logger.debug("LibreTranslate error (%s→%s): %s", from_lang, to_lang, exc)
            return None

    # ------------------------------------------------------------------
    # Ollama (local LLM)
    # ------------------------------------------------------------------

    def _ollama_translate(
        self, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        """Translate via a local Ollama model.  No fallback — offline by design."""
        try:
            import requests  # system-wide

            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._ollama_model,
                    "system": _LLM_SYSTEM_PROMPT,
                    "prompt": _llm_prompt(text, from_lang, to_lang),
                    "stream": False,
                    # Translation should be repeatable, not creative.
                    "options": {"temperature": 0},
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            return _clean_llm_output(data.get("response", "")) or None
        except Exception as exc:
            logger.debug("Ollama error (%s→%s): %s", from_lang, to_lang, exc)
            return None

    # ------------------------------------------------------------------
    # OpenAI-compatible chat completions
    # ------------------------------------------------------------------

    def _openai_translate(
        self, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        """Translate via any /chat/completions endpoint (LM Studio, vLLM, …)."""
        try:
            import requests  # system-wide

            headers = {}
            if self._openai_api_key:
                headers["Authorization"] = f"Bearer {self._openai_api_key}"
            resp = requests.post(
                f"{self._openai_url}/chat/completions",
                json={
                    "model": self._openai_model,
                    "temperature": 0,
                    "messages": [
                        {"role": "system", "content": _LLM_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _llm_prompt(text, from_lang, to_lang),
                        },
                    ],
                },
                headers=headers or None,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            return _clean_llm_output(content) or None
        except Exception as exc:
            logger.debug(
                "OpenAI-compatible error (%s→%s): %s", from_lang, to_lang, exc
            )
            return None

    # ------------------------------------------------------------------
    # NLLB-200 (local, transformers)
    # ------------------------------------------------------------------

    def _nllb_translate(
        self, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        source = _NLLB_CODES.get(from_lang)
        target = _NLLB_CODES.get(to_lang)
        if not source or not target:
            logger.debug("NLLB has no code for %s→%s", from_lang, to_lang)
            return None
        try:
            return self._nllb_generate(text, source, target)
        except Exception as exc:
            logger.warning("NLLB error (%s→%s): %s", from_lang, to_lang, exc)
            return None

    def _nllb_resources(self):
        """Load tokenizer/model on first use; Inference threads share them."""
        with self._nllb_lock:
            if self._nllb is not None:
                return self._nllb
            import torch

            # Native download/load output would corrupt the Textual screen:
            # keep huggingface_hub and transformers quiet on stderr.
            _quiet_hf()
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self._on_status(f"Loading NLLB model {self._nllb_model_name}…")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(self._nllb_model_name)
            try:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._nllb_model_name, dtype=dtype
                )
            except TypeError:  # transformers < 5 spells the argument differently
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self._nllb_model_name, torch_dtype=dtype
                )
            model = model.to(device).eval()
            self._nllb = (tokenizer, model, device)
            logger.info("NLLB model %s loaded on %s", self._nllb_model_name, device)
            self._on_status("NLLB ready")
            return self._nllb

    def _nllb_generate(self, text: str, source: str, target: str) -> Optional[str]:
        import torch

        tokenizer, model, device = self._nllb_resources()
        with self._nllb_lock:  # serialise generation; sentences are short
            batch = tokenizer(
                text,
                src_lang=source,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(target),
                    max_length=256,
                    num_beams=1,
                )
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip() or None

    # ------------------------------------------------------------------
    # argostranslate fallback
    # ------------------------------------------------------------------

    def _argo_translate(self, text: str, from_lang: str, to_lang: str) -> str:
        if not self._argo_is_installed(from_lang, to_lang):
            self.ensure_package(from_lang, to_lang)
            logger.debug(
                "argostranslate package %s→%s not ready yet, returning source.",
                from_lang,
                to_lang,
            )
            return text
        try:
            _import_argo()
            import argostranslate.translate as at

            result = at.translate(text, from_lang, to_lang)
            return result or text
        except Exception as exc:
            logger.error("argostranslate error (%s→%s): %s", from_lang, to_lang, exc)
            return text

    def _argo_is_installed(self, from_lang: str, to_lang: str) -> bool:
        pair = (from_lang, to_lang)
        with self._lock:
            if pair in self._installed_pairs:
                return True
        try:
            _import_argo()
            import argostranslate.translate as at

            langs = {lang.code: lang for lang in at.get_installed_languages()}
            if from_lang in langs:
                # translations_to holds CachedTranslation objects exposing
                # .to_lang (a Language); .code does not exist on them.
                targets = set()
                for translation in langs[from_lang].translations_to:
                    target = getattr(translation, "to_lang", None)
                    code = getattr(target, "code", None) or getattr(
                        translation, "to_code", None
                    )
                    if code:
                        targets.add(code)
                if to_lang in targets:
                    with self._lock:
                        self._installed_pairs.add(pair)
                    return True
        except Exception:
            pass
        return False

    def _argo_download(self, from_lang: str, to_lang: str) -> None:
        pair = (from_lang, to_lang)
        try:
            self._on_status(f"Downloading offline package {from_lang}→{to_lang}…")
            _import_argo()
            import argostranslate.package as ap

            ap.update_package_index()
            pkgs = ap.get_available_packages()
            match = next(
                (p for p in pkgs if p.from_code == from_lang and p.to_code == to_lang),
                None,
            )
            if match is None:
                self._on_status(
                    f"No offline package available for {from_lang}→{to_lang}"
                )
                logger.warning(
                    "No argostranslate package for %s→%s", from_lang, to_lang
                )
                return
            ap.install_from_path(match.download())
            with self._lock:
                self._installed_pairs.add(pair)
                self._downloading.discard(pair)
            self._on_status(f"Offline package {from_lang}→{to_lang} ready.")
            logger.info("argostranslate package %s→%s installed.", from_lang, to_lang)
        except Exception as exc:
            logger.error("Package download failed (%s→%s): %s", from_lang, to_lang, exc)
            with self._lock:
                self._downloading.discard(pair)
            self._on_status(f"Package download failed: {exc}")


# ---------------------------------------------------------------------------
# Language display names (for Ollama prompt)
# ---------------------------------------------------------------------------

_LANG_NAMES: dict[str, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "ar": "Arabic",
    "hi": "Hindi",
    "tr": "Turkish",
    "pl": "Polish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "no": "Norwegian",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "hr": "Croatian",
    "sr": "Serbian",
    "uk": "Ukrainian",
    "el": "Greek",
    "he": "Hebrew",
    "fa": "Persian",
    "vi": "Vietnamese",
    "th": "Thai",
    "id": "Indonesian",
    "ms": "Malay",
    "ca": "Catalan",
    "la": "Latin",
    "af": "Afrikaans",
    "sq": "Albanian",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "eu": "Basque",
    "be": "Belarusian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "gl": "Galician",
    "ka": "Georgian",
    "gu": "Gujarati",
    "ht": "Haitian Creole",
    "is": "Icelandic",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "mn": "Mongolian",
    "ne": "Nepali",
    "pa": "Punjabi",
    "si": "Sinhala",
    "sl": "Slovenian",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "tl": "Filipino",
    "ur": "Urdu",
    "uz": "Uzbek",
}


# ---------------------------------------------------------------------------
# LLM prompt helpers (Ollama and OpenAI-compatible engines)
# ---------------------------------------------------------------------------


def _llm_prompt(text: str, from_lang: str, to_lang: str) -> str:
    from_name = _LANG_NAMES.get(from_lang, from_lang)
    to_name = _LANG_NAMES.get(to_lang, to_lang)
    return f"Translate the following {from_name} text to {to_name}.\n\n{text}"


def _clean_llm_output(result: str) -> str:
    """Strip whitespace and a single pair of wrapping quotes."""
    result = (result or "").strip()
    if len(result) >= 2 and result[0] in ('"', "'") and result[-1] == result[0]:
        result = result[1:-1].strip()
    return result


# ---------------------------------------------------------------------------
# NLLB-200 language codes
# ---------------------------------------------------------------------------
# NLLB uses BCP-47-like codes with script tags; map the ISO-639-1 codes that
# Whisper emits.  Languages missing here fall back to argostranslate.


def _quiet_hf() -> None:
    """Keep huggingface download/progress output off the Textual screen."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except Exception:
        pass


def _nllb_model_cached(model_name: str) -> bool:
    """Whether config + weights for *model_name* are already in the HF cache."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:
        return False
    try:
        config = try_to_load_from_cache(model_name, "config.json")
        if not isinstance(config, str):
            return False
        for filename in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
        ):
            if isinstance(try_to_load_from_cache(model_name, filename), str):
                return True
        return False
    except Exception:
        return False


def _download_nllb(model_name: str, on_progress=None) -> bool:
    try:
        _quiet_hf()
        from huggingface_hub import snapshot_download

        if on_progress:
            on_progress(f"Downloading {model_name}…")
        snapshot_download(repo_id=model_name)
        logger.info("NLLB model %s downloaded", model_name)
        return True
    except Exception as exc:
        logger.warning("NLLB download failed (%s): %s", model_name, exc)
        return False

_NLLB_CODES: dict[str, str] = {
    "af": "afr_Latn",
    "ar": "arb_Arab",
    "az": "azj_Latn",
    "be": "bel_Cyrl",
    "bg": "bul_Cyrl",
    "bn": "ben_Beng",
    "bs": "bos_Latn",
    "ca": "cat_Latn",
    "cs": "ces_Latn",
    "da": "dan_Latn",
    "de": "deu_Latn",
    "el": "ell_Grek",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "et": "est_Latn",
    "eu": "eus_Latn",
    "fa": "pes_Arab",
    "fi": "fin_Latn",
    "fr": "fra_Latn",
    "gl": "glg_Latn",
    "gu": "guj_Gujr",
    "he": "heb_Hebr",
    "hi": "hin_Deva",
    "hr": "hrv_Latn",
    "hu": "hun_Latn",
    "hy": "hye_Armn",
    "id": "ind_Latn",
    "is": "isl_Latn",
    "it": "ita_Latn",
    "ja": "jpn_Jpan",
    "ka": "kat_Geor",
    "kk": "kaz_Cyrl",
    "ko": "kor_Hang",
    "lt": "lit_Latn",
    "lv": "lvs_Latn",
    "mk": "mkd_Cyrl",
    "ml": "mal_Mlym",
    "mn": "khk_Cyrl",
    "mr": "mar_Deva",
    "ms": "zsm_Latn",
    "ne": "npi_Deva",
    "nl": "nld_Latn",
    "no": "nob_Latn",
    "pa": "pan_Guru",
    "pl": "pol_Latn",
    "pt": "por_Latn",
    "ro": "ron_Latn",
    "ru": "rus_Cyrl",
    "sk": "slk_Latn",
    "sl": "slv_Latn",
    "sq": "als_Latn",
    "sr": "srp_Cyrl",
    "sv": "swe_Latn",
    "sw": "swh_Latn",
    "ta": "tam_Taml",
    "te": "tel_Telu",
    "th": "tha_Thai",
    "tl": "tgl_Latn",
    "tr": "tur_Latn",
    "uk": "ukr_Cyrl",
    "ur": "urd_Arab",
    "uz": "uzn_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}
