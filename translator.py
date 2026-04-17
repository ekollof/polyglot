"""
translator.py — Multi-engine translation with automatic fallback.

Supported engines (selectable at runtime via set_engine()):
  google        — Google Translate via deep-translator (needs internet)
  mymemory      — MyMemory free API via deep-translator (needs internet)
  libretranslate— LibreTranslate REST API; configurable base URL for
                  self-hosted instances (needs internet or local server)
  ollama        — Local LLM inference via Ollama REST API (offline, GPU)
  argos         — argostranslate fully offline; packages downloaded on demand

Fallback:
  When the active online engine fails (network error / timeout), the call
  falls through to argostranslate automatically.  A background probe restores
  the primary engine after connectivity returns (every 30 s).
"""

from __future__ import annotations

import logging
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
_OLLAMA_DEFAULT_MODEL = "mistral"

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
        ollama_url: str = _OLLAMA_DEFAULT_URL,
        ollama_model: str = _OLLAMA_DEFAULT_MODEL,
    ):
        self._on_status = on_status or (lambda s: None)
        self._lock = threading.Lock()

        # Engine selection
        self._engine: str = engine

        # Online state (relevant only when engine is an online engine)
        self._online: bool = _is_online()
        self._last_probe: float = time.monotonic()

        # LibreTranslate config
        self._libretranslate_url = libretranslate_url.rstrip("/")
        self._libretranslate_api_key = libretranslate_api_key

        # Ollama config
        self._ollama_url = ollama_url.rstrip("/")
        self._ollama_model = ollama_model

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
            self._engine = engine
        logger.info("Translation engine changed to '%s'.", engine)
        self._on_status(f"Translation engine: {engine}")

    def set_ollama_model(self, model: str) -> None:
        """Switch the Ollama model used for translation.  Thread-safe."""
        with self._lock:
            self._ollama_model = model
        logger.info("Ollama model changed to '%s'.", model)

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
            return result if result is not None else text

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

            # Language display names for a more natural prompt.
            from_name = _LANG_NAMES.get(from_lang, from_lang)
            to_name = _LANG_NAMES.get(to_lang, to_lang)
            prompt = (
                f"Translate the following {from_name} text to {to_name}. "
                f"Output ONLY the translation with no explanation, no quotes, "
                f"no prefix, no extra words.\n\n"
                f"Text: {text}\n\nTranslation:"
            )
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json={
                    "model": self._ollama_model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data.get("response", "").strip()
            # Strip any leading/trailing quotes the model might add.
            if result and result[0] in ('"', "'") and result[-1] == result[0]:
                result = result[1:-1].strip()
            return result or None
        except Exception as exc:
            logger.debug("Ollama error (%s→%s): %s", from_lang, to_lang, exc)
            return None

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
                targets = {t.code for t in langs[from_lang].translations_to}
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
