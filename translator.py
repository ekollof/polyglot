"""
translator.py — Two-tier offline/online translation.

Primary:  Google Translate via deep-translator (requires internet).
Fallback: argostranslate (fully offline, pre-downloaded packages).

On every translate() call the primary is tried first.  If it raises any
network-related exception (or any exception at all) the fallback is used
instead and a flag is set so subsequent calls skip the primary until a
connectivity probe succeeds again (checked in a background thread every 30s).

argostranslate packages are downloaded on demand the first time a language
pair is needed.  The download is non-blocking; the original text is returned
while the package is fetched, after which future calls will use it.
"""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

StatusCallback = Callable[[str], None]

# How long to wait before re-probing connectivity after a failure (seconds).
_CONNECTIVITY_RETRY_S = 30
# Host used for the connectivity probe — just a DNS lookup, no HTTP.
_PROBE_HOST = "translate.googleapis.com"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Whisper uses standard ISO-639-1 codes, but Google Translate's API (via
# deep-translator) uses a slightly different set for a handful of languages.
# This table maps Whisper output codes → GoogleTranslator source/target codes.
_GOOGLE_CODE: dict[str, str] = {
    "zh": "zh-CN",  # Whisper returns 'zh'; Google needs 'zh-CN' or 'zh-TW'
    "he": "iw",  # Whisper returns 'he'; Google uses the older code 'iw'
}


def _to_google_lang(code: str) -> str:
    """Normalise a Whisper/ISO-639-1 language code to a GoogleTranslator code."""
    return _GOOGLE_CODE.get(code, code)


def _is_online() -> bool:
    """Return True if we can resolve Google's translate endpoint."""
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
    Translate text using Google Translate (online) with argostranslate fallback.

    Thread-safe; translate() may be called from any thread.
    """

    def __init__(self, on_status: Optional[StatusCallback] = None):
        self._on_status = on_status or (lambda s: None)
        self._lock = threading.Lock()

        # Online state
        self._online: bool = _is_online()
        self._last_probe: float = time.monotonic()

        # argostranslate package tracking
        self._installed_pairs: set[tuple[str, str]] = set()
        self._downloading: set[tuple[str, str]] = set()

        logger.info("Translator initialised — online=%s", self._online)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str, from_lang: str, to_lang: str, **_kwargs) -> str:
        """
        Translate text from from_lang to to_lang.

        Extra keyword arguments (e.g. whisper_already_translated) are accepted
        and silently ignored for API compatibility.

        Returns translated text, or original text on unrecoverable failure.
        """
        if not text.strip():
            return text
        if from_lang == to_lang:
            return text

        # Try Google first if we believe we're online.
        if self._check_online():
            result = self._google_translate(text, from_lang, to_lang)
            if result is not None:
                return result
            # Google failed — mark offline and fall through.
            with self._lock:
                self._online = False
                self._last_probe = time.monotonic()
            logger.warning(
                "Google Translate failed (%s→%s), falling back to argostranslate.",
                from_lang,
                to_lang,
            )
            self._on_status("Google Translate unavailable — using offline fallback")

        return self._argo_translate(text, from_lang, to_lang)

    def ensure_package(
        self,
        from_lang: str,
        to_lang: str,
        blocking: bool = False,
    ) -> bool:
        """
        Ensure the argostranslate package for (from_lang→to_lang) is installed.

        Returns True if already installed, False if a download was started.
        Downloads happen in a background thread unless blocking=True.
        """
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
        """Return language codes for installed argostranslate packages."""
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
        """Return current online belief; re-probe if enough time has passed."""
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
                    "Connectivity restored — switching back to Google Translate."
                )
                self._on_status("Online — Google Translate resumed")
            return now_online

        return online

    # ------------------------------------------------------------------
    # Google Translate
    # ------------------------------------------------------------------

    def _google_translate(
        self, text: str, from_lang: str, to_lang: str
    ) -> Optional[str]:
        """Call Google Translate. Returns None on any error."""
        try:
            from deep_translator import GoogleTranslator

            g_from = _to_google_lang(from_lang)
            g_to = _to_google_lang(to_lang)
            result = GoogleTranslator(source=g_from, target=g_to).translate(text)
            return result or None
        except Exception as exc:
            logger.debug("Google Translate error (%s→%s): %s", from_lang, to_lang, exc)
            return None

    # ------------------------------------------------------------------
    # argostranslate fallback
    # ------------------------------------------------------------------

    def _argo_translate(self, text: str, from_lang: str, to_lang: str) -> str:
        """Translate via argostranslate. Returns original text on failure."""
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
