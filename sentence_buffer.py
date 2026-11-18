"""Assemble confirmed ASR chunks into sentences before translation."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr vs etc inc corp ltd est dept approx fig "
    "jan feb mar apr jun jul aug sep oct nov dec st ave blvd u.s u.k e.g i.e "
    "bijv dhr mevr o.a m.a.w d.w.z z.b bzw nr mme m sra dra".split()
)
_CLOSING = frozenset("\"'”’»)]}」』】》")
_TERMINAL = frozenset(".!?。！？؟۔।॥")
_LEADING_ELLIPSIS = re.compile(r"^\s*(?:\.{2,}|…+)\s*")
_TRAILING_ELLIPSIS = re.compile(r"(?:\.{2,}|…+)\s*$")

# Generic cap for punctuation-free ASR output.  Whisper can return a long
# block with no sentence punctuation, and holding it until the 2400-character
# flush delays translation by many sentences.  No language-specific words are
# involved: a run is cut at clause punctuation when available, otherwise at
# the last space, and hard-cut at the limit for scripts that do not use spaces.
_SOFT_LIMIT = 180
_MIN_PIECE = 48
_CLAUSE_BREAKS = ",;:，、；：،—–-"


def _best_length_cut(text: str, limit: int = _SOFT_LIMIT) -> int:
    """Cut position for an unpunctuated run, preferring a natural boundary."""
    start = min(_MIN_PIECE, len(text) // 2)
    best = -1
    for character in _CLAUSE_BREAKS:
        index = text.rfind(character, start, limit)
        if index > best:
            best = index
    if best >= start:
        return best + 1
    space = text.rfind(" ", start, limit)
    if space >= start:
        return space + 1
    return limit  # scripts without spaces


def split_long_run(text: str):
    """Split punctuation-free text into translation-sized pieces.

    Returns (finished pieces, remainder).  The remainder is short enough to
    wait for the next confirmed chunk.
    """
    pieces: list[str] = []
    text = text.strip()
    while len(text) > _SOFT_LIMIT:
        cut = _best_length_cut(text)
        piece = text[:cut].strip()
        if piece:
            pieces.append(piece)
        text = text[cut:].strip()
    return pieces, text


def join_fragments(left: str, right: str) -> str:
    """Remove only ellipses at the join, preserving internal punctuation."""
    left = _TRAILING_ELLIPSIS.sub("", left).rstrip()
    right = _LEADING_ELLIPSIS.sub("", right).lstrip()
    cjk = r"[\u3400-\u9fff\u3040-\u30ff]"
    no_space = bool(
        left and right and re.fullmatch(cjk, left[-1]) and re.fullmatch(cjk, right[0])
    )
    no_space = no_space or bool(
        right and right[0] in _TERMINAL | frozenset(",;:、，；：")
    )
    separator = "" if no_space else " "
    return f"{left}{separator}{right}".strip()


def completed_sentences(text: str):
    """Return complete sentences and an unfinished suffix; ellipses are not stops."""
    sentences = []
    start = index = 0
    while index < len(text):
        character = text[index]
        if character not in _TERMINAL:
            index += 1
            continue
        if character == ".":
            end = index + 1
            while end < len(text) and text[end] == ".":
                end += 1
            if end - index > 1:
                index = end
                continue
            before, after = text[index - 1 : index], text[index + 1 : index + 2]
            token = re.search(r"[\w.]+$", text[start:index])
            word = token.group() if token else ""
            if (
                (before.isdigit() and after.isdigit())
                or word.lower() in _ABBREVIATIONS
                or (len(word) == 1 and word.isupper())
                or (after.islower() and before.isalpha())
            ):
                index += 1
                continue
        end = index + 1
        while end < len(text) and text[end] in "!?！？":
            end += 1
        while end < len(text) and text[end] in _CLOSING:
            end += 1
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = index = end

    pending = text[start:].strip()
    if len(pending) > _SOFT_LIMIT:
        extra, pending = split_long_run(pending)
        sentences.extend(extra)
    return sentences, pending


@dataclass(frozen=True)
class Sentence:
    text: str
    language: str
    confidence: float | None


class SentenceBuffer:
    """Owned by the UI thread; accepts confirmed chunks, never ASR partials."""

    def __init__(self, max_chars=2400):
        self.max_chars = max_chars
        self.clear()

    def clear(self):
        self.pending = ""
        self.language = None
        self.confidence = None

    def flush(self):
        if not self.pending:
            return []
        sentence = Sentence(self.pending, self.language or "?", self.confidence)
        self.clear()
        return [sentence]

    def push(self, text: str, language: str, confidence: float | None = None):
        text = text.strip()
        if not text:
            return []
        ready = []
        if self.pending and language != self.language:
            ready.extend(self.flush())
        had_pending = bool(self.pending)
        self.language = language
        self.confidence = (
            min(self.confidence, confidence)
            if had_pending and self.confidence is not None and confidence is not None
            else None
            if had_pending
            else confidence
        )
        combined = join_fragments(self.pending, text)
        sentences, self.pending = completed_sentences(combined)
        ready.extend(
            Sentence(sentence, language, self.confidence) for sentence in sentences
        )
        if not self.pending:
            self.clear()
        elif len(self.pending) >= self.max_chars:
            # Punctuation-free speech must not grow unbounded or exceed the
            # translation engine's request size. A natural pause normally flushes first.
            ready.extend(self.flush())
        return ready

    def preview(self, text: str, language: str):
        if self.pending:
            if language != self.language:
                return self.pending, self.language
            return join_fragments(self.pending, text), self.language
        return text, language
