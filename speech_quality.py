"""Pure helpers for ordered inference work and conservative speech evidence."""

from __future__ import annotations

import math
from dataclasses import dataclass

MIN_ROLLING_PAUSE_SECONDS = 0.12


def rolling_audio_boundary(
    total_samples,
    speech_offset,
    target_seconds,
    pause_cut,
    pause_seconds,
    sample_rate=16000,
):
    """Prefer a recent acoustic pause, with a bounded wait before a hard cut.

    The returned index partitions existing audio; callers retain everything
    after it. A clearer pause can also end an unpunctuated spoken utterance.
    """
    speech_samples = total_samples - speech_offset
    if speech_samples < target_seconds * sample_rate:
        return None
    if (
        pause_cut is not None
        and pause_seconds >= MIN_ROLLING_PAUSE_SECONDS
        and pause_cut - speech_offset >= 2 * sample_rate
        and 0 <= total_samples - pause_cut <= 0.8 * sample_rate
    ):
        return pause_cut, pause_seconds >= 0.35
    if speech_samples >= (target_seconds + 0.6) * sample_rate:
        return total_samples, False
    return None


@dataclass(frozen=True)
class LanguageEvidence:
    captured_at: float
    probabilities: dict[str, float]


def coalesce_inference_work(items):
    """Keep control events FIFO; retain the latest detection before each commit.

    A partial is expendable, but a commit/reset/stop must never be moved to
    the back of the queue. Detection must not starve behind slow decoding.
    """
    result = []
    latest = {}
    for index, item in enumerate(items):
        kind = item[0]
        if kind in ("partial", "detect"):
            latest[kind] = (index, item)
            continue
        if kind == "commit" and "detect" in latest:
            result.append(latest["detect"][1])
        latest.clear()
        result.append(item)
        if kind == "stop":
            return result
    result.extend(
        item for _, item in sorted(latest.values(), key=lambda entry: entry[0])
    )
    return result


def _probabilities(values):
    clean = {
        key: float(value)
        for key, value in values.items()
        if math.isfinite(value) and value > 0
    }
    total = sum(clean.values())
    return {key: value / total for key, value in clean.items()} if total else {}


def fuse_language_probabilities(whisper, speechbrain=None):
    """Pool distributions; strong disagreement stays ambiguous, not a coin flip."""
    whisper = _probabilities(whisper)
    if not whisper:
        return None, 0.0, 0.0, False
    wh_lang = max(whisper, key=whisper.get)
    pooled = whisper
    agreement = False
    if speechbrain:
        # Do not renormalize a tiny supported slice into strong evidence.
        supported = {
            lang: prob for lang, prob in speechbrain.items() if lang in whisper
        }
        if sum(supported.values()) >= 0.75:
            sb = _probabilities(supported)
            sb_lang = max(sb, key=sb.get)
            sb_conf, wh_conf = sb[sb_lang], whisper[wh_lang]
            agreement = sb_lang == wh_lang and sb_conf >= 0.65
            if sb_conf >= 0.65:
                weight = 0.6
                if sb_lang != wh_lang:
                    weight = 0.75 if wh_conf < 0.6 and sb_conf >= 0.9 else 0.5
                    if wh_conf >= 0.85 and sb_conf < 0.85:
                        weight = 0.0
                pooled = {
                    lang: (1 - weight) * prob + weight * sb.get(lang, 0.0)
                    for lang, prob in whisper.items()
                }
    ranked = sorted(pooled, key=pooled.get, reverse=True)
    language = ranked[0]
    confidence = pooled[language]
    margin = confidence - (pooled[ranked[1]] if len(ranked) > 1 else 0.0)
    return language, confidence, margin, agreement


def closest_language_evidence(history, captured_at, reset_at, max_age=6.0):
    """Use evidence from this audio window, never a later speaker or old session."""
    candidates = [
        item
        for item in history
        if item.captured_at >= reset_at
        and 0 <= captured_at - item.captured_at <= max_age
    ]
    return max(candidates, key=lambda item: item.captured_at) if candidates else None


def segment_confidence_reason(segment):
    try:
        logprob = float(segment.get("avg_logprob", 0.0))
        silence = float(segment.get("no_speech_prob", 0.0))
        compression = float(segment.get("compression_ratio", 0.0))
    except TypeError, ValueError:
        return "invalid_scores"
    if not all(math.isfinite(value) for value in (logprob, silence, compression)):
        return "invalid_scores"
    # A high no-speech score alone must not discard confident recognized speech.
    if silence > 0.7 and logprob < -0.5:
        return "no_speech"
    if logprob < -0.95:
        return "low_confidence"
    if compression > 2.4:
        return "repetition"
    return None
