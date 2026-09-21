"""Explicit One Voice -> identity evidence -> ASR, plus a fair raw ASR baseline."""

import time

import numpy as np

from poc.common import RATE, merge_spans, overlap
from poc.models import ASR_SETTINGS

# Selected by a fixed grid on 32 public development cases (poc.calibrate).
# These are similarity cutoffs, not calibrated probabilities or authentication.
GATE = {
    "version": "public-dev32-v1",
    "accepted_extracted": 0.30,
    "accepted_original": 0.10,
    "uncertain_extracted": 0.20,
    "minimum_speech_seconds": 0.60,
    "word_overlap": 0.80,
}


def classify(extracted, original, speech_seconds, settings=GATE):
    if speech_seconds < settings["minimum_speech_seconds"]:
        return "uncertain"
    if extracted >= settings["accepted_extracted"] and original >= settings["accepted_original"]:
        return "accepted"
    if extracted >= settings["uncertain_extracted"]:
        return "uncertain"
    return "excluded"


def attribute(words, windows, speech_spans, duration):
    accepted = merge_spans(
        [[w["start"], w["end"]] for w in windows if w["attribution"] == "accepted"]
    )
    possible = merge_spans(
        [[w["start"], w["end"]] for w in windows if w["attribution"] != "excluded"]
    )
    tagged = []
    for word in words:
        start, end = max(0.0, word["start"]), min(duration, word["end"])
        if end <= start:
            continue
        length = end - start
        if overlap(start, end, speech_spans) / length < GATE["word_overlap"]:
            state = "excluded"
        elif overlap(start, end, accepted) / length >= GATE["word_overlap"]:
            state = "accepted"
        elif overlap(start, end, possible) > 0:
            state = "uncertain"
        else:
            state = "excluded"
        tagged.append({**word, "start": start, "end": end, "attribution": state})
    segments = []
    for word in tagged:
        if (
            segments
            and segments[-1]["attribution"] == word["attribution"]
            and word["start"] - segments[-1]["end"] < 0.75
            and word["end"] - segments[-1]["start"] < 7
        ):
            segments[-1]["end"] = word["end"]
            segments[-1]["text"] += word["text"]
            segments[-1]["words"].append(word)
        else:
            segments.append(
                {
                    "start": word["start"],
                    "end": word["end"],
                    "text": word["text"],
                    "attribution": word["attribution"],
                    "words": [word],
                }
            )
    return segments


def run(runtime, mixture, reference, compare=True, progress=lambda _: None):
    started = time.monotonic()
    timings = {}

    def phase(name):
        progress(name)
        timings[name] = round(time.monotonic() - started, 3)

    duration = len(mixture) / RATE
    if not 3 <= len(reference) / RATE <= 10:
        raise ValueError("Choose a clean reference lasting 3–10 seconds.")
    phase("Checking the reference")
    reference_spans = runtime.speech(reference)
    if sum(b - a for a, b in reference_spans) < 1.5:
        raise ValueError(
            "The reference needs at least 1.5 seconds of clear speech. Choose another sample."
        )
    if float(np.mean(np.abs(reference) >= 0.995)) > 0.02:
        raise ValueError("The reference is heavily clipped. Choose a cleaner recording.")
    phase("Finding speech")
    original_speech = runtime.speech(mixture)
    result = {
        "duration": duration,
        "sample_rate": RATE,
        "segments": [],
        "windows": [],
        "speech": original_speech,
        "outcome": "no_speech",
        "comparison": None,
        "gate": dict(GATE),
        "model_manifest": runtime.manifest,
        "asr_settings": ASR_SETTINGS,
        "timings": timings,
        "identity_notice": "Heuristic voice matching, not verified identity. Review attributed text.",
        "pipeline": [
            "Silero speech detection",
            "One Voice waveform extraction",
            "ECAPA similarity",
            "faster-whisper transcription",
            "timestamp attribution",
        ],
    }
    if not original_speech:
        result["processing_seconds"] = round(time.monotonic() - started, 3)
        return result, np.zeros_like(mixture)
    phase("Extracting with One Voice")
    estimate = runtime.extract(mixture, reference)
    if len(estimate) != len(mixture) or not np.isfinite(estimate).all():
        raise ValueError("One Voice returned invalid audio.")
    runtime.release_separator()
    phase("Checking the selected speaker")
    extracted_speech = runtime.speech(estimate)
    reference_audio = np.concatenate(
        [reference[int(a * RATE) : int(b * RATE)] for a, b in reference_spans]
    )
    embedding = runtime.embedding(reference_audio)
    windows = []
    for start in np.arange(0, duration, 0.75):
        end = min(duration, float(start) + 0.75)
        left, right = max(0, float(start) - 1.125), min(duration, end + 1.125)
        speech_seconds = min(
            overlap(left, right, original_speech), overlap(left, right, extracted_speech)
        )
        original_score = extracted_score = 0.0
        state = "excluded"
        if speech_seconds >= 0.35:
            original_score = float(
                np.dot(embedding, runtime.embedding(mixture[int(left * RATE) : int(right * RATE)]))
            )
            extracted_score = float(
                np.dot(embedding, runtime.embedding(estimate[int(left * RATE) : int(right * RATE)]))
            )
            state = classify(extracted_score, original_score, speech_seconds)
        windows.append(
            {
                "start": round(float(start), 3),
                "end": round(end, 3),
                "context_start": round(left, 3),
                "context_end": round(right, 3),
                "original_similarity": round(original_score, 4),
                "extracted_similarity": round(extracted_score, 4),
                "speech_seconds": round(speech_seconds, 3),
                "attribution": state,
            }
        )
    phase("Transcribing the One Voice output")
    # Whole-clip decoding preserves context and timestamps. The identity filter
    # labels words afterwards; it never alters audio or feeds identity into ASR.
    extracted_words = runtime.transcribe(estimate) if extracted_speech else []
    valid_speech = [
        [max(a, c), min(b, d)]
        for a, b in original_speech
        for c, d in extracted_speech
        if min(b, d) > max(a, c)
    ]
    segments = attribute(extracted_words, windows, valid_speech, duration)
    result.update(segments=segments, windows=windows, extracted_speech=extracted_speech)
    if compare:
        phase("Transcribing the original for comparison")
        original_words = runtime.transcribe(mixture)
        result["comparison"] = {
            "raw": {
                "text": "".join(w["text"] for w in original_words).strip(),
                "words": original_words,
            },
            "one_voice": {
                "text": "".join(w["text"] for w in extracted_words).strip(),
                "words": extracted_words,
            },
            "same_asr_settings": True,
            "reference_supplied_to_asr": False,
            "note": "Both use the same ASR model and settings. Only One Voice receives the reference and separates audio.",
        }
    result["outcome"] = (
        "matched" if any(s["attribution"] == "accepted" for s in segments) else "no_confident_match"
    )
    result["coverage"] = {
        "accepted_words": sum(len(s["words"]) for s in segments if s["attribution"] == "accepted"),
        "uncertain_words": sum(
            len(s["words"]) for s in segments if s["attribution"] == "uncertain"
        ),
        "excluded_words": sum(len(s["words"]) for s in segments if s["attribution"] == "excluded"),
    }
    result["processing_seconds"] = round(time.monotonic() - started, 3)
    return result, estimate
