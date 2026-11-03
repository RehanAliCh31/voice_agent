"""Finding where speech starts and ends, on probabilities written by hand.

Each frame below is filled with its own index, so what an utterance contains
-- and in what order -- can be read straight off it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voicebot.assistant.config import VadConfig
from voicebot.assistant.vad import (
    FRAME,
    Endpointer,
    NoSpeech,
    SileroVad,
    SpeechStarted,
    UtteranceEnded,
)

# 3 frames of pre-roll, 3 voiced frames to count as speech, 10 silent to end.
CONFIG = VadConfig(
    pre_roll_ms=96,
    min_speech_ms=96,
    end_silence_ms=320,
    max_utterance_s=1.0,
)
SPEECH, QUIET, UNSURE = 0.9, 0.1, 0.4


def run(probs: list[float], endpointer: Endpointer | None = None):  # type: ignore[no-untyped-def]
    endpointer = endpointer or Endpointer(CONFIG)
    if endpointer.phase == "done":
        endpointer.start()
    events = []
    for i, p in enumerate(probs):
        event = endpointer.process(np.full(FRAME, i, dtype=np.int16), p)
        if event is not None:
            events.append((i, event))
    return events


def frames_in(event: UtteranceEnded) -> list[int]:
    audio = event.utterance.audio
    return [int(audio[i]) for i in range(0, len(audio), FRAME)]


def test_speech_keeps_its_pre_roll_and_ends_after_the_silence() -> None:
    probs = [QUIET] * 5 + [SPEECH] * 4 + [QUIET] * 10
    events = run(probs)
    assert isinstance(events[0][1], SpeechStarted) and events[0][0] == 7  # third voiced frame
    index, ended = events[1]
    assert index == 18 and isinstance(ended, UtteranceEnded)
    # Pre-roll 2,3,4; speech 5-8; then 200 ms of the ten silent frames, rounded
    # up to whole frames: seven of them, 9-15.
    assert frames_in(ended) == [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    assert ended.utterance.speech_ms == 4 * 32 and ended.utterance.ended_by == "silence"


def test_a_click_too_short_to_be_speech_is_ignored() -> None:
    probs = [QUIET] * 3 + [SPEECH] * 2 + [QUIET] * 6
    assert run(probs) == []


def test_a_click_becomes_pre_roll_for_the_speech_after_it() -> None:
    probs = [SPEECH, QUIET, SPEECH, SPEECH, SPEECH] + [QUIET] * 10
    _, ended = run(probs)[1]
    assert frames_in(ended)[:2] == [0, 1]


def test_an_unsure_frame_neither_starts_nor_ends_speech() -> None:
    probs = [SPEECH] * 3 + [UNSURE] * 10 + [QUIET] * 10
    events = run(probs)
    assert isinstance(events[0][1], SpeechStarted)
    index, ended = events[1]
    assert index == 3 + 10 + 9  # only the quiet frames counted towards the end
    assert ended.utterance.ended_by == "silence"


def test_an_utterance_is_cut_at_the_maximum_length() -> None:
    events = run([SPEECH] * 40)
    index, ended = events[1]
    assert ended.utterance.ended_by == "max_length" and index == 31  # 32 frames = 1.024 s


def test_nothing_said_is_reported_once_the_wait_is_over() -> None:
    endpointer = Endpointer(CONFIG)
    endpointer.start(0.5)
    events = run([QUIET] * 30, endpointer)
    assert events == [(15, NoSpeech())]  # 16 frames = 512 ms


def test_waiting_for_ever_never_gives_up() -> None:
    endpointer = Endpointer(CONFIG)
    endpointer.start(None)
    assert run([QUIET] * 500, endpointer) == []


def test_a_stricter_minimum_for_barge_in() -> None:
    endpointer = Endpointer(CONFIG)
    endpointer.start(None, min_speech_ms=320)
    events = run([SPEECH] * 9 + [QUIET] * 3, endpointer)
    assert events == []
    endpointer.start(None, min_speech_ms=320)
    assert isinstance(run([SPEECH] * 10, endpointer)[0][1], SpeechStarted)


def test_once_done_it_does_nothing_until_started_again() -> None:
    endpointer = Endpointer(CONFIG)
    assert endpointer.process(np.zeros(FRAME, np.int16), SPEECH) is None


def test_the_thresholds_must_be_ordered() -> None:
    with pytest.raises(ValueError, match="neg_threshold"):
        VadConfig(threshold=0.3, neg_threshold=0.5)


# --- the real model, where it has been downloaded ------------------------------------

MODEL = Path("models/silero/silero_vad.onnx")
needs_model = pytest.mark.skipif(
    not MODEL.is_file(), reason="run: python -m voicebot.assistant setup"
)


@needs_model
def test_silence_and_noise_are_not_speech() -> None:
    vad = SileroVad(MODEL)
    rng = np.random.default_rng(0)
    silence = [vad(np.zeros(FRAME, np.int16)) for _ in range(30)]
    vad.reset()
    noise = [vad((rng.standard_normal(FRAME) * 500).astype(np.int16)) for _ in range(30)]
    assert max(silence) < 0.1 and max(noise) < 0.5


@needs_model
def test_float_frames_are_accepted_and_wrong_sizes_are_not() -> None:
    vad = SileroVad(MODEL)
    assert 0 <= vad(np.zeros(FRAME, np.float32)) <= 1
    with pytest.raises(ValueError, match="512"):
        vad(np.zeros(480, np.int16))


def test_a_missing_model_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="setup"):
        SileroVad(tmp_path / "absent.onnx")
