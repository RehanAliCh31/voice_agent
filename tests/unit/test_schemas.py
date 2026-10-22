import numpy as np
import pytest

from voicebot.schemas import (
    AudioBuffer,
    DialogueState,
    Segment,
    Slot,
    Transcript,
    Understanding,
)


def test_audio_buffer_reports_duration():
    buf = AudioBuffer(samples=np.zeros(16000, dtype=np.float32), sample_rate=16000)
    assert buf.duration_s == pytest.approx(1.0)


def test_audio_buffer_rejects_stereo():
    with pytest.raises(ValueError, match="mono"):
        AudioBuffer(samples=np.zeros((2, 100), dtype=np.float32), sample_rate=16000)


def test_audio_buffer_rejects_non_positive_sample_rate():
    with pytest.raises(ValueError, match="sample_rate"):
        AudioBuffer(samples=np.zeros(10, dtype=np.float32), sample_rate=0)


def test_transcript_holds_segments():
    t = Transcript(
        text="hello",
        confidence=0.9,
        duration_s=1.0,
        segments=(Segment(text="hello", start_s=0.0, end_s=1.0),),
    )
    assert t.segments[0].text == "hello"


def test_understanding_defaults_to_in_scope_with_no_slots():
    u = Understanding(intent="balance", intent_confidence=0.95)
    assert u.slots == ()
    assert u.is_out_of_scope is False


def test_slot_records_character_span():
    s = Slot(name="amount", value="$50", start=9, end=12)
    assert (s.start, s.end) == (9, 12)


def test_dialogue_state_defaults_are_independent_between_instances():
    a = DialogueState(session_id="a")
    b = DialogueState(session_id="b")
    a.filled_slots["amount"] = "$50"
    assert b.filled_slots == {}
