"""The data contracts the pipeline passes between stages."""

import numpy as np
import pytest

from voicebot.schemas import AudioBuffer, Segment, Transcript


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
