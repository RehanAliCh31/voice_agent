import numpy as np
import pytest

from voicebot.pipeline.audio_io import (
    decode_wav,
    encode_wav,
    is_silent,
    normalise,
    resample,
    to_mono,
    trim_silence,
)
from voicebot.schemas import AudioBuffer


def _tone(freq: float = 440.0, seconds: float = 0.5, rate: int = 48_000) -> AudioBuffer:
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False, dtype=np.float32)
    return AudioBuffer(samples=np.sin(2 * np.pi * freq * t).astype(np.float32), sample_rate=rate)


# --- resampling --------------------------------------------------------------


def test_resample_changes_the_rate_and_length_proportionally():
    source = _tone(rate=48_000, seconds=1.0)
    out = resample(source, 16_000)
    assert out.sample_rate == 16_000
    assert out.duration_s == pytest.approx(1.0, abs=0.01)
    assert len(out.samples) == pytest.approx(16_000, abs=10)


def test_resample_is_a_no_op_at_the_same_rate():
    source = _tone(rate=16_000)
    assert resample(source, 16_000) is source


def test_resample_preserves_the_tone_rather_than_aliasing_it():
    """A 440 Hz tone must still peak near 440 Hz after downsampling."""
    out = resample(_tone(freq=440.0, seconds=1.0, rate=48_000), 16_000)
    spectrum = np.abs(np.fft.rfft(out.samples))
    peak_hz = np.fft.rfftfreq(len(out.samples), 1 / out.sample_rate)[int(np.argmax(spectrum))]
    assert peak_hz == pytest.approx(440.0, abs=5.0)


def test_resample_handles_an_awkward_ratio():
    out = resample(_tone(rate=44_100, seconds=0.5), 16_000)
    assert out.sample_rate == 16_000
    assert out.duration_s == pytest.approx(0.5, abs=0.01)


# --- channels and level ------------------------------------------------------


def test_to_mono_averages_stereo():
    stereo = np.array([[1.0, -1.0], [0.5, 0.5]], dtype=np.float32)  # (frames, channels)
    assert to_mono(stereo).tolist() == pytest.approx([0.0, 0.5])


def test_to_mono_leaves_mono_alone():
    mono = np.array([0.1, 0.2], dtype=np.float32)
    assert to_mono(mono).tolist() == pytest.approx([0.1, 0.2])


def test_normalise_brings_the_peak_to_the_target():
    quiet = AudioBuffer(samples=np.array([0.01, -0.02], dtype=np.float32), sample_rate=16_000)
    out = normalise(quiet, peak=0.95)
    assert float(np.max(np.abs(out.samples))) == pytest.approx(0.95, abs=1e-5)


def test_normalise_leaves_pure_silence_alone():
    silence = AudioBuffer(samples=np.zeros(10, dtype=np.float32), sample_rate=16_000)
    assert float(np.max(np.abs(normalise(silence).samples))) == 0.0


# --- WAV round trip ----------------------------------------------------------


def test_wav_round_trip_preserves_rate_length_and_shape():
    source = _tone(seconds=0.1, rate=16_000)
    restored = decode_wav(encode_wav(source))
    assert restored.sample_rate == source.sample_rate
    assert len(restored.samples) == len(source.samples)
    # 16-bit quantisation, so compare with that tolerance
    assert np.allclose(restored.samples, source.samples, atol=1e-3)


def test_encode_produces_a_riff_header():
    data = encode_wav(_tone(seconds=0.01, rate=16_000))
    assert data[:4] == b"RIFF"
    assert data[8:12] == b"WAVE"


def test_samples_above_full_scale_are_clipped_not_wrapped():
    """Without clipping, 1.5 would wrap to a large negative value: a loud click."""
    loud = AudioBuffer(samples=np.array([1.5, -1.5], dtype=np.float32), sample_rate=16_000)
    restored = decode_wav(encode_wav(loud))
    assert restored.samples[0] > 0.99
    assert restored.samples[1] < -0.99


def test_decode_rejects_unsupported_bit_depth():
    import io
    import wave

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(1)  # 8-bit
        handle.setframerate(16_000)
        handle.writeframes(b"\x00" * 10)
    with pytest.raises(ValueError, match="16-bit"):
        decode_wav(buffer.getvalue())


# --- silence -----------------------------------------------------------------


def test_trim_silence_removes_leading_and_trailing_quiet():
    rate = 16_000
    quiet = np.zeros(rate, dtype=np.float32)
    loud = np.ones(rate // 2, dtype=np.float32) * 0.5
    padded = AudioBuffer(samples=np.concatenate([quiet, loud, quiet]), sample_rate=rate)
    out = trim_silence(padded)
    assert len(out.samples) < len(padded.samples)
    assert out.duration_s == pytest.approx(0.7, abs=0.05)  # 0.5s speech + 0.1s pad either side


def test_trim_silence_on_pure_silence_returns_nothing():
    silence = AudioBuffer(samples=np.zeros(16_000, dtype=np.float32), sample_rate=16_000)
    assert len(trim_silence(silence).samples) == 0


def test_is_silent_detects_an_empty_recording():
    assert is_silent(AudioBuffer(samples=np.zeros(100, dtype=np.float32), sample_rate=16_000))
    assert not is_silent(_tone(seconds=0.1, rate=16_000))
