"""Audio plumbing: resampling, WAV encoding and decoding.

Deliberately dependency-light. `scipy.signal` arrives with scikit-learn, which
is already required, and the WAV container is simple enough to read and write
with the standard library's `wave` module. Adding soundfile or librosa for
this would be two large dependencies for a few hundred lines of work already
installed.
"""

from __future__ import annotations

import io
import wave

import numpy as np
from scipy import signal  # type: ignore[import-untyped]

from voicebot.schemas import AudioBuffer

TARGET_SAMPLE_RATE = 16_000


def resample(audio: AudioBuffer, target_rate: int = TARGET_SAMPLE_RATE) -> AudioBuffer:
    """Resample to `target_rate`, low-pass filtering as it goes.

    `resample_poly` applies an anti-aliasing filter; naive index-picking or
    linear interpolation does not, and the aliasing it creates lands squarely
    in the frequency band speech recognition cares about.
    """
    if audio.sample_rate == target_rate:
        return audio
    gcd = np.gcd(audio.sample_rate, target_rate)
    up = target_rate // gcd
    down = audio.sample_rate // gcd
    resampled = signal.resample_poly(audio.samples, up, down).astype(np.float32)
    return AudioBuffer(samples=resampled, sample_rate=target_rate)


def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average any channel layout down to one channel."""
    if samples.ndim == 1:
        return samples.astype(np.float32)
    # Accept both (channels, frames) and (frames, channels).
    axis = 0 if samples.shape[0] < samples.shape[1] else 1
    return samples.mean(axis=axis).astype(np.float32)


def normalise(audio: AudioBuffer, peak: float = 0.95) -> AudioBuffer:
    """Scale so the loudest sample sits at `peak`. Silence is returned as-is."""
    loudest = float(np.max(np.abs(audio.samples))) if audio.samples.size else 0.0
    if loudest == 0.0:
        return audio
    scaled = (audio.samples / loudest * peak).astype(np.float32)
    return AudioBuffer(samples=scaled, sample_rate=audio.sample_rate)


def decode_wav(data: bytes) -> AudioBuffer:
    """Read 16-bit PCM WAV bytes into float32 samples in [-1, 1]."""
    with wave.open(io.BytesIO(data), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())

    if width != 2:
        raise ValueError(f"Only 16-bit PCM WAV is supported, got {width * 8}-bit")

    raw = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        raw = raw.reshape(-1, channels)
    return AudioBuffer(samples=to_mono(raw), sample_rate=rate)


def encode_wav(audio: AudioBuffer) -> bytes:
    """Write float32 samples as 16-bit PCM WAV bytes.

    Samples are clipped before conversion: a value above 1.0 would wrap around
    to a large negative integer and arrive as a loud click.
    """
    clipped = np.clip(audio.samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(audio.sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def trim_silence(audio: AudioBuffer, threshold: float = 0.01, pad_ms: int = 100) -> AudioBuffer:
    """Drop leading and trailing quiet, keeping a short pad either side.

    A cheap energy gate, not a speech/noise classifier. It exists so the ASR
    is not handed two seconds of room tone, which costs time and invites
    hallucinated words.
    """
    if audio.samples.size == 0:
        return audio
    loud = np.abs(audio.samples) > threshold
    if not loud.any():
        return AudioBuffer(samples=np.zeros(0, dtype=np.float32), sample_rate=audio.sample_rate)

    pad = int(audio.sample_rate * pad_ms / 1000)
    first = max(0, int(np.argmax(loud)) - pad)
    last = min(len(audio.samples), len(audio.samples) - int(np.argmax(loud[::-1])) + pad)
    return AudioBuffer(samples=audio.samples[first:last], sample_rate=audio.sample_rate)


def is_silent(audio: AudioBuffer, threshold: float = 0.01) -> bool:
    """True when nothing in the buffer rises above the gate."""
    if audio.samples.size == 0:
        return True
    return bool(np.max(np.abs(audio.samples)) <= threshold)
