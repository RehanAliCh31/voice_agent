"""Voice activity: where speech starts and, above all, where it ends.

Silero VAD, run as its ONNX model on onnxruntime -- no PyTorch. Measured on
speech from both voices: speech is found within 20-90 ms of its start, and
digital silence, white noise at -35 and -20 dBFS and a 150 ms click are all
scored as no speech at all.

`Endpointer` turns those per-frame probabilities into utterances, and is
pure logic, so it is tested with probabilities written by hand:

* a pre-roll of the frames just before speech is kept, so the first
  syllable is not cut off;
* speech must last `min_speech_ms` before it counts, so a cough does not;
* it ends after `end_silence_ms` of silence, or at `max_utterance_s`;
* if nothing is said within the wait, it says so.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from voicebot.assistant.config import VadConfig

FRAME = 512  # samples: 32 ms at 16 kHz, the window Silero was trained on
FRAME_MS = 32
_CONTEXT = 64  # samples of the previous frame Silero sees with each new one
KEEP_TRAILING_MS = 200  # of the silence that ended an utterance


class SileroVad:
    """The probability that a 32 ms frame is speech."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"VAD model not found at {path}. Run: python -m voicebot.assistant setup"
            )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._rate = np.array(16000, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        if frame.shape != (FRAME,):
            raise ValueError(f"a VAD frame is {FRAME} samples, not {frame.shape}")
        x = (
            frame.astype(np.float32) / 32768
            if frame.dtype == np.int16
            else frame.astype(np.float32)
        )
        chunk = np.concatenate([self._context, x[None, :]], axis=1)
        prob, self._state = self._session.run(
            None, {"input": chunk, "state": self._state, "sr": self._rate}
        )
        self._context = chunk[:, -_CONTEXT:]
        return float(prob[0, 0])


@dataclass(frozen=True)
class Utterance:
    audio: np.ndarray  # int16, 16 kHz, mono
    speech_ms: int
    ended_by: Literal["silence", "max_length"]


@dataclass(frozen=True)
class SpeechStarted:
    pass


@dataclass(frozen=True)
class UtteranceEnded:
    utterance: Utterance


@dataclass(frozen=True)
class NoSpeech:
    pass


EndpointEvent = SpeechStarted | UtteranceEnded | NoSpeech


class Endpointer:
    def __init__(self, config: VadConfig) -> None:
        self.config = config
        self._phase: Literal["waiting", "speech", "done"] = "done"
        self._pre: deque[np.ndarray] = deque(
            maxlen=max(0, math.ceil(config.pre_roll_ms / FRAME_MS))
        )
        self._candidate: list[np.ndarray] = []
        self._audio: list[np.ndarray] = []
        self._voiced_ms = self._waited_ms = self._speech_ms = self._silence_ms = 0
        self._wait_ms = 0.0
        self._min_speech_ms = config.min_speech_ms

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def speech_ms(self) -> int:
        """How much of the utterance so far was voiced."""
        return self._speech_ms if self._phase == "speech" else self._voiced_ms

    @property
    def holding(self) -> bool:
        """Whether frames that may yet start speech are held, waiting to see if they do."""
        return self._phase == "waiting" and bool(self._candidate)

    @property
    def silence_ms(self) -> int:
        """How long the silence after the last voiced frame has lasted."""
        return self._silence_ms

    def heard_so_far(self) -> np.ndarray:
        """The utterance so far, pre-roll and silence included, while it goes on."""
        if self._phase != "speech" or not self._audio:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._audio).astype(np.int16)

    def start(self, wait_s: float | None = None, min_speech_ms: int | None = None) -> None:
        """Begin listening for speech. `wait_s` None waits for ever."""
        self._phase = "waiting"
        self._pre.clear()
        self._candidate = []
        self._audio = []
        self._voiced_ms = self._waited_ms = self._speech_ms = self._silence_ms = 0
        self._wait_ms = math.inf if wait_s is None else wait_s * 1000
        self._min_speech_ms = self.config.min_speech_ms if min_speech_ms is None else min_speech_ms

    def stop(self) -> None:
        self._phase = "done"

    def process(self, frame: np.ndarray, prob: float) -> EndpointEvent | None:
        if self._phase == "waiting":
            return self._waiting(frame, prob)
        if self._phase == "speech":
            return self._speaking(frame, prob)
        return None

    def _waiting(self, frame: np.ndarray, prob: float) -> EndpointEvent | None:
        cfg = self.config
        self._waited_ms += FRAME_MS
        if prob >= cfg.threshold:
            self._candidate.append(frame)
            self._voiced_ms += FRAME_MS
            if self._voiced_ms >= self._min_speech_ms:
                self._phase = "speech"
                self._audio = list(self._pre) + self._candidate
                self._speech_ms = self._voiced_ms
                self._silence_ms = 0
                return SpeechStarted()
        elif prob < cfg.neg_threshold:
            # Too short to be speech: what was held back becomes pre-roll.
            for held in self._candidate:
                self._pre.append(held)
            self._candidate = []
            self._voiced_ms = 0
            self._pre.append(frame)
        elif self._candidate:
            self._candidate.append(frame)
        else:
            self._pre.append(frame)
        if self._waited_ms >= self._wait_ms:
            self._phase = "done"
            return NoSpeech()
        return None

    def _speaking(self, frame: np.ndarray, prob: float) -> EndpointEvent | None:
        cfg = self.config
        self._audio.append(frame)
        if prob >= cfg.threshold:
            self._speech_ms += FRAME_MS
            self._silence_ms = 0
        elif prob < cfg.neg_threshold:
            self._silence_ms += FRAME_MS
        if self._silence_ms >= cfg.end_silence_ms:
            drop = max(0, (self._silence_ms - KEEP_TRAILING_MS) // FRAME_MS)
            return self._finish("silence", drop)
        if len(self._audio) * FRAME_MS >= cfg.max_utterance_s * 1000:
            return self._finish("max_length", 0)
        return None

    def _finish(self, ended_by: Literal["silence", "max_length"], drop: int) -> UtteranceEnded:
        frames = self._audio[: len(self._audio) - drop] if drop else self._audio
        self._phase = "done"
        audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.int16)
        return UtteranceEnded(Utterance(audio.astype(np.int16), self._speech_ms, ended_by))
