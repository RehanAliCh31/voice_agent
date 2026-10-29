"""The model files the assistant runs locally, and where they come from.

Four ONNX files, about 6 MB in all, kept in models/ (not committed). Each is
checked against a pinned SHA-256 before use, so a file that changed upstream
or arrived damaged is refused rather than run.

* openWakeWord's three, from its v0.5.1 GitHub release: the melspectrogram
  and speech-embedding front end, and the wake word itself. Their licence is
  CC BY-NC-SA 4.0 -- non-commercial use only.
* Silero VAD 6.2.3 (MIT), taken from its own release on PyPI, whose hash
  PyPI publishes. Only the ONNX file is extracted; none of the package's
  code is installed or run.
"""

from __future__ import annotations

import hashlib
import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from voicebot.assistant.config import AssistantConfig
from voicebot.utils.logging import get_logger

log = get_logger("assistant.models")

OPENWAKEWORD_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
SILERO_VERSION = "6.2.3"
SILERO_MEMBER = "silero_vad/data/silero_vad.onnx"

# What each file was when it was downloaded and verified for this project.
SHA256 = {
    "melspectrogram.onnx": "ba2b0e0f8b7b875369a2c89cb13360ff53bac436f2895cced9f479fa65eb176f",
    "embedding_model.onnx": "70d164290c1d095d1d4ee149bc5e00543250a7316b59f31d056cff7bd3075c1f",
    "hey_jarvis_v0.1.onnx": "94a13cfe60075b132f6a472e7e462e8123ee70861bc3fb58434a73712ee0d2cb",
    "silero_vad.onnx": "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3",
}


@dataclass(frozen=True)
class ModelFile:
    path: Path
    present: bool
    verified: bool


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wanted(config: AssistantConfig) -> list[Path]:
    folder = Path(config.wake_word.models_dir)
    return [
        folder / "melspectrogram.onnx",
        folder / "embedding_model.onnx",
        folder / f"{config.wake_word.model}.onnx",
        Path(config.vad.model_path),
    ]


def status(config: AssistantConfig) -> list[ModelFile]:
    out = []
    for path in wanted(config):
        present = path.is_file()
        expected = SHA256.get(path.name)
        # A custom wake word has no pinned hash; being there is all we can check.
        verified = present and (expected is None or _sha256(path.read_bytes()) == expected)
        out.append(ModelFile(path, present, verified))
    return out


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return bytes(response.read())


def _save(path: Path, data: bytes) -> None:
    expected = SHA256.get(path.name)
    if expected is not None and _sha256(data) != expected:
        raise RuntimeError(f"{path.name} does not match its pinned SHA-256; refusing to use it")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log.info("saved %s (%.1f MB)", path, len(data) / 1e6)


def download(config: AssistantConfig) -> list[ModelFile]:
    """Fetch whatever is missing or fails its check. Safe to run again."""
    current = {m.path: m for m in status(config)}
    folder = Path(config.wake_word.models_dir)
    for name in ("melspectrogram.onnx", "embedding_model.onnx", f"{config.wake_word.model}.onnx"):
        if not current[folder / name].verified:
            _save(folder / name, _fetch(f"{OPENWAKEWORD_RELEASE}/{name}"))
    vad = Path(config.vad.model_path)
    if not current[vad].verified:
        meta = json.loads(_fetch(f"https://pypi.org/pypi/silero-vad/{SILERO_VERSION}/json", 15))
        wheel = next(f for f in meta["urls"] if f["filename"].endswith(".whl"))
        data = _fetch(wheel["url"])
        if _sha256(data) != wheel["digests"]["sha256"]:
            raise RuntimeError("the Silero VAD release does not match the hash PyPI publishes")
        _save(vad, zipfile.ZipFile(io.BytesIO(data)).read(SILERO_MEMBER))
    return status(config)
