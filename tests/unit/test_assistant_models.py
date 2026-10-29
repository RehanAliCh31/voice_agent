"""The model files: fetched once, checked against pinned hashes, refused if changed."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from voicebot.assistant import models
from voicebot.assistant.config import load_assistant_config

# --- the model files -------------------------------------------------------------------


def _config_in(tmp_path: Path):  # type: ignore[no-untyped-def]
    config = load_assistant_config()
    config.wake_word.models_dir = str(tmp_path / "oww")
    config.vad.model_path = str(tmp_path / "silero" / "silero_vad.onnx")
    return config


def test_missing_and_damaged_files_are_told_apart(tmp_path: Path) -> None:
    config = _config_in(tmp_path)
    (tmp_path / "oww").mkdir()
    (tmp_path / "oww" / "melspectrogram.onnx").write_bytes(b"not the real file")
    found = {m.path.name: (m.present, m.verified) for m in models.status(config)}
    assert found["melspectrogram.onnx"] == (True, False)
    assert found["hey_jarvis_v0.1.onnx"] == (False, False)


def test_download_saves_only_files_that_match_their_pinned_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_in(tmp_path)
    content = {name: f"model {name}".encode() for name in models.SHA256}
    monkeypatch.setattr(
        models, "SHA256", {n: hashlib.sha256(c).hexdigest() for n, c in content.items()}
    )
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(models.SILERO_MEMBER, content["silero_vad.onnx"])
    wheel_bytes = wheel.getvalue()
    meta = {
        "urls": [
            {
                "filename": "silero_vad-6.2.3-py3-none-any.whl",
                "url": "https://files/wheel",
                "digests": {"sha256": hashlib.sha256(wheel_bytes).hexdigest()},
            }
        ]
    }

    def fetch(url: str, timeout: float = 60.0) -> bytes:
        if url.endswith("/json"):
            return json.dumps(meta).encode()
        if url == "https://files/wheel":
            return wheel_bytes
        return content[url.rsplit("/", 1)[1]]

    monkeypatch.setattr(models, "_fetch", fetch)
    assert all(m.verified for m in models.download(config))


def test_a_file_that_changed_upstream_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config_in(tmp_path)
    monkeypatch.setattr(models, "_fetch", lambda url, timeout=60.0: b"something else entirely")
    with pytest.raises(RuntimeError, match="pinned SHA-256"):
        models.download(config)
    assert not (tmp_path / "oww" / "melspectrogram.onnx").exists()
