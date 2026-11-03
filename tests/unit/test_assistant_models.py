"""The model files: fetched once, checked against pinned hashes, refused if changed."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from voicebot.assistant import models

# --- the model files -------------------------------------------------------------------


def test_missing_and_damaged_files_are_told_apart(tmp_path: Path) -> None:
    path = tmp_path / "silero" / "silero_vad.onnx"
    assert [(m.present, m.verified) for m in models.status(path)] == [(False, False)]
    path.parent.mkdir()
    path.write_bytes(b"not the real file")
    assert [(m.present, m.verified) for m in models.status(path)] == [(True, False)]


def test_download_saves_only_a_file_that_matches_its_pinned_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"the model"
    monkeypatch.setattr(models, "SHA256", {"silero_vad.onnx": hashlib.sha256(content).hexdigest()})
    wheel = io.BytesIO()
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr(models.SILERO_MEMBER, content)
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
        return json.dumps(meta).encode() if url.endswith("/json") else wheel_bytes

    monkeypatch.setattr(models, "_fetch", fetch)
    path = tmp_path / "silero" / "silero_vad.onnx"
    assert all(m.verified for m in models.download(path))
    assert path.read_bytes() == content


def test_a_file_that_changed_upstream_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(models, "_fetch", lambda url, timeout=60.0: b"something else entirely")
    path = tmp_path / "silero" / "silero_vad.onnx"
    with pytest.raises(RuntimeError):
        models.download(path)
    assert not path.exists()
