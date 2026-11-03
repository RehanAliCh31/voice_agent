"""Fault injection: the model file, the endpoint check and the command line.

A download that fails once, one refused outright, a write cut off partway, a
server that lists its models oddly, Ctrl+C, a thread that dies: each ends in
one line and the right exit code, never a traceback or half a file.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import urllib.error
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

from voicebot.assistant import __main__ as cli
from voicebot.assistant import health, models
from voicebot.assistant.config import AssistantConfig
from voicebot.utils.observability import error_counts


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr("voicebot.utils.resilience.time.sleep", slept.append)
    return slept


@pytest.fixture
def logged() -> Iterator[list[logging.LogRecord]]:
    records: list[logging.LogRecord] = []

    class Keep(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Keep()
    logging.getLogger("voicebot").addHandler(handler)
    yield records
    logging.getLogger("voicebot").removeHandler(handler)


# --- the model file, the endpoint check and the command line -----------------------


def test_a_download_that_fails_once_is_retried_and_written_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    data = b"the model"
    monkeypatch.setattr(models, "SHA256", {"silero_vad.onnx": hashlib.sha256(data).hexdigest()})
    calls = []

    def fetch(url: str, timeout: float = 60.0) -> bytes:
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.URLError("connection reset by peer")
        return data

    monkeypatch.setattr(models, "_fetch", fetch)
    path = tmp_path / "silero_vad.onnx"
    models._save(path, models._download("https://release/silero_vad.onnx"))
    assert path.read_bytes() == data and len(calls) == 2
    assert no_sleep == [models.RETRY_PAUSE_S]
    assert [p.name for p in tmp_path.iterdir()] == ["silero_vad.onnx"]  # no temporary left


def test_a_download_refused_outright_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, no_sleep: list[float]
) -> None:
    def fetch(url: str, timeout: float = 60.0) -> bytes:
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    monkeypatch.setattr(models, "_fetch", fetch)
    with pytest.raises(models.ModelDownloadError, match="404"):
        models._download("https://release/missing.onnx")
    assert no_sleep == []


def test_a_write_that_is_interrupted_leaves_the_old_file_and_no_half_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "another_model.onnx"  # no pinned hash
    path.write_bytes(b"the old, whole file")

    def interrupted(source: str, target: Path) -> None:
        raise OSError("the disk is full")

    monkeypatch.setattr(models.os, "replace", interrupted)
    with pytest.raises(OSError):
        models._save(path, b"the new file")
    assert path.read_bytes() == b"the old, whole file"
    assert [p.name for p in tmp_path.iterdir()] == ["another_model.onnx"]


@pytest.mark.parametrize(
    "body", [[{"id": "m"}], "models", {"data": "m"}, {"data": [1, None, {"id": 5}]}]
)
def test_a_model_list_of_another_shape_is_no_models_not_a_crash(body: object) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json=body)

    check = health.check_service(
        "LLM",
        "http://llm.test/v1/chat/completions",
        "m",
        httpx.Client(transport=httpx.MockTransport(handler)),
    )
    assert check.ok and check.warning and "no models listed" in check.detail


def test_setup_that_cannot_download_is_one_line_and_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def down() -> list[models.ModelFile]:
        raise models.ModelDownloadError("could not download https://release/x: timed out")

    monkeypatch.setattr(cli.models, "download", down)
    assert cli.main(["setup"]) == 1
    out = capsys.readouterr().out
    assert "Could not set up the models: could not download" in out and "Traceback" not in out


def test_ctrl_c_during_the_check_exits_cleanly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupted(config: AssistantConfig) -> list[health.Check]:
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.health, "check_all", interrupted)
    hook = threading.excepthook
    assert cli.main(["check"]) == 130
    assert "Stopped." in capsys.readouterr().out
    assert threading.excepthook is hook  # put back as it was found


def test_a_thread_that_dies_is_logged_and_counted(
    monkeypatch: pytest.MonkeyPatch, logged: list[logging.LogRecord]
) -> None:
    def boom() -> None:
        raise ValueError("a bug")

    monkeypatch.setattr(threading, "excepthook", cli._log_thread_crash)
    thread = threading.Thread(target=boom, name="phrases")
    thread.start()
    thread.join()
    assert any(r.getMessage() == "phrases died: ValueError" for r in logged)
    assert error_counts()[("assistant", "internal")] == 1
