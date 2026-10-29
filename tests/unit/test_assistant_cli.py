"""The endpoint check and the command line."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from voicebot.assistant import __main__ as cli
from voicebot.assistant import health, models
from voicebot.assistant.config import load_assistant_config

# --- the endpoint check ---------------------------------------------------------------


def _client(models_by_host: dict[str, list[str]], down: str = "") -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.netloc.decode()
        if host == down:
            raise httpx.ConnectError("unreachable", request=request)
        if request.url.path == "/health":
            return httpx.Response(200)
        return httpx.Response(200, json={"data": [{"id": m} for m in models_by_host.get(host, [])]})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_every_service_up_and_serving_its_model() -> None:
    config = load_assistant_config()
    served = {
        health._base(url).split("//")[1]: [model] for _, url, model in health.services(config)
    }
    checks = health.check_all(config, _client(served))
    assert all(c.ok and not c.warning for c in checks)
    assert "unreachable" not in health.report(checks)


def test_a_service_listing_another_model_is_a_warning_not_a_failure() -> None:
    check = health.check_service(
        "English TTS",
        "http://tts.test/v1/audio/speech",
        "qwen3-tts",
        _client({"tts.test": ["Qwen3-TTS-12Hz-0.6B-Base"]}),
    )
    assert check.ok and check.warning and "Qwen3-TTS-12Hz-0.6B-Base" in check.detail


def test_a_service_that_is_down_is_reported_with_the_likely_cause() -> None:
    checks = [
        health.check_service(
            "LLM", "http://llm.test/v1/chat/completions", "m", _client({}, down="llm.test")
        )
    ]
    assert not checks[0].ok
    assert "office network" in health.report(checks)


# --- the command line ------------------------------------------------------------------


def test_check_reports_models_and_endpoints(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli.health,
        "check_all",
        lambda config: [health.Check("LLM", "http://x", True, 5, "serving m")],
    )
    monkeypatch.setattr(
        cli.models, "status", lambda config: [models.ModelFile(Path("m.onnx"), True, True)]
    )
    assert cli.main(["check"]) == 0
    out = capsys.readouterr().out
    assert "Configuration: valid" in out and "OK   m.onnx" in out and "serving m" in out


def test_check_fails_when_a_model_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.health, "check_all", lambda config: [])
    monkeypatch.setattr(
        cli.models, "status", lambda config: [models.ModelFile(Path("m.onnx"), False, False)]
    )
    assert cli.main(["check"]) == 1
    assert "voicebot.assistant setup" in capsys.readouterr().out


def test_run_refuses_to_start_without_its_models(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli.health, "check_all", lambda config: [])
    monkeypatch.setattr(
        cli.models, "status", lambda config: [models.ModelFile(Path("vad.onnx"), False, False)]
    )
    assert cli.main(["run", "--lang", "ur", "--echo", "headphones"]) == 1
    assert "setup" in capsys.readouterr().out


def test_an_unknown_command_is_refused() -> None:
    with pytest.raises(SystemExit):
        cli.main(["fly"])
