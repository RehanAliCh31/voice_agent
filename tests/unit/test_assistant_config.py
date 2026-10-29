"""The assistant's configuration: one file, strictly checked."""

from __future__ import annotations

from pathlib import Path

import pytest

from voicebot.assistant.config import CONFIG_ENV, load_assistant_config

# --- configuration ------------------------------------------------------------------


def test_the_committed_config_loads_and_names_every_service() -> None:
    config = load_assistant_config()
    urls = [
        config.asr.en.url,
        config.asr.ur.url,
        config.tts.en.url,
        config.tts.ur.url,
        config.llm.url,
    ]
    assert all(u.startswith("http://") for u in urls)
    assert config.session.language == "auto" and config.session.echo_mode == "speakers"
    assert config.debug.save_audio is False  # microphone audio is not kept by default


def test_another_file_can_be_named_in_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = (
        Path("configs/assistant.yaml")
        .read_text(encoding="utf-8")
        .replace("follow_up_s: 8", "follow_up_s: 3")
    )
    (tmp_path / "other.yaml").write_text(text, encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV, str(tmp_path / "other.yaml"))
    assert load_assistant_config().session.follow_up_s == 3


def test_a_typo_in_the_config_is_refused(tmp_path: Path) -> None:
    text = (
        Path("configs/assistant.yaml")
        .read_text(encoding="utf-8")
        .replace("follow_up_s:", "folow_up_s:")
    )
    (tmp_path / "typo.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="folow_up_s"):
        load_assistant_config(tmp_path / "typo.yaml")


def test_a_missing_config_says_where_it_looked(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"absent\.yaml"):
        load_assistant_config(tmp_path / "absent.yaml")
