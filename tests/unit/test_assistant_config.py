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


# --- who the bot is ---------------------------------------------------------------


def _with(tmp_path: Path, old: str, new: str) -> Path:
    text = Path("configs/assistant.yaml").read_text(encoding="utf-8")
    assert old in text, old
    path = tmp_path / "agent.yaml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_the_shipped_agent_is_one_gender_in_every_voice() -> None:
    config = load_assistant_config()
    assert config.agent is not None
    gender = config.agent.gender
    for language in ("en", "ur"):
        assert config.tts.voice(language).gender == gender  # type: ignore[arg-type]
        backup = config.tts.backup(language)  # type: ignore[arg-type]
        assert backup is None or backup.gender == gender
    assert config.profile().gender == gender


def test_the_gender_switch_brings_the_voices_and_the_name_with_it(tmp_path: Path) -> None:
    config = load_assistant_config(_with(tmp_path, "gender: female\n", "gender: male\n"))
    assert config.profile().gender == "male" and config.profile().name("en") == "Omar"
    assert config.tts.en.gender == config.tts.ur.gender == "male"
    assert config.tts.backup_en is None  # the male voices have no backups configured


def test_a_voice_of_the_other_gender_stops_startup(tmp_path: Path) -> None:
    path = _with(
        tmp_path,
        "voice: urdu_male5\n        gender: male",
        "voice: urdu_male5\n        gender: female",
    )
    with pytest.raises(ValueError, match=r"agent\.voices\.male\.en is female, not male"):
        load_assistant_config(path)


def test_a_backup_of_the_other_gender_stops_startup(tmp_path: Path) -> None:
    path = _with(
        tmp_path,
        "voice: assistant-ur\n        gender: female",
        "voice: assistant-ur\n        gender: male",
    )
    with pytest.raises(ValueError, match=r"backup_voices\.female\.ur is male"):
        load_assistant_config(path)


def test_a_voice_that_says_nothing_of_its_gender_stops_startup(tmp_path: Path) -> None:
    text = Path("configs/assistant.yaml").read_text(encoding="utf-8")
    start = text.index("voice: assistant-pk")
    end = text.index("gender: female\n", start) + len("gender: female\n")
    path = _with(tmp_path, text[start:end], "voice: assistant-pk\n")
    with pytest.raises(ValueError, match="of no stated gender"):
        load_assistant_config(path)


def test_no_voices_for_the_agents_gender_stops_startup(tmp_path: Path) -> None:
    text = Path("configs/assistant.yaml").read_text(encoding="utf-8")
    start, end = text.index("    male:\n      en:"), text.index(
        "  # Tried when the main voice fails"
    )
    path = tmp_path / "agent.yaml"
    path.write_text(text[:start] + text[end:], encoding="utf-8")
    path.write_text(
        path.read_text(encoding="utf-8").replace("gender: female\n", "gender: male\n", 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no male voices for a male agent"):
        load_assistant_config(path)


def test_voices_are_set_in_one_place_only(tmp_path: Path) -> None:
    path = _with(tmp_path, "tts:\n", "tts:\n  en: {url: http://x.test, model: m}\n")
    with pytest.raises(ValueError, match=r"tts\.en cannot be set when there is an agent"):
        load_assistant_config(path)


def test_a_reference_voice_needs_its_file_and_its_words(tmp_path: Path) -> None:
    missing = _with(
        tmp_path,
        "voice: urdu_male5\n        gender: male",
        "reference: voice_samples/references/no-such.wav\n"
        "        reference_text: x\n        gender: male",
    )
    with pytest.raises(ValueError, match=r"no file at voice_samples/references/no-such\.wav"):
        load_assistant_config(missing)
    (tmp_path / "ref.wav").write_bytes(b"RIFF")
    wordless = _with(
        tmp_path,
        "voice: urdu_male5\n        gender: male",
        f"reference: {(tmp_path / 'ref.wav').as_posix()}\n        gender: male",
    )
    with pytest.raises(ValueError, match="a reference needs its reference_text"):
        load_assistant_config(wordless)
