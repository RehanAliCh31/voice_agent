import pytest
import yaml

from voicebot.config import AppConfig, load_config


def test_defaults_select_fake_engines():
    cfg = AppConfig()
    assert cfg.asr.engine == "fake"
    assert cfg.nlu.engine == "fake"
    assert cfg.tts.engine == "fake"
    assert cfg.seed == 42


def test_load_config_overrides_only_given_keys(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump({"seed": 7, "asr": {"engine": "whisper"}}))
    cfg = load_config(path)
    assert cfg.seed == 7
    assert cfg.asr.engine == "whisper"
    assert cfg.nlu.engine == "fake"  # untouched default survives


def test_load_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump({"sedd": 7}))
    with pytest.raises(ValueError, match="sedd"):
        load_config(path)


def test_load_config_reports_missing_file_by_path(tmp_path):
    missing = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError, match=r"nope\.yaml"):
        load_config(missing)


def test_config_round_trips_to_dict_for_result_files():
    cfg = AppConfig()
    assert cfg.model_dump()["training"]["learning_rate"] == pytest.approx(3e-5)
