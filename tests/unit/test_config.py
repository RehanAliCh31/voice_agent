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


# --- v2: the LLM and the database --------------------------------------------


def test_the_llm_and_database_sections_load(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "llm": {"engine": "openai", "model": "qwen3.8-27b-fp8", "max_tool_rounds": 6},
                "database": {"path": "data/bank.db"},
            }
        )
    )
    cfg = load_config(path)
    assert cfg.llm.model == "qwen3.8-27b-fp8"
    assert cfg.llm.max_tool_rounds == 6
    assert cfg.database.path == "data/bank.db"


def test_the_llm_defaults_to_the_fake(tmp_path):
    # A bare config must start the service with no network and no downloads.
    cfg = AppConfig()
    assert cfg.llm.engine == "fake"
    assert cfg.database.path == ":memory:"


def test_an_unknown_llm_key_is_rejected(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump({"llm": {"modle": "typo"}}))
    with pytest.raises(ValueError, match="modle"):
        load_config(path)


def test_the_api_key_is_named_not_stored(tmp_path):
    # The key is read from the environment, so a config file is safe to commit.
    cfg = AppConfig()
    assert cfg.llm.api_key_env == "VOICEBOT_LLM_KEY"
    assert not hasattr(cfg.llm, "api_key")


def test_browser_is_a_valid_tts_engine(tmp_path):
    path = tmp_path / "app.yaml"
    path.write_text(yaml.safe_dump({"tts": {"engine": "browser"}}))
    assert load_config(path).tts.engine == "browser"
