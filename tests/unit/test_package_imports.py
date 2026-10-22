"""Guards that the package imports cleanly with no optional dependency installed."""

import importlib


def test_core_modules_import_without_optional_extras():
    for module in [
        "voicebot",
        "voicebot.schemas",
        "voicebot.config",
        "voicebot.utils.seed",
        "voicebot.utils.logging",
        "voicebot.utils.registry",
        "voicebot.asr",
        "voicebot.nlu",
        "voicebot.tts",
    ]:
        importlib.import_module(module)


def test_package_exposes_version():
    import voicebot

    assert isinstance(voicebot.__version__, str)
