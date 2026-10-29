"""The command line.

    python -m voicebot.assistant setup              download the wake-word and VAD models
    python -m voicebot.assistant check              the configuration, the models, every endpoint
    python -m voicebot.assistant devices            microphones and speakers, with their indexes
    python -m voicebot.assistant say "text" --lang ur   speak a sentence, to hear a voice
    python -m voicebot.assistant run                start listening for the wake word

`run` takes --lang, --echo, --input and --output to override the config for
one session, and --debug to print transcripts (they are logged nowhere
otherwise).
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
from typing import Any

from voicebot.assistant import health, models
from voicebot.assistant.config import AssistantConfig, Language, load_assistant_config
from voicebot.utils.env import load_env_file

STATUS_LABEL = {
    "idle": "Idle",
    "listening": "Listening",
    "transcribing": "Transcribing",
    "thinking": "Thinking",
    "speaking": "Speaking",
    "follow-up": "Listening",
}


def _config(args: argparse.Namespace) -> AssistantConfig:
    load_env_file()
    config = load_assistant_config(args.config)
    session = config.session
    if getattr(args, "lang", None):
        session.language = args.lang
    if getattr(args, "echo", None):
        session.echo_mode = args.echo
    if getattr(args, "input", None) is not None:
        config.audio.input_device = args.input
    if getattr(args, "output", None) is not None:
        config.audio.output_device = args.output
    return config


def cmd_setup(args: argparse.Namespace) -> int:
    config = _config(args)
    files = models.download(config)
    for f in files:
        print(f"  {'OK  ' if f.verified else 'FAIL'} {f.path}")
    return 0 if all(f.verified for f in files) else 1


def cmd_check(args: argparse.Namespace) -> int:
    config = _config(args)
    print(
        f"Configuration: valid (language {config.session.language}, "
        f"echo mode {config.session.echo_mode})"
    )
    print("\nModels:")
    files = models.status(config)
    for f in files:
        state = "OK  " if f.verified else ("BAD " if f.present else "MISSING")
        print(f"  {state} {f.path}")
    if not all(f.verified for f in files):
        print("  Run: python -m voicebot.assistant setup")
    print("\nEndpoints:")
    checks = health.check_all(config)
    print(health.report(checks))
    return 0 if all(f.verified for f in files) and all(c.ok for c in checks) else 1


def cmd_devices(args: argparse.Namespace) -> int:
    from voicebot.assistant.audio_io import list_devices, sounddevice

    sd = sounddevice()
    default_in, default_out = sd.default.device
    for d in list_devices(sd):
        kinds = ("in " if d.inputs else "   ") + ("out" if d.outputs else "   ")
        mark = " (default)" if d.index in (default_in, default_out) else ""
        print(f"  [{d.index:>2}] {kinds}  {d.host_api:<22} {d.name}{mark}")
    return 0


def _voices(config: AssistantConfig) -> dict[Language, Any]:
    from voicebot.assistant.tts_client import TtsClient

    tts = config.tts
    return {
        "en": TtsClient(tts.en, "en", tts.timeout_s, tts.retries, tts.trim_silence),
        "ur": TtsClient(tts.ur, "ur", tts.timeout_s, tts.retries, tts.trim_silence),
    }


def cmd_say(args: argparse.Namespace) -> int:
    from voicebot.assistant.player import Player
    from voicebot.assistant.text_prep import prepare_for_speech

    config = _config(args)
    speech = _voices(config)[args.lang].synthesize(prepare_for_speech(args.text, args.lang))
    print(f"  synthesised {speech.duration_s:.2f} s in {speech.latency_ms} ms; playing")
    player = Player(config.audio.output_device)
    player.start()
    player.enqueue(speech.samples, speech.sample_rate)
    player.wait(speech.duration_s + 5)
    player.close()
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from voicebot.assistant.asr_client import AsrClient
    from voicebot.assistant.audio_io import AudioDeviceError, Microphone
    from voicebot.assistant.language import LanguageRouter
    from voicebot.assistant.llm_client import ChatClient
    from voicebot.assistant.orchestrator import Orchestrator, State
    from voicebot.assistant.player import Player
    from voicebot.assistant.vad import SileroVad
    from voicebot.assistant.wake_word import WakeWord

    config = _config(args)
    if args.debug:
        logging.getLogger("voicebot").setLevel(logging.DEBUG)

    print("Checking the endpoints:")
    print(health.report(health.check_all(config)))
    missing = [f.path for f in models.status(config) if not f.verified]
    if missing:
        print(f"\nModel files missing or damaged: {', '.join(map(str, missing))}")
        print("Run: python -m voicebot.assistant setup")
        return 1

    asr = config.asr
    router = LanguageRouter(
        config.session.language,
        AsrClient(asr.en, "en", asr.timeout_s, asr.retries),
        AsrClient(asr.ur, "ur", asr.timeout_s, asr.retries),
    )

    def show(state: State, message: str) -> None:
        print(f"[{STATUS_LABEL[state.value]:<12}] {message}", flush=True)

    orchestrator: Orchestrator
    player = Player(
        config.audio.output_device, on_start=lambda tag: orchestrator.audio_started(tag)
    )
    orchestrator = Orchestrator(
        config=config,
        wake=WakeWord(config.wake_word),
        vad=SileroVad(config.vad.model_path),
        router=router,
        llm=ChatClient(config.llm),
        voices=_voices(config),
        player=player,
        status=show,
    )
    threading.Thread(target=orchestrator.prepare_phrases, name="phrases", daemon=True).start()
    microphone = Microphone(config.audio.input_device, config.audio.queue_frames)
    stop = threading.Event()
    print(
        f'\nWake word "{config.wake_word.phrase}", language {config.session.language}, '
        f"echo mode {config.session.echo_mode}. Ctrl+C to quit.\n"
    )
    try:
        player.start()
        orchestrator.run(microphone, stop)
    except AudioDeviceError as exc:
        print(
            f"\nAudio device problem: {exc}\n"
            "List devices with: python -m voicebot.assistant devices"
        )
        return 1
    except KeyboardInterrupt:
        stop.set()
        print("\nStopped.")
    finally:
        player.close()
        router.close()
        if microphone.dropped:
            print(f"(the microphone ran ahead and {microphone.dropped} frames were dropped)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voicebot.assistant", description="The hands-free assistant."
    )
    parser.add_argument(
        "--config",
        help="the config file (default: $VOICEBOT_ASSISTANT_CONFIG, else configs/assistant.yaml)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="download the model files")
    commands.add_parser("check", help="check the configuration, the models and every endpoint")
    commands.add_parser("devices", help="list microphones and speakers")
    say = commands.add_parser("say", help="speak a sentence with one of the voices")
    say.add_argument("text")
    say.add_argument("--lang", choices=["en", "ur"], default="en")
    run = commands.add_parser("run", help="start listening for the wake word")
    run.add_argument("--lang", choices=["auto", "en", "ur"])
    run.add_argument("--echo", choices=["speakers", "headphones"])
    run.add_argument("--input", help="microphone: an index or part of a name")
    run.add_argument("--output", help="speaker: an index or part of a name")
    run.add_argument("--debug", action="store_true", help="log transcripts and every decision")
    args = parser.parse_args(argv)
    handler = {
        "setup": cmd_setup,
        "check": cmd_check,
        "devices": cmd_devices,
        "say": cmd_say,
        "run": cmd_run,
    }
    return handler[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
