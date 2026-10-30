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
from collections.abc import Callable
from typing import Any

from voicebot import profile
from voicebot.assistant import health, models
from voicebot.assistant.config import AssistantConfig, Language, load_assistant_config
from voicebot.errors import InternalError
from voicebot.utils.env import load_env_file
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report

log = get_logger("assistant.cli")

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
    # Who the bot is, for every sentence it says from here on.
    profile.use(config.profile())
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


def _log_thread_crash(args: threading.ExceptHookArgs) -> None:
    """A thread that died of an exception: logged like everything else, and counted.

    Python's own hook prints the traceback to stderr, outside the log, with
    no call id; and a thread that dies there leaves no count behind.
    """
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "a thread"
    log.error("%s died: %s", name, args.exc_type.__name__, exc_info=args.exc_value)
    report(
        InternalError(
            f"{name} died: {args.exc_type.__name__}", component="assistant", cause=args.exc_value
        )
    )


def _close_quietly(what: str, close: Callable[[], None]) -> None:
    """Close one thing on the way out, so one that fails cannot keep the rest open."""
    try:
        close()
    except Exception as exc:  # the process is ending; say so, and close the next thing
        log.warning("could not close %s: %s", what, exc)


def cmd_setup(args: argparse.Namespace) -> int:
    config = _config(args)
    try:
        files = models.download(config)
    except (models.ModelDownloadError, OSError) as exc:
        print(f"  Could not set up the models: {exc}")
        return 1
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
    agent = config.profile()
    print(f"\nAgent: {agent.name('en')}, {agent.gender}")
    missing = health.missing_voices(config)
    for line in missing:
        print(f"  MISSING voice {line}")
    ok = all(f.verified for f in files) and all(c.ok for c in checks) and not missing
    return 0 if ok else 1


def cmd_devices(args: argparse.Namespace) -> int:
    from voicebot.assistant.audio_io import AudioDeviceError, list_devices, sounddevice

    try:
        sd = sounddevice()
        default_in, default_out = sd.default.device
        devices = list_devices(sd)
    except AudioDeviceError as exc:
        print(f"  Could not list the sound devices: {exc}")
        return 1
    for d in devices:
        kinds = ("in " if d.inputs else "   ") + ("out" if d.outputs else "   ")
        mark = " (default)" if d.index in (default_in, default_out) else ""
        print(f"  [{d.index:>2}] {kinds}  {d.host_api:<22} {d.name}{mark}")
    return 0


def _voices(config: AssistantConfig) -> dict[Language, Any]:
    from voicebot.assistant.tts_client import build_voices

    return dict(build_voices(config))


def cmd_say(args: argparse.Namespace) -> int:
    from voicebot.assistant.audio_io import AudioDeviceError
    from voicebot.assistant.player import Player
    from voicebot.assistant.text_prep import prepare_for_speech
    from voicebot.assistant.tts_client import TtsError

    config = _config(args)
    voices = _voices(config)
    player = Player(config.audio.output_device)
    try:
        speech = voices[args.lang].synthesize(prepare_for_speech(args.text, args.lang))
        print(f"  synthesised {speech.duration_s:.2f} s in {speech.latency_ms} ms; playing")
        player.start()
        player.enqueue(speech.samples, speech.sample_rate)
        if not player.wait(speech.duration_s + 5):
            print("  the speaker did not finish playing it")
            return 1
    except TtsError as exc:
        print(f"  Could not synthesise it: {exc}")
        return 1
    except AudioDeviceError as exc:
        print(f"  Could not play it: {exc}")
        return 1
    finally:
        _close_quietly("the speaker", player.close)
        for name, voice in voices.items():
            _close_quietly(f"the {name} voice", voice.close)
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

    def show(state: State, message: str) -> None:
        print(f"[{STATUS_LABEL[state.value]:<12}] {message}", flush=True)

    asr = config.asr
    stop = threading.Event()
    microphone: Microphone | None = None
    # Everything opened, in the order it is closed: the speaker first, so
    # nothing more is heard, then every client's connections.
    opened: list[tuple[str, Callable[[], None]]] = []
    try:
        orchestrator: Orchestrator
        player = Player(
            config.audio.output_device, on_start=lambda tag: orchestrator.audio_started(tag)
        )
        opened.append(("the speaker", player.close))
        router = LanguageRouter(
            config.session.language,
            AsrClient(asr.en, "en", asr.timeout_s, asr.retries),
            AsrClient(asr.ur, "ur", asr.timeout_s, asr.retries),
        )
        opened.append(("the recognisers", router.close))
        llm = ChatClient(config.llm)
        opened.append(("the language model client", llm.close))
        voices = _voices(config)
        opened += [(f"the {name} voice", voice.close) for name, voice in voices.items()]
        orchestrator = Orchestrator(
            config=config,
            wake=WakeWord(config.wake_word),
            vad=SileroVad(config.vad.model_path),
            router=router,
            llm=llm,
            voices=voices,
            player=player,
            status=show,
        )
        threading.Thread(target=orchestrator.prepare_phrases, name="phrases", daemon=True).start()
        microphone = Microphone(config.audio.input_device, config.audio.queue_frames)
        print(
            f'\nWake word "{config.wake_word.phrase}", language {config.session.language}, '
            f"echo mode {config.session.echo_mode}. Ctrl+C to quit.\n"
        )
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
        for what, close in opened:
            _close_quietly(what, close)
        if microphone is not None and microphone.dropped:
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
    previous_hook = threading.excepthook
    threading.excepthook = _log_thread_crash
    try:
        return handler[args.command](args)
    except KeyboardInterrupt:
        # Ctrl+C before anything was running -- during the endpoint check or
        # while the models load: nothing to tidy beyond what has closed itself.
        print("\nStopped.")
        return 130
    finally:
        threading.excepthook = previous_hook


if __name__ == "__main__":
    sys.exit(main())
