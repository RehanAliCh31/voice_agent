"""The command line.

python -m voicebot.assistant setup    download the voice activity model
python -m voicebot.assistant check    the configuration, the model, every endpoint
"""

from __future__ import annotations

import argparse
import sys
import threading

from voicebot import profile
from voicebot.assistant import health, models
from voicebot.assistant.config import AssistantConfig, load_assistant_config
from voicebot.errors import InternalError
from voicebot.utils.env import load_env_file
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report

log = get_logger("assistant.cli")


def _config(args: argparse.Namespace) -> AssistantConfig:
    load_env_file()
    config = load_assistant_config(args.config)
    # Who the bot is, for every sentence it says from here on.
    profile.use(config.profile())
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


def cmd_setup(args: argparse.Namespace) -> int:
    _config(args)
    try:
        files = models.download()
    except (models.ModelDownloadError, OSError) as exc:
        print(f"  Could not set up the models: {exc}")
        return 1
    for f in files:
        print(f"  {'OK  ' if f.verified else 'FAIL'} {f.path}")
    return 0 if all(f.verified for f in files) else 1


def cmd_check(args: argparse.Namespace) -> int:
    config = _config(args)
    print("Configuration: valid")
    print("\nModels:")
    files = models.status()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m voicebot.assistant",
        description="The office services and the model the voice bot uses.",
    )
    parser.add_argument(
        "--config",
        help="the config file (default: $VOICEBOT_ASSISTANT_CONFIG, else configs/assistant.yaml)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("setup", help="download the voice activity model")
    commands.add_parser("check", help="check the configuration, the model and every endpoint")
    args = parser.parse_args(argv)
    handler = {"setup": cmd_setup, "check": cmd_check}
    previous_hook = threading.excepthook
    threading.excepthook = _log_thread_crash
    try:
        return handler[args.command](args)
    except KeyboardInterrupt:
        # Ctrl+C during the endpoint check or a download: nothing to tidy
        # beyond what has closed itself.
        print("\nStopped.")
        return 130
    finally:
        threading.excepthook = previous_hook


if __name__ == "__main__":
    sys.exit(main())
