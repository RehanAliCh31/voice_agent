"""Command line entry point for looking after the demo bank.

    python -m voicebot.bank --show
    python -m voicebot.bank --deposit 500 --account checking
    python -m voicebot.bank --reset

A module of its own rather than running `seed` directly: the package's
__init__ already imports it, so executing it as __main__ would load it twice
and Python says so in a warning the person running this does not need to see.
"""

from voicebot.bank.seed import main

if __name__ == "__main__":
    main()
