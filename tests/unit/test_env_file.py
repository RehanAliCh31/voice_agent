"""Reading a .env file.

Small enough to write rather than depend on. The behaviour that matters is
the last test: something already in the environment wins, so a shell export
is never quietly overridden by a file somebody forgot they had.
"""

from __future__ import annotations

import os

from voicebot.utils.env import load_env_file


def write(tmp_path, text: str):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_value_reaches_the_environment(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VB_TEST_KEY", raising=False)

    load_env_file(write(tmp_path, "VB_TEST_KEY=secret\n"))

    assert os.environ["VB_TEST_KEY"] == "secret"


def test_comments_and_blank_lines_are_skipped(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VB_TEST_KEY", raising=False)

    load_env_file(write(tmp_path, "# a note\n\n   \nVB_TEST_KEY=kept\n# another\n"))

    assert os.environ["VB_TEST_KEY"] == "kept"


def test_quotes_are_not_part_of_the_value(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VB_TEST_KEY", raising=False)

    load_env_file(write(tmp_path, 'VB_TEST_KEY="in quotes"\n'))

    assert os.environ["VB_TEST_KEY"] == "in quotes"


def test_only_the_first_equals_splits_the_line(tmp_path, monkeypatch) -> None:
    """A URL has one in it, and the value keeps it."""
    monkeypatch.delenv("VB_TEST_KEY", raising=False)

    load_env_file(write(tmp_path, "VB_TEST_KEY=http://host/v1?a=b\n"))

    assert os.environ["VB_TEST_KEY"] == "http://host/v1?a=b"


def test_the_shell_wins(tmp_path, monkeypatch) -> None:
    """Already set is already decided. A forgotten file does not override it."""
    monkeypatch.setenv("VB_TEST_KEY", "from the shell")

    load_env_file(write(tmp_path, "VB_TEST_KEY=from the file\n"))

    assert os.environ["VB_TEST_KEY"] == "from the shell"


def test_no_file_is_not_a_problem(tmp_path) -> None:
    """The usual case: there is no .env, and that is fine."""
    load_env_file(tmp_path / "nothing-here")


def test_a_line_with_no_equals_is_ignored(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("VB_TEST_KEY", raising=False)

    load_env_file(write(tmp_path, "this is not a setting\nVB_TEST_KEY=fine\n"))

    assert os.environ["VB_TEST_KEY"] == "fine"


def test_an_empty_value_is_treated_as_unset(tmp_path, monkeypatch) -> None:
    """`VOICEBOT_LLM_KEY=` in the example means no key, not an empty key."""
    monkeypatch.delenv("VB_TEST_KEY", raising=False)

    load_env_file(write(tmp_path, "VB_TEST_KEY=\n"))

    assert "VB_TEST_KEY" not in os.environ
