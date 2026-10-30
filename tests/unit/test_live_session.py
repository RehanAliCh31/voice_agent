"""The live session with synthetic audio: turns, pauses, silence, noise, barge-in, stopping."""

from __future__ import annotations

import time

from tests.live_support import (
    MURMUR,
    Page,
    SlowLLM,
    open_session,
    quiet,
    services_for,
    settle,
    speak,
    tone,
)
from voicebot.asr.base import ASRUnavailable
from voicebot.errors import ErrorCode
from voicebot.live.session import INTERRUPTED, State
from voicebot.llm.base import Reply, ToolCall
from voicebot.messages import say

# A sentence: speech, then the silence that ends it (end_silence_ms is 500).
SENTENCE = (tone(900), quiet(700))


def _assistant_lines(services, session_id: str = "live-test") -> list[str]:  # type: ignore[no-untyped-def]
    conversation = services.sessions.get(session_id).conversation
    return [m["content"] for m in conversation.messages if m["role"] == "assistant"]


# --- a conversation ---------------------------------------------------------------


def test_turn_after_turn_with_no_button_pressed() -> None:
    services = services_for(
        [("hello there", "en"), ("what can you do", "en")],
        [Reply(text="Hi. How can I help?"), Reply(text="I can help with your accounts.")],
    )
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: len(p.of("reply")) == 1)
        assert settle(page, "listening")
        speak(session, *SENTENCE)
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        assert settle(page, "listening")
        assert [t["text"] for t in page.of("transcript")] == ["hello there", "what can you do"]
        assert page.states()[:5] == [
            "listening",
            "user_speaking",
            "thinking",
            "agent_speaking",
            "listening",
        ]
        # Every piece of every reply arrived, header then audio.
        assert len(page.of("audio")) == len(page.audio) == 3
        assert session.illegal == 0
    finally:
        session.close()


def test_every_turn_is_measured() -> None:
    services = services_for([("hello", "en")], [Reply(text="Hi.")])
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: bool(p.of("metrics")))
        metrics = page.of("metrics")[0]
        for key in (
            "vad_to_endpoint_ms",
            "stt_ms",
            "agent_ms",
            "tts_first_ms",
            "end_of_speech_to_first_audio_ms",
            "interruptions",
        ):
            assert key in metrics, key
        assert metrics["vad_to_endpoint_ms"] >= 500
    finally:
        session.close()


# --- what is not a turn -------------------------------------------------------------


def test_a_click_or_a_cough_is_not_speech() -> None:
    services = services_for([("should not be heard", "en")], [])
    page = Page()
    session = open_session(services, page)
    try:
        # 64 ms bursts: shorter than min_speech_ms (200).
        for _ in range(10):
            speak(session, tone(64), quiet(400))
        time.sleep(0.3)
        assert page.states() == ["listening"]
        assert services.asr.calls == []  # type: ignore[attr-defined]
    finally:
        session.close()


def test_noise_that_recognises_to_nothing_is_ignored_without_a_word() -> None:
    services = services_for([("", "en")], [])
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, tone(400), quiet(700))
        assert page.wait(lambda p: p.states()[-1:] == ["listening"] and len(p.states()) > 2)
        assert not page.of("reply") and not page.of("say")
        assert session.false_starts == 1
    finally:
        session.close()


def test_silence_asks_then_ends_politely() -> None:
    services = services_for([], [], idle_prompt_s=2.0, idle_end_s=2.0)
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, quiet(2200))
        assert page.wait(lambda p: [e["key"] for e in p.of("say")] == ["still_there"])
        assert settle(page, "listening")
        speak(session, quiet(2200))
        assert page.wait(lambda p: "ended" in p.kinds())
        assert [e["key"] for e in page.of("say")] == ["still_there", "idle_goodbye"]
        assert session.state is State.ENDED
    finally:
        session.close()


# --- pausing mid-thought ------------------------------------------------------------


def test_a_pause_mid_thought_is_one_turn_not_two() -> None:
    # Recognising takes a while, so the caller resumes while the bot is
    # still thinking about the first half: the turn is asked again, whole.
    services = services_for(
        [("transfer fifty dollars", "en"), ("transfer fifty dollars to Ali", "en")],
        [Reply(text="To Ali from checking?")],
        asr_delay_s=0.5,
    )
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, tone(700), quiet(560))  # ends the first half
        assert settle(page, "thinking")
        speak(session, tone(700), quiet(700))  # ... and the rest of it
        assert page.wait(lambda p: len(p.of("reply")) == 1)
        time.sleep(0.6)
        assert len(page.of("reply")) == 1  # one answer, to the whole
        heard = services.asr.calls  # type: ignore[attr-defined]
        assert len(heard) == 2 and heard[1] > heard[0]
        users = [
            m
            for m in services.sessions.get("live-test").conversation.messages
            if m["role"] == "user"
        ]
        assert len(users) == 1 and "Ali" in users[0]["content"]
    finally:
        session.close()


# --- barge-in --------------------------------------------------------------------------


def _speaking(page: Page, session, services) -> None:  # type: ignore[no-untyped-def]
    speak(session, *SENTENCE)
    assert page.wait(lambda p: bool(p.of("reply")))
    assert settle(page, "agent_speaking")


def test_speaking_over_the_bot_stops_it_and_it_answers_the_new_question() -> None:
    services = services_for(
        [("what is my balance", "en"), ("actually my savings", "en")],
        [
            Reply(text="Which account would you like. Checking or savings. Or both."),
            Reply(text="Your savings are fine."),
        ],
    )
    page = Page(auto_finish=False)
    page.stop_at = (1, 150.0)  # part way into the second piece
    session = open_session(services, page)
    try:
        _speaking(page, session, services)
        speak(session, tone(1000), quiet(700))
        assert page.wait(lambda p: bool(p.of("stop_audio")))
        assert page.wait(lambda p: bool(p.of("discard_audio")))
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        lines = _assistant_lines(services)
        # What was heard of the first answer, marked as cut off -- not all of it.
        assert lines[0].endswith(INTERRUPTED) and "Or both" not in lines[0]
        assert lines[0].startswith("Which account would you like.")
        assert session.interruptions == 1
        assert page.wait(lambda p: len(p.of("metrics")) == 2)
        # Stopped once barge_in_min_speech_ms of speech had been heard.
        assert 150 <= page.of("metrics")[1]["barge_in_stop_ms"] <= 400
    finally:
        session.close()


def test_a_reply_let_go_of_is_not_said_to_have_ended() -> None:
    services = services_for(
        [("what is my balance", "en"), ("actually my savings", "en")],
        [
            Reply(text="Which account would you like. Checking or savings. Or both."),
            Reply(text="Your savings are fine."),
        ],
    )
    services.tts.slow = {"Checking or savings.": 1.0}  # still being made when interrupted
    page = Page(auto_finish=False)
    session = open_session(services, page)
    try:
        _speaking(page, session, services)
        first = page.of("reply")[0]["audio_id"]
        speak(session, tone(1000), quiet(700))
        assert page.wait(lambda p: bool(p.of("discard_audio")))
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        time.sleep(1.2)  # the piece being made is finished -- and dropped
        # Nothing more of it went out, and the page was not told it had ended.
        assert [e["piece"] for e in page.of("audio") if e["audio_id"] == first] == [0]
        assert not [e for e in page.of("audio_end") if e["audio_id"] == first]
        assert "Or both." not in [text for text, _ in services.tts.said]
    finally:
        session.close()


def test_a_mm_hmm_does_not_interrupt() -> None:
    services = services_for(
        [("what is my balance", "en"), ("mm hmm", "en")],
        [Reply(text="Your savings balance is fine. Anything else.")],
    )
    page = Page(auto_finish=False)
    session = open_session(services, page)
    try:
        _speaking(page, session, services)
        speak(session, tone(400), quiet(700))
        assert page.wait(lambda p: bool(p.of("resume_audio")))
        assert settle(page, "agent_speaking")
        assert not page.of("discard_audio") and len(page.of("reply")) == 1
        assert session.interruptions == 0
        assert not _assistant_lines(services)[0].endswith(INTERRUPTED)
    finally:
        session.close()


def test_a_murmur_while_the_bot_speaks_is_not_an_interruption() -> None:
    # Unsure sound -- the bot's own voice faintly through the microphone --
    # is above the listening threshold but below barge-in's.
    services = services_for([("what is my balance", "en")], [Reply(text="It is fine. Truly.")])
    page = Page(auto_finish=False)
    session = open_session(services, page)
    try:
        _speaking(page, session, services)
        speak(session, tone(1500, MURMUR))
        time.sleep(0.3)
        assert not page.of("stop_audio")
        assert session.state is State.AGENT_SPEAKING
    finally:
        session.close()


def test_two_interruptions_in_a_row_leave_nothing_stuck() -> None:
    services = services_for(
        [("one", "en"), ("two", "en"), ("three", "en")],
        [
            Reply(text="First answer. More."),
            Reply(text="Second answer. More."),
            Reply(text="Done."),
        ],
    )
    page = Page(auto_finish=False)
    session = open_session(services, page)
    try:
        _speaking(page, session, services)
        speak(session, tone(1000), quiet(700))
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        speak(session, tone(1000), quiet(700))
        assert page.wait(lambda p: len(p.of("reply")) == 3)
        assert session.interruptions == 2
        session.playback_done(page.of("reply")[-1]["audio_id"])
        assert settle(page, "listening")
        assert session.illegal == 0
    finally:
        session.close()


def test_interrupting_the_first_word_keeps_none_of_the_reply() -> None:
    services = services_for(
        [("what is my balance", "en"), ("no wait", "en")],
        [Reply(text="Your savings balance is fine. Anything else."), Reply(text="Yes?")],
    )
    page = Page(auto_finish=False)
    page.stop_at = (0, 0.0)  # stopped before a word of it was heard
    session = open_session(services, page)
    try:
        _speaking(page, session, services)
        speak(session, tone(1000), quiet(700))
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        # Nothing of the first answer was heard, so none of it is on record.
        assert _assistant_lines(services)[0] == INTERRUPTED.strip()
        assert session.interruptions == 1
    finally:
        session.close()


def test_interrupting_at_the_very_end_still_counts() -> None:
    services = services_for(
        [("hello", "en"), ("wait", "en")], [Reply(text="Hello there."), Reply(text="Yes?")]
    )
    page = Page(auto_finish=False)
    page.stop_at = (0, 10_000.0)  # it had all been played
    session = open_session(services, page)
    try:
        _speaking(page, session, services)
        assert page.wait(lambda p: "audio_end" in p.kinds())
        speak(session, tone(1000), quiet(700))
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        assert _assistant_lines(services)[0] == "Hello there." + INTERRUPTED
    finally:
        session.close()


# --- stopping ------------------------------------------------------------------------


def test_start_twice_and_stop_twice_are_one_session() -> None:
    services = services_for([], [])
    page = Page()
    session = open_session(services, page)
    session.start()
    session.stop()
    session.stop()
    session.close()
    assert page.states() == ["listening", "ended"]
    assert page.kinds().count("ended") == 1
    assert session.threads_alive == 0


def test_stopping_in_every_state_ends_cleanly() -> None:
    for where in ("listening", "user_speaking", "thinking", "agent_speaking"):
        services = services_for(
            [("hello", "en")],
            [],
            llm=SlowLLM([Reply(text="Hi. There.")], delay_s=0.4),
        )
        page = Page(auto_finish=False)
        session = open_session(services, page)
        if where == "user_speaking":
            speak(session, tone(600))
        elif where in ("thinking", "agent_speaking"):
            speak(session, *SENTENCE)
        assert settle(page, where), where
        session.stop()
        session.close(timeout_s=5)
        assert session.state is State.ENDED, where
        events_at_end = len(page.events)
        time.sleep(0.6)  # whatever was still running finishes -- and says nothing
        assert len(page.events) == events_at_end, where
        assert session.threads_alive == 0, where


def test_the_time_limit_says_goodbye_once_and_ends() -> None:
    services = services_for([], [], max_session_minutes=0.01)  # 0.6 s
    goodbye = say("session_ended", "en")
    # The voice takes its time, as it does under load: the sentence is asked
    # for once, not again on every tick while it is being made.
    services.tts.slow = {goodbye: 1.0}
    page = Page()
    session = open_session(services, page)
    try:
        assert page.wait(lambda p: "ended" in p.kinds(), timeout=8)
        assert [e["key"] for e in page.of("say")] == ["session_ended"]
        assert [text for text, _ in services.tts.said].count(goodbye) == 1
    finally:
        session.close()
    assert session.threads_alive == 0


# --- when a provider fails --------------------------------------------------------------


def test_the_recognisers_down_is_said_not_silent() -> None:
    services = services_for([ASRUnavailable("down")], [])
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: bool(p.of("say")))
        assert page.of("say")[0]["key"] == "speech_unavailable"
        assert settle(page, "listening")
    finally:
        session.close()


# --- two languages -----------------------------------------------------------------------


def test_english_then_urdu_each_in_its_own_voice() -> None:
    services = services_for(
        [("what is my balance", "en"), ("میرا بیلنس کیا ہے", "ur")],
        [Reply(text="It is fine."), Reply(text="ٹھیک ہے۔")],
    )
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: len(p.of("reply")) == 1)
        assert settle(page, "listening")
        speak(session, *SENTENCE)
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        voices = [voice for _, voice in services.tts.said]  # type: ignore[attr-defined]
        assert voices == ["en", "ur"]
        assert [t["language"] for t in page.of("transcript")] == ["en", "ur"]
    finally:
        session.close()


def test_the_fallback_is_in_the_conversations_language() -> None:
    services = services_for(
        [("میرا بیلنس", "ur"), ASRUnavailable("down")], [Reply(text="ٹھیک ہے۔")]
    )
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: len(p.of("reply")) == 1)
        assert settle(page, "listening")
        speak(session, *SENTENCE)
        assert page.wait(lambda p: bool(p.of("say")))
        assert page.of("say")[0]["reply"] == say("speech_unavailable", "ur")
    finally:
        session.close()


# --- a long session -----------------------------------------------------------------


def test_fifteen_minutes_and_ninety_turns_leak_nothing() -> None:
    import threading
    import tracemalloc

    turns = 90
    services = services_for(
        [(f"question number {n}", "en") for n in range(turns)],
        [Reply(text=f"Answer number {n}. Anything else.") for n in range(turns)],
        session={"max_turns": 1000, "max_minutes": 120},
    )
    page = Page(keep_audio=False)  # the stand-in page keeps no audio of its own
    session = open_session(services, page)
    tracemalloc.start()
    baseline = None
    try:
        for n in range(turns):
            # 1.5 s of speech, the silence that ends it, then quiet: 10 s a turn.
            speak(session, tone(1500), quiet(700))
            assert page.wait(lambda p, n=n: len(p.of("reply")) == n + 1, timeout=10), n
            assert settle(page, "listening"), n
            speak(session, quiet(7800))
            if n == 10:
                baseline = tracemalloc.take_snapshot()
        assert baseline is not None
        after = tracemalloc.take_snapshot()
        diffs = [
            stat
            for stat in after.compare_to(baseline, "traceback")
            if any("voicebot" in frame.filename for frame in stat.traceback)
        ]
        grown = sum(stat.size_diff for stat in diffs)
        # 80 turns on from the baseline: what remains is the history the
        # conversation trims to, the turns' metrics and the bank's records.
        assert grown < 1_000_000, f"{grown} bytes more after 80 turns"
        assert page.audio_count >= 2 * turns  # every piece of every reply was sent
        assert len(session._threads) < 10  # finished threads are let go
        assert session.illegal == 0
    finally:
        tracemalloc.stop()
        session.close()
    assert session.threads_alive == 0
    assert not [t.name for t in threading.enumerate() if t.name.startswith("live-")]


def test_the_model_down_is_said_in_the_callers_language_not_silent() -> None:
    from voicebot.llm.base import LLMClient, LLMUnavailable

    class Down(LLMClient):
        def complete(self, messages, tools, tool_choice="auto"):  # type: ignore[no-untyped-def]
            raise LLMUnavailable("timed out")

    services = services_for([("میرا بیلنس کیا ہے", "ur")], [], llm=Down())
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: bool(p.of("reply")))
        assert page.of("reply")[0]["reply"] == say("unavailable", "ur")
        assert page.wait(lambda p: p.audio_count >= 1)  # said, not just shown
        assert settle(page, "listening")
    finally:
        session.close()


def test_a_recogniser_that_times_out_is_said_not_silent() -> None:
    timed_out = ASRUnavailable(
        "the recogniser did not answer within 4.0s", code=ErrorCode.PROVIDER_TIMEOUT
    )
    services = services_for([timed_out, ("hello", "en")], [Reply(text="Hi.")])
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: bool(p.of("say")))
        assert page.of("say")[0]["key"] == "speech_unavailable"
        assert settle(page, "listening")
        # And the next turn is heard as usual.
        speak(session, *SENTENCE)
        assert page.wait(lambda p: bool(p.of("reply")))
    finally:
        session.close()


def test_a_model_slower_than_the_turns_budget_is_said_not_silent() -> None:
    # The first round asks for a tool and takes most of the budget: there is
    # not time for a second, so the turn says so rather than wait on.
    slow = SlowLLM(
        [
            Reply(tool_calls=(ToolCall("c1", "get_balance", {"account": "savings"}),)),
            Reply(text="Your savings are fine."),
        ],
        delay_s=0.8,
    )
    services = services_for(
        [("what is my balance", "en")], [], llm=slow, resilience={"turn_budget_s": 1.5}
    )
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: bool(p.of("reply")))
        assert page.of("reply")[0]["reply"] == say("unavailable", "en")
        assert page.wait(lambda p: p.audio_count >= 1)  # said, not just shown
        assert settle(page, "listening")
    finally:
        session.close()


# --- races --------------------------------------------------------------------------


def test_resuming_after_a_pause_mid_thought_waits_for_the_first_half_not_says_busy() -> None:
    # The model is slow, and the caller resumes while it is answering the
    # first half: the whole is answered once the first half lets go -- never
    # "one moment, I'm still answering".
    services = services_for(
        [("transfer fifty dollars", "en"), ("transfer fifty dollars to Ali", "en")],
        [],
        # Longer than take_turn waits for a busy conversation (2 s).
        llm=SlowLLM([Reply(text="First half."), Reply(text="To Ali from checking?")], delay_s=2.5),
    )
    page = Page()
    session = open_session(services, page)
    try:
        speak(session, tone(700), quiet(560))
        assert settle(page, "thinking")
        time.sleep(0.3)  # the model is answering the first half now
        speak(session, tone(700), quiet(700))
        assert page.wait(lambda p: len(p.of("reply")) == 1, timeout=15)
        reply = page.of("reply")[0]
        assert reply["reply"] == "To Ali from checking?"
        assert reply["fallback"] != "busy"
        users = [
            m
            for m in services.sessions.get("live-test").conversation.messages
            if m["role"] == "user"
        ]
        assert len(users) == 1  # the first half's turn was put back
    finally:
        session.close()


def test_a_resumed_reply_has_time_again_to_finish() -> None:
    services = services_for(
        [("hello", "en"), ("mm hmm", "en")], [Reply(text="A reply of one piece only")]
    )
    page = Page(auto_finish=False)
    session = open_session(services, page)
    try:
        speak(session, *SENTENCE)
        assert page.wait(lambda p: "audio_end" in p.kinds())
        speak(session, tone(400), quiet(700))
        assert page.wait(lambda p: bool(p.of("resume_audio")))
        playing = session._playing
        assert playing is not None and playing.deadline > time.monotonic() + 0.5
    finally:
        session.close()
