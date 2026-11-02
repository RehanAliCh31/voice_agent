"""The whole turn: audio in, audio out.

This is the only place the four stages meet. Each one is held behind its
interface, so the pipeline is identical whether it is driving fakes in a test
or Whisper and a 27B model over the network.

v1 ran a normalisation stage between the ASR and the NLU, rewriting Whisper's
output into the lowercased, punctuation-free form the trained model expected.
There is no such stage now: casing and punctuation are signal to a language
model, and stripping them would be deliberate damage.
"""

from __future__ import annotations

import contextlib
import dataclasses
import re
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np

from voicebot.agent.loop import Agent, AgentResult, Conversation
from voicebot.asr.base import ASREngine, ASRUnavailable
from voicebot.assistant.language import typed_language
from voicebot.errors import Cancelled, InternalError
from voicebot.llm.base import listening
from voicebot.messages import say
from voicebot.pipeline.audio_io import is_silent, trim_silence
from voicebot.pipeline.latency import LatencyBreakdown, LatencyTracker
from voicebot.pipeline.spoken import SpokenReply, in_background
from voicebot.schemas import AudioBuffer, Transcript
from voicebot.tts.base import TTSEngine
from voicebot.utils.logging import get_logger
from voicebot.utils.observability import report
from voicebot.utils.resilience import checkpoint

log = get_logger("pipeline.orchestrator")

SILENCE = np.zeros(0, dtype=np.float32)

# Whisper returns punctuation-only strings on room tone -- "...", ".", "?!".
# Anything with no letter or digit in it carries no request. A letter of any
# script counts: an Urdu question has no A to Z in it at all.
_HAS_CONTENT = re.compile(r"[^\W_]")


@dataclass(frozen=True)
class TurnResult:
    """Everything one turn produced, including the parts the UI inspects."""

    transcript: Transcript
    text: str
    agent: AgentResult
    # The first piece of the reply's audio; `rest`, when there is more, is
    # the remainder being made in the background (voicebot.pipeline.spoken).
    audio: AudioBuffer
    latency: LatencyBreakdown
    rest: SpokenReply | None = None
    # The text of each piece of the audio, in order: the first is `audio`,
    # the others are `rest`'s. What was said when someone interrupts.
    pieces: tuple[str, ...] = ()


def _hear_language(transcript: Transcript, conversation: Conversation) -> None:
    """The conversation's language after hearing `transcript`.

    The recognisers' "Urdu" stands. Anything else is read as typing is, so a
    word or two -- "yes", "savings" -- leaves the conversation in the
    language it was already in.
    """
    conversation.language = (
        "ur"
        if transcript.language == "ur"
        else typed_language(transcript.text, conversation.language)  # type: ignore[arg-type]
    )


class _EarlyFirstPiece:
    """A reply's first piece, made by the voice while the model is still writing.

    Fed the model's words as they come (a TextListener). The first piece the
    voice would cut from them, once they settle it, is synthesised in the
    background. `take` gives that audio only for the very same piece of the
    reply that passed the guards; anything else -- the guard changed the
    reply, the language was asked for again, a round started over -- and it
    is not used, and the piece is made as it always was.
    """

    def __init__(
        self, tts: TTSEngine, language: str, speak: Callable[[str, str], AudioBuffer]
    ) -> None:
        self._tts = tts
        self._language = language
        self._speak = speak
        self._lock = threading.Lock()
        self._text = ""
        self._piece: str | None = None
        self._audio: Future[AudioBuffer] | None = None

    def begin(self) -> None:
        with self._lock:
            self._text, self._piece, self._audio = "", None, None

    def feed(self, text: str) -> None:
        with self._lock:
            if self._audio is not None:
                return
            self._text += text
            piece = self._tts.first_piece(self._text, self._language)
            if piece is None:
                return
            self._piece = piece
            language, speak = self._language, self._speak
            self._audio = in_background(lambda: speak(piece, language))

    def take(self, piece: str) -> AudioBuffer | None:
        """The audio made early, if it was made for exactly `piece`."""
        with self._lock:
            audio, early = self._audio, self._piece
        if audio is None:
            return None
        if early != piece:
            log.info("the reply's first piece changed after it was begun; making it again")
            return None
        try:
            return audio.result()
        except Exception as exc:  # made again below, the usual way, and reported there if it fails
            log.warning("the first piece begun early failed: %s", type(exc).__name__)
            return None


class VoicePipeline:
    """Runs one conversational turn end to end."""

    def __init__(
        self,
        asr: ASREngine,
        agent: Agent,
        tts: TTSEngine,
        speak: bool = True,
    ) -> None:
        self.asr = asr
        self.agent = agent
        self.tts = tts
        self.speak = speak
        self._early: _EarlyFirstPiece | None = None

    def converse(self, audio: AudioBuffer, conversation: Conversation) -> TurnResult:
        """Audio in, audio out."""
        tracker = LatencyTracker()

        with tracker.stage("vad"):
            trimmed = trim_silence(audio)
            heard_something = not is_silent(trimmed)

        # Guard here, not only inside the ASR engine. A new engine that forgets
        # to check would otherwise let silence through, and running a speech
        # model on room tone is both wasted time and an invitation to
        # hallucinate. Checking first also skips the expensive call entirely.
        if not heard_something:
            return self._nothing_heard(
                Transcript(text="", confidence=0.0, duration_s=audio.duration_s),
                conversation,
                tracker,
            )

        checkpoint("speech recognition")
        try:
            with tracker.stage("asr"):
                transcript = self.asr.transcribe(trimmed, language=conversation.language)
        except ASRUnavailable:
            # Not "I didn't catch that": the customer said something and
            # deserves to know it was the service, not them -- and that they
            # can type instead. Reported where it failed, with its code.
            return self._say(
                say("speech_unavailable", conversation.language),
                Transcript(text="", confidence=0.0, duration_s=audio.duration_s),
                tracker,
                key="speech_unavailable",
            )

        # Nothing was actually said. Short-circuit rather than spend a remote
        # round trip asking a model what "..." means.
        if not _HAS_CONTENT.search(transcript.text or ""):
            return self._nothing_heard(transcript, conversation, tracker)

        _hear_language(transcript, conversation)
        return self._finish(transcript, transcript.text.strip(), conversation, tracker)

    def respond_to_heard(
        self, transcript: Transcript, conversation: Conversation, asr_ms: float = 0.0
    ) -> TurnResult:
        """Same turn, on speech already recognised.

        The live session recognises an utterance itself, so the caller sees
        their words before the reply, and an interruption can be told from a
        "mm-hmm". The rest is the turn every other path takes.
        """
        tracker = LatencyTracker()
        tracker.stages["asr"] = asr_ms
        if not _HAS_CONTENT.search(transcript.text or ""):
            return self._nothing_heard(transcript, conversation, tracker)
        _hear_language(transcript, conversation)
        return self._finish(transcript, transcript.text.strip(), conversation, tracker)

    def respond_to_text(self, text: str, conversation: Conversation) -> TurnResult:
        """Same turn, skipping the microphone.

        Used by tests and by the API's text endpoint. Going through the same
        path keeps one implementation of the turn rather than two that drift.
        """
        tracker = LatencyTracker()
        transcript = Transcript(text=text, confidence=1.0, duration_s=0.0)

        if not _HAS_CONTENT.search(text or ""):
            return self._nothing_heard(transcript, conversation, tracker)

        # Typed Urdu is often in Latin letters, "mera balance kitna hai", and
        # is answered in Urdu like the spoken kind -- in Urdu script, so the
        # Urdu voice can read it.
        conversation.language = typed_language(text, conversation.language)  # type: ignore[arg-type]
        return self._finish(transcript, text.strip(), conversation, tracker)

    # --- internals -----------------------------------------------------------

    def _finish(
        self,
        transcript: Transcript,
        text: str,
        conversation: Conversation,
        tracker: LatencyTracker,
    ) -> TurnResult:
        checkpoint("the model")
        with tracker.stage("agent"), self._speaking_early(conversation.language):
            result = self._safely(lambda: self.agent.handle(text, conversation), conversation)

        checkpoint("speaking")
        with tracker.stage("tts"):
            audio, rest, pieces = self._speak_reply(result.text, conversation.language)

        turn = TurnResult(
            transcript=transcript,
            text=text,
            agent=result,
            audio=audio,
            latency=tracker.breakdown(),
            rest=rest,
            pieces=pieces,
        )
        self._log(turn, conversation.language)
        return turn

    def resolve_pending(self, confirmed: bool, conversation: Conversation) -> TurnResult:
        """The Yes or Cancel button: act, then let the model say what happened."""
        tracker = LatencyTracker()
        transcript = Transcript(text="yes" if confirmed else "no", confidence=1.0, duration_s=0.0)

        with tracker.stage("agent"), self._speaking_early(conversation.language):
            result = self._safely(
                lambda: self.agent.resolve_pending(confirmed, conversation), conversation
            )

        checkpoint("speaking")
        with tracker.stage("tts"):
            audio, rest, pieces = self._speak_reply(result.text, conversation.language)

        turn = TurnResult(
            transcript=transcript,
            text=transcript.text,
            agent=result,
            audio=audio,
            latency=tracker.breakdown(),
            rest=rest,
            pieces=pieces,
        )
        self._log(turn, conversation.language)
        return turn

    @contextlib.contextmanager
    def _speaking_early(self, language: str) -> Iterator[None]:
        """While the model writes, its reply's first piece goes to the voice.

        The model's words are streamed, and once they settle the first piece
        it is synthesised while the rest is still being written. It is heard
        only if the reply that passes every guard starts with exactly that
        piece (`_speak_reply`): nothing is spoken before all of it is checked.
        """
        if not self.speak:
            yield
            return
        self._early = _EarlyFirstPiece(self.tts, language, self._speak)
        with listening(self._early):
            yield

    def _safely(self, work: Callable[[], AgentResult], conversation: Conversation) -> AgentResult:
        """Run the agent; anything it did not expect ends the turn in a sentence.

        Its own failures it already says well. This is for the rest -- a bug,
        a shape nobody planned for -- which would otherwise reach the caller
        as a 500 and silence. Cancelled goes straight up: there is no one left
        to say anything to.
        """
        try:
            return work()
        except Cancelled:
            raise
        except Exception as exc:  # a bug, not a provider failure: said as internal
            log.error("the turn failed unexpectedly: %s", type(exc).__name__, exc_info=exc)
            report(
                InternalError(
                    f"the turn failed: {type(exc).__name__}", component="agent", cause=exc
                )
            )
            return AgentResult(text=say("internal", conversation.language), fallback="internal")

    def _speak_reply(
        self, text: str, language: str
    ) -> tuple[AudioBuffer, SpokenReply | None, tuple[str, ...]]:
        """The reply's first piece now, and the rest being made in the background.

        The customer hears the first piece as soon as it exists rather than
        waiting for the whole reply. If the voice cannot make even that, the
        bot's own recording -- made earlier, in its own voice -- says the
        voice is not working, and the answer is on the screen.
        """
        if not self.speak:
            return AudioBuffer(samples=SILENCE, sample_rate=16_000), None, ()
        try:
            pieces = self.tts.pieces(text, voice=language)
        except Exception as exc:  # cutting it up failed: say it whole
            report(InternalError(f"the reply could not be cut: {type(exc).__name__}", cause=exc))
            pieces = [text]
        early, self._early = self._early, None
        made = early.take(pieces[0]) if early is not None else None
        first = made if made is not None else self._speak(pieces[0], language)
        if not len(first.samples) and text.strip():
            kept = self.tts.kept("voice_unavailable", language)
            if kept is not None:
                log.warning("the voice failed; saying so in the bot's own recorded voice")
                return kept, None, (say("voice_unavailable", language),)
            return first, None, ()
        if len(pieces) == 1:
            return first, None, tuple(pieces)
        rest = SpokenReply(pieces[1:], lambda piece: self._speak(piece, language))
        return first, rest, tuple(pieces)

    def _speak(self, text: str, language: str | None = None) -> AudioBuffer:
        """`text` in the voice of `language` -- the customer's, never the reply's own script.

        Without a language (a sentence of the bot's own, already in the right
        one) the engine goes by the text's script.
        """
        if not self.speak:
            return AudioBuffer(samples=SILENCE, sample_rate=16_000)
        try:
            return self.tts.synthesize(text, voice=language if language in ("en", "ur") else None)
        except Cancelled:
            # Nobody is left to hear it: not a failure, and not reported as one.
            raise
        except Exception as exc:  # an engine that raises is treated as silence
            # An engine that raises rather than returning silence is treated
            # as silence: the reply is still on the page.
            report(
                InternalError(f"the voice raised {type(exc).__name__}", component="tts", cause=exc)
            )
            return AudioBuffer(samples=SILENCE, sample_rate=16_000)

    def _nothing_heard(
        self, transcript: Transcript, conversation: Conversation, tracker: LatencyTracker
    ) -> TurnResult:
        """Silence, or a transcript with nothing in it. The model is not called."""
        return self._say(
            say("not_caught", conversation.language), transcript, tracker, key="not_caught"
        )

    def _say(
        self,
        sentence: str,
        transcript: Transcript,
        tracker: LatencyTracker,
        key: str | None = None,
    ) -> TurnResult:
        """A turn the bot answers with a sentence of its own, without the model."""
        with tracker.stage("agent"):
            pass  # named so every turn reports the same four stages
        with tracker.stage("tts"):
            audio = self._speak(sentence)
        return TurnResult(
            transcript=transcript,
            text="",
            agent=AgentResult(text=sentence, fallback=key),
            audio=audio,
            latency=tracker.breakdown(),
            pieces=(sentence,),
        )

    # --- for the conversation's own rules, around a turn ------------------------

    def say(self, sentence: str, key: str | None = None, heard: str = "") -> TurnResult:
        """A whole turn that is one of the bot's own sentences."""
        return self._say(
            sentence,
            Transcript(text=heard, confidence=1.0, duration_s=0.0),
            LatencyTracker(),
            key=key,
        )

    def say_after(self, result: TurnResult, sentence: str) -> TurnResult:
        """`result`, with `sentence` said after its reply."""
        agent = dataclasses.replace(result.agent, text=f"{result.agent.text} {sentence}".strip())
        if result.rest is not None:
            # Still being spoken a piece at a time: it goes on the end.
            result.rest.add(sentence)
            return dataclasses.replace(result, agent=agent, pieces=(*result.pieces, sentence))
        extra = self._speak(sentence)
        audio = result.audio
        if len(audio.samples) and len(extra.samples) and extra.sample_rate == audio.sample_rate:
            audio = AudioBuffer(
                samples=np.concatenate([audio.samples, extra.samples]),
                sample_rate=audio.sample_rate,
            )
        # One clip now, so one piece: both sentences in it.
        joined = " ".join((*result.pieces, sentence)).strip()
        return dataclasses.replace(result, agent=agent, audio=audio, pieces=(joined,))

    def replace_reply(self, result: TurnResult, agent: AgentResult) -> TurnResult:
        """`result`, answered with `agent`'s reply instead, spoken afresh."""
        if result.rest is not None:
            result.rest.cancel()
        return dataclasses.replace(
            result, agent=agent, audio=self._speak(agent.text), rest=None, pieces=(agent.text,)
        )

    def _log(self, result: TurnResult, language: str) -> None:
        slowest = result.latency.slowest()
        # What was said is the customer's, and is logged only at DEBUG. At
        # INFO a turn is its shape: how much was said, in which language,
        # what was done, how long it took to the first sound (the turn's
        # total: it answers with the first piece), how many pieces followed,
        # and whether it ended in a fallback.
        log.info(
            "turn language=%s chars=%d tools=%s guard_hit=%s fallback=%s first_audio=%.0fms "
            "pieces_after=%d slowest=%s",
            language,
            len(result.text),
            [call.name for call in result.agent.tool_calls] or "none",
            result.agent.guard_hit,
            result.agent.fallback or "-",
            result.latency.total_ms,
            result.rest.count if result.rest is not None else 0,
            f"{slowest[0]} {slowest[1]:.0f}ms" if slowest else "n/a",
        )
        log.debug("heard=%r replied=%r", result.text, result.agent.text)
