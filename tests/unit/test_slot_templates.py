import pytest

from voicebot.nlu.slot_templates import SLOT_NAMES, SlotExample, generate, slot_tag_vocab


@pytest.fixture(scope="module")
def corpus() -> list[SlotExample]:
    return generate(400, seed=0)


def test_generate_returns_requested_count(corpus):
    assert len(corpus) == 400


def test_tokens_and_tags_are_the_same_length(corpus):
    for ex in corpus:
        assert len(ex.tokens) == len(ex.tags), ex.text


def test_text_is_tokens_joined_by_single_spaces(corpus):
    for ex in corpus:
        assert ex.text == " ".join(ex.tokens)


def test_every_tag_is_in_the_declared_vocabulary(corpus):
    vocab = set(slot_tag_vocab())
    for ex in corpus:
        assert set(ex.tags) <= vocab, set(ex.tags) - vocab


def test_no_inside_tag_follows_o_or_a_different_slot_type(corpus):
    for ex in corpus:
        previous = "O"
        for tag in ex.tags:
            if tag.startswith("I-"):
                slot = tag[2:]
                assert previous in (f"B-{slot}", f"I-{slot}"), f"{ex.tags} in {ex.text!r}"
            previous = tag


def test_multi_word_slot_values_produce_b_then_i(corpus):
    found = False
    for ex in corpus:
        for index, tag in enumerate(ex.tags):
            if tag.startswith("B-") and index + 1 < len(ex.tags):
                if ex.tags[index + 1].startswith("I-"):
                    found = True
                    assert ex.tags[index + 1][2:] == tag[2:]
    assert found, "corpus should contain at least one multi-word slot value"


def test_generation_is_deterministic_for_a_given_seed():
    assert [ex.text for ex in generate(30, seed=5)] == [ex.text for ex in generate(30, seed=5)]


def test_different_seeds_produce_different_corpora():
    assert [ex.text for ex in generate(30, seed=1)] != [ex.text for ex in generate(30, seed=2)]


def test_every_declared_slot_name_appears_somewhere(corpus):
    seen = {tag[2:] for ex in corpus for tag in ex.tags if tag != "O"}
    assert seen == set(SLOT_NAMES)


def test_intents_are_non_empty_strings(corpus):
    assert all(ex.intent and isinstance(ex.intent, str) for ex in corpus)
