import pytest

from voicebot.utils.registry import available, create, register


def test_registered_class_can_be_created_by_name():
    @register("widget", "spinner")
    class Spinner:
        def __init__(self, turns: int = 1):
            self.turns = turns

    obj = create("widget", "spinner", turns=3)
    assert obj.turns == 3


def test_unknown_name_error_lists_available_names():
    @register("gadget", "alpha")
    class Alpha:
        pass

    @register("gadget", "beta")
    class Beta:
        pass

    with pytest.raises(KeyError) as exc:
        create("gadget", "gamma")
    message = str(exc.value)
    assert "gamma" in message
    assert "alpha" in message and "beta" in message


def test_unknown_kind_error_names_the_kind():
    with pytest.raises(KeyError, match="nosuchkind"):
        create("nosuchkind", "anything")


def test_available_returns_sorted_names():
    @register("tool", "zeta")
    class Zeta:
        pass

    @register("tool", "alpha")
    class AlphaTool:
        pass

    assert available("tool") == ["alpha", "zeta"]


def test_duplicate_registration_is_rejected():
    @register("dup", "one")
    class One:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @register("dup", "one")
        class OneAgain:
            pass
