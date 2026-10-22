import random

import numpy as np

from voicebot.utils.seed import set_seed


def test_set_seed_makes_python_random_deterministic():
    set_seed(42)
    first = [random.random() for _ in range(5)]
    set_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_set_seed_makes_numpy_deterministic():
    set_seed(7)
    first = np.random.rand(5).tolist()
    set_seed(7)
    second = np.random.rand(5).tolist()
    assert first == second


def test_set_seed_returns_none_and_accepts_zero():
    assert set_seed(0) is None
