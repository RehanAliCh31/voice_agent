"""Deterministic seeding across every library the project uses."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python and NumPy.

    Called at the start of every training and evaluation run. The seed is
    recorded in each result file so a run can be reproduced exactly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
