"""Deterministic seeding across every library the project uses."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) torch.

    Called at the start of every training and evaluation run. The seed is
    recorded in each result file so a run can be reproduced exactly.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
